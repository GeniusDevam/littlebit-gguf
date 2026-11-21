import argparse
import ctypes
import json
import logging
import math
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from littlebit import (
    BLOCK_SIZE,
    FIELDS,
    field_name,
    fp16_scale,
    prefixes,
    q1_blocks,
    read_legacy,
    read_path,
)

ROOT = Path(__file__).resolve().parent
LLAMA = ROOT / "third_party" / "llama.cpp"
sys.path[:0] = [str(LLAMA / "gguf-py"), str(LLAMA)]

import gguf  # noqa: E402
from conversion.llama import LlamaModel  # noqa: E402
from conversion.qwen import Qwen2Model, Qwen3Model  # noqa: E402

Q = gguf.GGMLQuantizationType
TYPES = {"f16": Q.F16, "q8_0": Q.Q8_0, "q6_k": Q.Q6_K, "q4_k": Q.Q4_K}
ARCHITECTURES = {
    "LlamaForCausalLM": LlamaModel,
    "Qwen2ForCausalLM": Qwen2Model,
    "Qwen3ForCausalLM": Qwen3Model,
}
EMBEDDING = "token_embd.weight"
OUTPUT = "output.weight"
DENSE_CHUNK_ROWS = 1024


def dense_type(name: str, width: int, requested: Q) -> Q:
    block, _ = gguf.GGML_QUANT_SIZES[requested]
    if width % block == 0:
        return requested
    fallback = {Q.Q4_K: Q.Q5_0, Q.Q6_K: Q.Q8_0}.get(requested, Q.F16)
    if width % gguf.GGML_QUANT_SIZES[fallback][0]:
        fallback = Q.F16
    print(
        f"{name}: width {width} is incompatible with {requested.name}; using {fallback.name}"
    )
    return fallback


def k_quantizer(qtype: Q, binary_dir: Path) -> Callable[..., int]:
    lib_path = binary_dir / "libggml-base.so"
    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise ValueError(
            f"build the CPU tools first to use {qtype.name}: {lib_path}"
        ) from exc
    quantize = lib.ggml_quantize_chunk
    quantize.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    quantize.restype = ctypes.c_size_t
    return quantize


class LittleBitMixin:
    options: argparse.Namespace
    model_tensors: dict[str, Callable[[], torch.Tensor]]
    hparams: dict[str, Any]
    gguf_writer: gguf.GGUFWriter
    block_count: int

    def index_tensors(
        self, remote_hf_model_id: str | None = None
    ) -> dict[str, Callable[[], torch.Tensor]]:
        legacy = self.options.checkpoint / "model_state.pt"
        if legacy.is_file() and not list(legacy.parent.glob("model*.safetensors*")):
            return {
                name: (lambda tensor=tensor: tensor)
                for name, tensor in read_legacy(legacy.parent).items()
            }
        return super().index_tensors(remote_hf_model_id)

    def write_dense(self, name: str, tensor: torch.Tensor, qtype: Q) -> None:
        if tensor.ndim != 2 or not all(tensor.shape):
            raise ValueError(f"{name} must be a nonempty matrix")
        writer = self.gguf_writer
        spool = writer.temp_file
        assert spool is not None
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        height, width = tensor.shape
        row_bytes = width // block * size
        nbytes = height * row_bytes
        quantize = None
        if qtype in (Q.Q4_K, Q.Q6_K):
            quantize = k_quantizer(qtype, self.options.binary_dir)

        for offset in range(0, height, DENSE_CHUNK_ROWS):
            rows = tensor[offset : offset + DENSE_CHUNK_ROWS]
            if qtype == Q.F16 and rows.dtype == torch.float16:
                chunk = rows.contiguous().numpy()
            elif quantize is None:
                chunk = gguf.quantize(rows.float().numpy(), qtype)
            else:
                source = rows.float().contiguous().numpy()
                chunk = np.empty((len(rows), row_bytes), dtype=np.uint8)
                count = quantize(
                    qtype, source.ctypes.data, chunk.ctypes.data, 0, *source.shape, None
                )
                if count != chunk.nbytes:
                    raise ValueError(
                        f"{name}: {qtype.name} wrote {count} bytes, expected {chunk.nbytes}"
                    )
            if offset == 0:
                writer.add_tensor_info(
                    name, (height, chunk.shape[1]), chunk.dtype, nbytes, qtype
                )
            chunk.tofile(spool)
        writer.write_padding(spool, nbytes)

    def prepare_tensors(self) -> None:
        layers = prefixes(self.block_count)
        options = self.options
        selected = set(
            np.random.default_rng(0).choice(layers, options.validate, replace=False)
        )
        # The base converter promotes whole BF16 tensors to F32.
        embedding = self.model_tensors.pop("model.embed_tokens.weight")
        output = self.model_tensors.pop("lm_head.weight", None)
        if output is None and not self.hparams.get("tie_word_embeddings", False):
            raise ValueError("untied model is missing lm_head.weight")
        self.gguf_writer.add_uint32("littlebit.format_version", 1)
        self.gguf_writer.add_bool("littlebit.residual", options.residual)
        self.gguf_writer.add_float32("littlebit.eff_bits", options.eff_bit)
        self.gguf_writer.add_bool("littlebit.use_itq", options.use_itq)
        rng = np.random.default_rng(1)
        self.compressed_bytes = self.compressed_weights = 0
        padded_weights = flushed_count = 0
        largest_flushed = 0.0
        max_relative = 0.0
        for prefix in layers:
            dense_name = prefix + ".weight"
            if dense_name in self.model_tensors:
                raise ValueError(
                    f"{prefix}: dense weight and LittleBit factors cannot coexist"
                )
            base = self.map_tensor_name(dense_name).removesuffix(".weight")
            reference = actual = 0.0
            used_bytes = 0
            main_shape = None
            for residual in range(1 + options.residual):
                names = [field_name(prefix, field, bool(residual)) for field in FIELDS]
                missing = set(names) - self.model_tensors.keys()
                if missing:
                    raise ValueError(f"missing tensors: {', '.join(sorted(missing))}")
                factors = read_path(
                    lambda name: self.model_tensors[name](), prefix, bool(residual)
                )
                for name in names:
                    del self.model_tensors[name]
                out, rank, width = len(factors.h), len(factors.latent), len(factors.g)
                if main_shape is not None and main_shape != (out, rank, width):
                    raise ValueError(
                        f"{prefix}: residual dimensions differ from primary"
                    )
                main_shape = (out, rank, width)
                if width % BLOCK_SIZE:
                    raise ValueError(
                        f"{prefix}: input width {width} is not a multiple of {BLOCK_SIZE}"
                    )
                padded = (rank + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE

                # The Llama converter reorders Q/K rows for ggml's RoPE layout.
                order = None
                if isinstance(self, LlamaModel) and prefix.endswith(
                    (".q_proj", ".k_proj")
                ):
                    heads = self.hparams["num_attention_heads"]
                    kv_heads = self.hparams.get("num_key_value_heads", heads)
                    order = self.permute(
                        torch.arange(out),
                        heads,
                        kv_heads if prefix.endswith(".k_proj") else heads,
                    ).numpy()
                try:
                    h, h_error, h_count, h_max = fp16_scale(factors.h)
                    latent, l_error, l_count, l_max = fp16_scale(factors.latent)
                except ValueError as exc:
                    raise ValueError(f"{prefix}: {exc}") from exc
                max_relative = max(max_relative, h_error, l_error)
                flushed_count += h_count + l_count
                largest_flushed = max(largest_flushed, h_max, l_max)
                padded_weights += (padded - rank) * (width + out)
                v = q1_blocks(
                    factors.v, np.pad(latent, (0, padded - rank)), width, width
                )
                u = q1_blocks(
                    factors.u if order is None else factors.u[order],
                    h if order is None else h[order],
                    padded,
                    rank,
                )
                suffix = "_r" if residual else ""
                self.gguf_writer.add_tensor(f"{base}.lb_g{suffix}", factors.g)
                self.gguf_writer.add_tensor(f"{base}.lb_v{suffix}", v, raw_dtype=Q.Q1_0)
                self.gguf_writer.add_tensor(f"{base}.lb_u{suffix}", u, raw_dtype=Q.Q1_0)
                used_bytes += factors.g.nbytes + v.nbytes + u.nbytes
                if prefix in selected:
                    if not residual:
                        x = rng.standard_normal((4, width), dtype=np.float32)
                    expected = factors.forward(x)
                    reference += expected if order is None else expected[:, order]
                    first = (x * factors.g) @ gguf.dequantize(v, Q.Q1_0).T
                    actual += first @ gguf.dequantize(u, Q.Q1_0).T
            for name in ("_eff_bit_target", "_split_dim_final", "_eff_bit_actual"):
                self.model_tensors.pop(f"{prefix}.{name}", None)
            self.compressed_bytes += used_bytes
            self.compressed_weights += width * out
            padding = (
                " (over half the padded rank is unused)" if rank * 2 < padded else ""
            )
            print(
                f"{base}: rank {rank}/{padded}, {used_bytes * 8 / (width * out):.4f} bpw{padding}"
            )
            if prefix in selected:
                error = np.abs(actual - reference)
                print(
                    f"  validation: max abs {error.max():.6g}, RMSE {np.sqrt((error**2).mean()):.6g}"
                )
        print(f"Largest relative fp16 scale change: {max_relative:.6g}")
        print(
            f"Flushed {flushed_count:,} nonzero scales to fp16 zero "
            f"(largest magnitude {largest_flushed:.6g})"
        )
        block, size = gguf.GGML_QUANT_SIZES[Q.Q1_0]
        padding_bytes = padded_weights * size / block
        print(
            f"Rank padding: {padding_bytes:,.0f} projection bytes "
            f"(+{padding_bytes / (self.compressed_bytes - padding_bytes):.2%} over unpadded)"
        )
        tensor = embedding()
        embd_type = dense_type(EMBEDDING, tensor.shape[-1], TYPES[options.embd_type])
        self.write_dense(EMBEDDING, tensor, embd_type)
        if output is not None:
            tensor = output()
        if output is not None or options.output_type != options.embd_type:
            out_type = dense_type(OUTPUT, tensor.shape[-1], TYPES[options.output_type])
            if output is not None or out_type != embd_type:
                self.write_dense(OUTPUT, tensor, out_type)
        del tensor
        super().prepare_tensors()


def convert(options: argparse.Namespace) -> None:
    checkpoint = options.checkpoint
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    extra = checkpoint / "littlebit_config.json"
    metadata = dict(config)
    if extra.is_file():
        metadata.update(json.loads(extra.read_text(encoding="utf-8")))
    if options.eff_bit is None:
        options.eff_bit = metadata.get("eff_bit")
    if options.residual is None:
        options.residual = metadata.get("residual")
    if options.use_itq is None:
        options.use_itq = metadata.get("use_itq", False)
    if (
        options.eff_bit is None
        or not math.isfinite(options.eff_bit)
        or options.eff_bit <= 0
    ):
        raise ValueError(
            "supply a positive eff_bit in the checkpoint config or --eff-bit"
        )
    if not isinstance(options.residual, bool) or not isinstance(options.use_itq, bool):
        raise ValueError(
            "supply boolean residual/use_itq fields or the corresponding flags"
        )
    if options.output_type is None:
        options.output_type = options.embd_type
    arch = config["architectures"][0]
    if arch not in ARCHITECTURES:
        raise ValueError(f"unsupported architecture: {arch}")
    if not 0 <= options.validate <= len(prefixes(config["num_hidden_layers"])):
        raise ValueError(
            "--validate must be between zero and the number of projections"
        )

    base_model = ARCHITECTURES[arch]

    class Converter(LittleBitMixin, base_model):
        model_arch = base_model.model_arch

    output = options.outfile.resolve()
    with tempfile.TemporaryDirectory(prefix=".littlebit-", dir=output.parent) as temp:
        Converter.options = options
        temporary = Path(temp) / output.name
        model = Converter(
            checkpoint,
            gguf.LlamaFileType.MOSTLY_Q1_0,
            temporary,
            eager=True,
            use_temp_file=True,
        )
        with tempfile.SpooledTemporaryFile(
            max_size=16 * 1024 * 1024, dir=temp
        ) as spool:
            model.gguf_writer.temp_file = spool
            model.write()
        reader = gguf.GGUFReader(temporary)
        sizes = {tensor.name: tensor.n_bytes for tensor in reader.tensors}
        dense_bytes = sum(sizes.get(name, 0) for name in (EMBEDDING, OUTPUT))
        total = temporary.stat().st_size
        print(
            f"Compressed projections: {model.compressed_bytes * 8 / model.compressed_weights:.4f} bpw"
        )
        print(
            f"File: {total:,} bytes; embeddings and output: {dense_bytes:,} bytes ({dense_bytes / total:.1%})"
        )
        del reader
        temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LittleBit checkpoint into GGUF."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--embd-type", choices=TYPES, default="f16")
    parser.add_argument("--output-type", choices=TYPES)
    parser.add_argument("--validate", type=int, default=0)
    parser.add_argument("--eff-bit", type=float)
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-itq", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--binary-dir",
        type=Path,
        default=Path(os.environ.get("LITTLEBIT_BUILD_DIR", ROOT / "build")) / "bin",
    )
    options = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        convert(options)
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(1, f"conversion failed: {exc}\n")


if __name__ == "__main__":
    main()
