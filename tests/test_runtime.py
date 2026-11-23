import argparse
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tiny_model import make_tiny_littlebit, make_tiny_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from convert_littlebit_to_gguf import LLAMA, ROOT, convert, gguf
from littlebit import prefixes, read_path, unpack_signs
from tools.make_synthetic import make_checkpoint

DATA = Path(os.environ.get("LITTLEBIT_TEST_DATA", ROOT / "checkpoints"))
BUILD = Path(os.environ.get("LITTLEBIT_BUILD_DIR", ROOT / "build"))
PROMPTS = (
    "The quick brown fox jumps over the lazy dog.",
    "Packing binary weights is fun.",
)


def q8_0_activations(x: torch.Tensor) -> torch.Tensor:
    # AVX2 uses 32 values per block, nearest-even integers and an FP16 scale.
    width = x.size(-1)
    blocks = F.pad(x, (0, -width % 32)).reshape(*x.shape[:-1], -1, 32)
    peak = blocks.abs().amax(-1, keepdim=True)
    limit = torch.full_like(peak, 127)
    inverse = torch.where(peak == 0, 0, limit / peak)
    return ((blocks * inverse).round() * (peak / limit).half().float()).flatten(-2)[
        ..., :width
    ]


def test_q8_0_activations_at_rounding_boundary() -> None:
    x = torch.tensor([0.14, 0.0016535432077944279], dtype=torch.float32)
    scale = torch.tensor(0.14 / 127, dtype=torch.float16).float()
    assert q8_0_activations(x)[1].item() == 2 * scale.item()


class FloatProjection(torch.nn.Module):
    def __init__(
        self, state: dict[str, torch.Tensor], prefix: str, residual: bool, q8_0: bool
    ) -> None:
        super().__init__()
        self.paths = []
        self.q8_0 = q8_0
        for extra in range(1 + residual):
            path = read_path(state.__getitem__, prefix, bool(extra))
            self.paths.append(
                tuple(
                    torch.from_numpy(value)
                    for value in (
                        unpack_signs(path.u, len(path.latent)).astype(np.int8),
                        unpack_signs(path.v, len(path.g)).astype(np.int8),
                        path.h,
                        path.latent,
                        path.g,
                    )
                )
            )
        self.bias = state.get(f"{prefix}.bias")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for u, v, h, latent, g in self.paths:
            z = x * g.to(x.dtype)
            if self.q8_0:
                z = q8_0_activations(z)
                h, latent = h.half(), latent.half()
            z = F.linear(z, v.to(x.dtype)) * latent.to(x.dtype)
            if self.q8_0:
                z = q8_0_activations(z)
            outputs.append(F.linear(z, u.to(x.dtype)) * h.to(x.dtype))
        result = sum(outputs)
        return result if self.bias is None else result + self.bias.to(x.dtype)


def reference_model(
    checkpoint: Path, dtype: torch.dtype = torch.float32, *, q8_0: bool = False
) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(checkpoint)
    state = load_file(checkpoint / "model.safetensors")
    metadata = json.loads(
        (checkpoint / "littlebit_config.json").read_text(encoding="utf-8")
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16).eval()
    for prefix in prefixes(config.num_hidden_layers):
        model.set_submodule(
            prefix, FloatProjection(state, prefix, metadata["residual"], q8_0)
        )
    model.to(dtype)
    dense_names = model.state_dict().keys()
    dense = {name: tensor for name, tensor in state.items() if name in dense_names}
    missing, unexpected = model.load_state_dict(dense, strict=False)
    allowed = ["lm_head.weight"] if config.tie_word_embeddings else []
    assert not unexpected and not set(missing).difference(allowed)
    return model


@pytest.fixture(scope="session")
def logits_tool(tmp_path_factory: pytest.TempPathFactory) -> Path:
    binary_dir = BUILD / "bin"
    if not (binary_dir / "libllama.so").is_file():
        pytest.skip(f"build CPU tools in {BUILD} to run runtime tests")
    output = tmp_path_factory.mktemp("tools") / "read-logits"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            str(ROOT / "tests" / "read-logits.cpp"),
            "-I",
            str(LLAMA / "include"),
            "-I",
            str(LLAMA / "ggml" / "include"),
            "-L",
            str(binary_dir),
            f"-Wl,-rpath,{binary_dir}",
            "-lllama",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


def logits(
    tool: Path, model: Path, ids: list[int], threads: int, kv_type: str
) -> torch.Tensor:
    result = subprocess.run(
        [str(tool), str(model), str(threads), kv_type, *map(str, ids)],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return torch.from_numpy(
        np.frombuffer(result.stdout, dtype="<f4").copy().reshape(len(ids), -1)
    )


@torch.inference_mode()
def reference_logits(
    checkpoint: Path,
    token_ids: list[list[int]],
    threads: int,
    dtype: torch.dtype = torch.float32,
    *,
    q8_0: bool = False,
) -> list[torch.Tensor]:
    torch.set_num_threads(threads)
    reference = reference_model(checkpoint, dtype, q8_0=q8_0)
    return [reference(torch.tensor([ids])).logits[0] for ids in token_ids]


def prompt_ids(checkpoint: Path) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return [tokenizer.encode(prompt) for prompt in PROMPTS]


def compare_logits(
    label: str, expected: list[torch.Tensor], actual: list[torch.Tensor]
) -> tuple[float, float, float]:
    metrics = []
    for ref, got in zip(expected, actual, strict=True):
        assert got.shape == ref.shape
        assert torch.isfinite(got).all() and torch.isfinite(ref).all()
        got, ref = got.double(), ref.double()
        agreement = (got.argmax(-1) == ref.argmax(-1)).double().mean().item()
        error = (got - ref).abs().max().item()
        kl = F.kl_div(
            got.log_softmax(-1), ref.softmax(-1), reduction="batchmean"
        ).item()
        metrics.append((agreement, error, kl))
    agreement = sum(row[0] for row in metrics) / len(metrics)
    error = max(row[1] for row in metrics)
    kl = sum(row[2] for row in metrics) / len(metrics)
    print(f"{label}: top-1 {agreement:.4f}, max abs {error:.6f}, KL {kl:.6g}")
    return agreement, error, kl


@pytest.mark.parametrize(
    "architecture,bits",
    [(arch, bits) for arch in ("llama", "qwen2", "qwen3") for bits in (0.1, 1.0)],
)
def test_tiny_runtime(
    tmp_path: Path,
    logits_tool: Path,
    architecture: str,
    bits: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, checkpoint, gguf_path = (
        tmp_path / name for name in ("source", "lb", "tiny.gguf")
    )
    torch.set_num_threads(2)
    monkeypatch.setattr("convert_littlebit_to_gguf.DENSE_CHUNK_ROWS", 7)
    make_tiny_model(source, architecture=architecture)
    make_checkpoint(str(source), checkpoint, bits, True, 0)
    options = argparse.Namespace(
        checkpoint=checkpoint,
        outfile=gguf_path,
        embd_type="f16",
        output_type=None,
        validate=2,
        eff_bit=None,
        residual=None,
        use_itq=None,
        binary_dir=BUILD / "bin",
    )
    convert(options)
    token_ids = prompt_ids(checkpoint)
    expected = reference_logits(checkpoint, token_ids, 2)
    for kv_type in ("f16", "q8_0"):
        actual = [logits(logits_tool, gguf_path, ids, 2, kv_type) for ids in token_ids]
        top1, error, kl = compare_logits(
            f"{architecture}-{bits} {kv_type}", expected, actual
        )
        assert top1 >= 0.95
        assert error < 0.1
        assert kl < 0.001

    if architecture == "llama" and bits == 0.1:
        options.outfile = tmp_path / "k-quants.gguf"
        options.embd_type, options.output_type = "q4_k", "q6_k"
        convert(options)
        reader = gguf.GGUFReader(options.outfile)
        types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
        assert types["token_embd.weight"] == gguf.GGMLQuantizationType.Q4_K
        assert types["output.weight"] == gguf.GGMLQuantizationType.Q6_K

    cli = BUILD / "bin" / "llama-cli"
    if architecture == "llama" and bits == 0.1 and cli.is_file():
        args = [
            str(cli),
            "-m",
            str(gguf_path),
            "-ngl",
            "0",
            "-t",
            "2",
            "-c",
            "256",
            "--single-turn",
            "--simple-io",
            "--no-warmup",
            "--ignore-eos",
            "-p",
            "Packing binary weights",
            "-n",
            "3",
        ]
        smoke = subprocess.run(
            args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60
        )
        assert smoke.returncode == 0, smoke.stderr
        for suffix, width in (("weight", 256), ("lb_u", 128)):
            adapter = tmp_path / f"adapter-{suffix}.gguf"
            writer = gguf.GGUFWriter(adapter, "llama")
            writer.add_type("adapter")
            writer.add_string("adapter.type", "lora")
            writer.add_float32("adapter.lora.alpha", 1.0)
            writer.add_tensor(
                f"blk.0.attn_q.{suffix}.lora_a", np.zeros((2, width), dtype=np.float32)
            )
            writer.add_tensor(
                f"blk.0.attn_q.{suffix}.lora_b", np.zeros((256, 2), dtype=np.float32)
            )
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            rejected = subprocess.run(
                [*args, "--lora", str(adapter)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert rejected.returncode != 0
            assert "targets a LittleBit projection" in rejected.stderr


@pytest.mark.slow
@pytest.mark.parametrize("model", ["qwen2", "qwen3"])
@pytest.mark.parametrize("bits", ["1.0", "0.1"])
def test_published_model(model: str, bits: str, logits_tool: Path) -> None:
    checkpoint, gguf_path = DATA / f"{model}-{bits}", DATA / f"{model}-{bits}-f16.gguf"
    if not checkpoint.is_dir() or not gguf_path.is_file():
        pytest.skip(f"generate and convert {checkpoint} before the model test")
    token_ids = prompt_ids(checkpoint)
    f16, f16_no_fa, f32 = (
        [logits(logits_tool, gguf_path, ids, 4, kv_type) for ids in token_ids]
        for kv_type in ("f16", "f16-no-fa", "f32")
    )
    ref32 = reference_logits(checkpoint, token_ids, 4)
    ref64 = reference_logits(checkpoint, token_ids, 4, torch.float64)
    ref_q8 = reference_logits(checkpoint, token_ids, 4, q8_0=True)
    name = f"{model}-{bits}"
    compare_logits(f"{name} KV F32 to F16 (no flash)", f32, f16_no_fa)
    compare_logits(f"{name} KV F16 no flash to auto", f16_no_fa, f16)
    compare_logits(f"{name} PyTorch F64 to F32", ref64, ref32)
    compare_logits(f"{name} PyTorch F32 to Q8", ref32, ref_q8)
    q8_match = compare_logits(f"{name} PyTorch Q8 to KV F32", ref_q8, f32)
    compare_logits(f"{name} PyTorch Q8 to KV F16", ref_q8, f16)
    top1, error, kl = compare_logits(f"{name} PyTorch F32 to KV F16", ref32, f16)
    minimum_top1, maximum_error, maximum_kl = 0.8, 1.25, 0.03
    if model == "qwen3" and bits == "1.0":
        # The cache-only comparison gives max error 5.54 and KL 0.314.
        minimum_top1, maximum_error, maximum_kl = 0.44, 5.6, 0.32
        q8_top1, q8_error, q8_kl = q8_match
        assert q8_top1 >= 0.75
        assert q8_error < 3.5
        assert q8_kl < 0.07
    assert top1 >= minimum_top1
    assert error < maximum_error
    assert kl < maximum_kl


@pytest.mark.slow
def test_trained_8b_perplexity(tmp_path: Path) -> None:
    checkpoint = Path(
        os.environ.get("LITTLEBIT_8B_CHECKPOINT", DATA / "littlebit-qwen3-8b-0.55bpw")
    )
    wiki = Path(os.environ.get("LITTLEBIT_WIKITEXT2", DATA / "wikitext-2-test.txt"))
    binary_dir = BUILD / "bin"
    perplexity = binary_dir / "llama-perplexity"
    if not (checkpoint / "model.safetensors").is_file():
        pytest.skip("set LITTLEBIT_8B_CHECKPOINT to the local trained checkpoint")
    if not wiki.is_file():
        pytest.skip("set LITTLEBIT_WIKITEXT2 to the raw WikiText-2 test text")
    if not perplexity.is_file():
        pytest.skip(f"build CPU tools in {BUILD} to run the perplexity test")

    torch.set_num_threads(4)
    gguf_path = tmp_path / "trained-8b.gguf"
    convert(
        argparse.Namespace(
            checkpoint=checkpoint,
            outfile=gguf_path,
            embd_type="q4_k",
            output_type="q6_k",
            validate=3,
            eff_bit=None,
            residual=None,
            use_itq=None,
            binary_dir=binary_dir,
        )
    )
    result = subprocess.run(
        [
            str(perplexity),
            "-m",
            str(gguf_path),
            "-f",
            str(wiki),
            "-ngl",
            "0",
            "-t",
            "4",
            "-tb",
            "4",
            "-c",
            "2048",
            "-b",
            "2048",
            "-ub",
            "512",
            "--chunks",
            "4",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    windows = re.findall(r"\[(\d+)\](\d+\.\d+),", result.stdout)
    assert [index for index, _ in windows] == ["1", "2", "3", "4"], result.stdout
    assert float(windows[-1][1]) == pytest.approx(26.3547, abs=0.05), result.stdout


def convert_layouts(tmp_path: Path, ranks: tuple[int, ...]) -> tuple[Path, Path, Path]:
    checkpoint = tmp_path / "lb"
    make_tiny_littlebit(checkpoint, ranks)
    files = []
    for name, unpadded_v in (("padded", False), ("unpadded", True)):
        files.append(tmp_path / f"{name}.gguf")
        convert(
            argparse.Namespace(
                checkpoint=checkpoint,
                outfile=files[-1],
                embd_type="f16",
                output_type=None,
                validate=0,
                eff_bit=None,
                residual=None,
                use_itq=None,
                binary_dir=BUILD / "bin",
                unpadded_v=unpadded_v,
            )
        )
    return checkpoint, files[0], files[1]


def load_error(tool: Path, model: Path) -> str:
    result = subprocess.run(
        [str(tool), str(model), "1", "f16", "1"], capture_output=True, check=False
    )
    assert result.returncode != 0
    return result.stderr.decode(errors="replace")


def test_unpadded_layout_gives_the_same_logits(
    tmp_path: Path, logits_tool: Path
) -> None:
    torch.set_num_threads(2)
    checkpoint, padded, unpadded = convert_layouts(
        tmp_path, (33, 128, 200, 256, 8, 64, 130)
    )
    assert padded.stat().st_size > unpadded.stat().st_size
    token_ids = prompt_ids(checkpoint)
    expected = reference_logits(checkpoint, token_ids, 2)
    for ids, ref in zip(token_ids, expected, strict=True):
        old, new = (
            logits(logits_tool, path, ids, 2, "f16") for path in (padded, unpadded)
        )
        assert torch.isfinite(new).all() and new.std() > 0
        assert torch.equal(old, new)
        assert (new - ref).abs().max() < 0.1


def edit_gguf(path: Path, edit: Callable[[gguf.GGUFReader], None]) -> None:
    reader = gguf.GGUFReader(path, "r+")
    edit(reader)
    reader.data.flush()
    del reader


def test_loader_refuses_other_format_versions(
    tmp_path: Path, logits_tool: Path
) -> None:
    torch.set_num_threads(2)
    _, padded, unpadded = convert_layouts(tmp_path, (33, 128))

    def set_version(reader: gguf.GGUFReader) -> None:
        reader.get_field("littlebit.format_version").parts[-1][0] = 3

    for path in (padded, unpadded):
        edit_gguf(path, set_version)
        assert "unsupported LittleBit format version 3" in load_error(logits_tool, path)


@pytest.mark.parametrize(
    "tensor,dims,message",
    [
        # Each new shape keeps the byte size, so the file stays well formed.
        ("blk.0.attn_k.lb_u", [128, 256], "invalid LittleBit tensor blk.0.attn_k.lb_u"),
        ("blk.0.attn_q.lb_u", [256, 128], "invalid LittleBit tensor blk.0.attn_q.lb_u"),
        ("blk.0.attn_q.lb_v", [128, 66], "blk.0.attn_q.lb_v"),
    ],
)
def test_loader_refuses_malformed_shapes(
    tmp_path: Path, logits_tool: Path, tensor: str, dims: list[int], message: str
) -> None:
    torch.set_num_threads(2)
    _, _, unpadded = convert_layouts(tmp_path, (33, 130))

    def edit(reader: gguf.GGUFReader) -> None:
        field = next(t for t in reader.tensors if t.name == tensor).field
        field.parts[3][:] = dims

    edit_gguf(unpadded, edit)
    assert message in load_error(logits_tool, unpadded)
