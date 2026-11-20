import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tiny_model import make_tiny_model
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


class FloatProjection(torch.nn.Module):
    def __init__(
        self, state: dict[str, torch.Tensor], prefix: str, residual: bool
    ) -> None:
        super().__init__()
        self.paths = []
        for extra in range(1 + residual):
            path = read_path(state.__getitem__, prefix, bool(extra))
            self.paths.append(
                tuple(
                    torch.from_numpy(value)
                    for value in (
                        unpack_signs(path.u, len(path.latent)),
                        unpack_signs(path.v, len(path.g)),
                        path.h,
                        path.latent,
                        path.g,
                    )
                )
            )
        self.bias = state.get(f"{prefix}.bias")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [
            F.linear(F.linear(x * g, v) * latent, u) * h
            for u, v, h, latent, g in self.paths
        ]
        result = sum(outputs)
        return result if self.bias is None else result + self.bias.float()


def reference_model(checkpoint: Path) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(checkpoint)
    state = load_file(checkpoint / "model.safetensors")
    metadata = json.loads(
        (checkpoint / "littlebit_config.json").read_text(encoding="utf-8")
    )
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16).eval()
    for prefix in prefixes(config.num_hidden_layers):
        model.set_submodule(
            prefix, FloatProjection(state, prefix, metadata["residual"])
        )
    model.float()
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
def compare_logits(
    checkpoint: Path, gguf_path: Path, tool: Path, threads: int, kv_type: str = "f16"
) -> tuple[float, float, float]:
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    reference = reference_model(checkpoint)
    limits = []
    for prompt in PROMPTS:
        ids = tokenizer.encode(prompt)
        expected = reference(torch.tensor([ids])).logits[0]
        actual = logits(tool, gguf_path, ids, threads, kv_type)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all()
        agreement = (actual.argmax(-1) == expected.argmax(-1)).float().mean().item()
        error = (actual - expected).abs().max().item()
        kl = F.kl_div(
            actual.log_softmax(-1), expected.softmax(-1), reduction="batchmean"
        ).item()
        limits.append((agreement, error, kl))
    agreement = sum(row[0] for row in limits) / len(limits)
    error = max(row[1] for row in limits)
    kl = sum(row[2] for row in limits) / len(limits)
    print(
        f"{gguf_path.stem} {kv_type}: top-1 {agreement:.4f}, max abs {error:.6f}, KL {kl:.6g}"
    )
    return agreement, error, kl


@pytest.mark.parametrize(
    "architecture,bits",
    [("llama", 0.1), ("llama", 1.0), ("qwen2", 0.1), ("qwen3", 0.1)],
)
def test_tiny_runtime(
    tmp_path: Path, logits_tool: Path, architecture: str, bits: float
) -> None:
    source, checkpoint, gguf_path = (
        tmp_path / name for name in ("source", "lb", "tiny.gguf")
    )
    torch.set_num_threads(2)
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
    for kv_type in ("f16", "q8_0"):
        top1, error, kl = compare_logits(checkpoint, gguf_path, logits_tool, 2, kv_type)
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
    top1, error, kl = compare_logits(checkpoint, gguf_path, logits_tool, 4)
    assert top1 >= 0.8
    assert error < 1.25
    assert kl < 0.03
