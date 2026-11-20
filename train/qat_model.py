import copy
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Qwen3Config

from littlebit import BLOCK_SIZE, field_name, pack_signs, prefixes

UPSTREAM = "https://github.com/mgiachetti/little-bit.git"
UPSTREAM_REV = "02d146a72551762b68cb250577c6f42c29cf65a0"
EFF_BIT = 0.55
ITQ_ITERATIONS = 50


def upstream_layer(directory: Path) -> ModuleType:
    if not directory.exists():
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory.parent) as temp:
            checkout = Path(temp) / "little-bit"
            subprocess.run(["git", "clone", "--quiet", UPSTREAM, checkout], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    checkout,
                    "checkout",
                    "--quiet",
                    "--detach",
                    UPSTREAM_REV,
                ],
                check=True,
            )
            checkout.rename(directory)
    revision = subprocess.check_output(
        ["git", "-C", directory, "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REV or subprocess.check_output(
        ["git", "-C", directory, "status", "--porcelain", "--untracked-files=no"]
    ):
        raise ValueError(f"expected a clean upstream checkout at {UPSTREAM_REV}")
    spec = importlib.util.spec_from_file_location(
        "littlebit_upstream_layer", directory / "littlebit" / "layer.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load LittleBit layers from {directory}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rank_for_shape(width: int, out: int) -> int:
    if width % BLOCK_SIZE or min(width, out) < BLOCK_SIZE:
        raise ValueError(f"unsupported projection shape {out} x {width}")
    target = (EFF_BIT * width * out / 2 - 16 * (width + out)) / (width + out + 16)
    rounded = int(target / BLOCK_SIZE + 0.5) * BLOCK_SIZE
    return max(BLOCK_SIZE, rounded)


def scale_for_export(scale: torch.Tensor) -> torch.Tensor:
    rounded = scale.detach().bfloat16().float()
    return rounded + (scale - scale.detach()) if scale.requires_grad else rounded


class ExportableProjection(nn.Module):
    def __init__(self, factors: nn.Module, upstream: ModuleType) -> None:
        super().__init__()
        self.factors = factors
        self.sign = upstream.smooth_sign

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = []
        for path in self.factors.paths():
            u, v, h, g, latent = path.latents()
            x = F.linear(inputs * scale_for_export(g), self.sign(v, path.k).T)
            x = F.linear(x * scale_for_export(latent), self.sign(u, path.k))
            outputs.append(x * scale_for_export(h))
        result = outputs[0] + outputs[1]
        bias = self.factors.bias
        return result if bias is None else result + scale_for_export(bias)


def make_student(
    teacher: PreTrainedModel, upstream: ModuleType, *, resume: bool
) -> PreTrainedModel:
    student = copy.deepcopy(teacher).requires_grad_(False).float()
    names = prefixes(teacher.config.num_hidden_layers)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.copy_(parameter.bfloat16())
        for name in names:
            linear = teacher.get_submodule(name)
            rank = rank_for_shape(linear.in_features, linear.out_features)
            factors = upstream.LittleBitLinear(
                linear, rank, residual=True, init="empty" if resume else "itq"
            )
            student.set_submodule(name, ExportableProjection(factors, upstream))
    student.activation_checkpointing_enable(
        activation_checkpointing_kwargs={"use_reentrant": False}
    )
    return student


def size_plan(config: Qwen3Config) -> dict:
    width = config.hidden_size
    query = config.head_dim * config.num_attention_heads
    kv = config.head_dim * config.num_key_value_heads
    expanded = config.intermediate_size
    rows = {
        "q_proj": (width, query),
        "k_proj": (width, kv),
        "v_proj": (width, kv),
        "o_proj": (query, width),
        "gate_proj": (width, expanded),
        "up_proj": (width, expanded),
        "down_proj": (expanded, width),
    }
    layers = {}
    dense = packed = gguf = 0
    for name, (input_width, output_width) in rows.items():
        rank = rank_for_shape(input_width, output_width)
        weights = input_width * output_width
        checkpoint_bytes = 2 * (
            rank * (input_width + output_width) // 8
            + 2 * (input_width + output_width + 2 * rank)
            + 32
        )
        gguf_bytes = 2 * (
            18 * rank * (input_width + output_width) // 128 + 4 * input_width
        )
        layers[name] = {"in": input_width, "out": output_width, "rank": rank}
        dense += weights
        packed += checkpoint_bytes
        gguf += gguf_bytes
    return {
        "per_layer": layers,
        "layers": config.num_hidden_layers,
        "dense_projection_weights": dense * config.num_hidden_layers,
        "checkpoint_projection_bpw": packed * 8 / dense,
        "gguf_projection_bpw": gguf * 8 / dense,
    }


@torch.no_grad()
def export_checkpoint(
    student: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output: Path,
    *,
    tokenizer_model: bytes | None = None,
) -> dict:
    state = {}
    prefixes_in_state = tuple(
        name + "." for name in prefixes(student.config.num_hidden_layers)
    )
    for name, value in student.state_dict().items():
        if name.startswith(prefixes_in_state):
            continue
        if name == "lm_head.weight" and student.config.tie_word_embeddings:
            continue
        state[name] = value.cpu().bfloat16().contiguous()

    projection_bytes = dense_weights = 0
    for name in prefixes(student.config.num_hidden_layers):
        projection = student.get_submodule(name).factors
        dense_weights += projection.in_features * projection.out_features
        if projection.bias is not None:
            state[name + ".bias"] = projection.bias.cpu().bfloat16()
        for residual, path in enumerate(projection.paths()):
            u, v, h, g, latent = path.latents()
            u, v = u.cpu(), v.T.cpu()
            fields = {
                "U_packed": torch.from_numpy(pack_signs(u.numpy())),
                "V_packed": torch.from_numpy(pack_signs(v.numpy())),
                "U_shape": torch.tensor(u.shape),
                "V_shape": torch.tensor(v.shape),
                "u1": h.cpu().bfloat16().unsqueeze(0),
                "u2": latent.cpu().bfloat16().unsqueeze(0),
                "v1": torch.ones((1, latent.numel()), dtype=torch.bfloat16),
                "v2": g.cpu().bfloat16().unsqueeze(0),
            }
            for field, tensor in fields.items():
                state[field_name(name, field, bool(residual))] = tensor
                projection_bytes += tensor.numel() * tensor.element_size()

    metadata = {
        "quant_func": "smoothsign",
        "eff_bit": EFF_BIT,
        "split_dim": None,
        "residual": True,
        "kv_factor": 1.0,
        "min_split_dim": BLOCK_SIZE,
        "quant_mod": "littlebit",
        "use_itq": True,
        "itq_n_iter": ITQ_ITERATIONS,
    }
    config = (
        student.config.to_dict() | metadata | {"use_cache": True, "dtype": "bfloat16"}
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".export-", dir=output.parent) as temp:
        pending = Path(temp) / "checkpoint"
        pending.mkdir()
        for name, value in (
            ("config.json", config),
            ("littlebit_config.json", metadata),
        ):
            (pending / name).write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )
        save_file(state, pending / "model.safetensors", metadata={"format": "pt"})
        tokenizer.save_pretrained(pending)
        if tokenizer_model is not None and not (pending / "tokenizer.model").exists():
            (pending / "tokenizer.model").write_bytes(tokenizer_model)
        pending.rename(output)
    return {
        "checkpoint_projection_bytes": projection_bytes,
        "checkpoint_projection_bpw": projection_bytes * 8 / dense_weights,
        "checkpoint_tensor_file_bytes": (output / "model.safetensors").stat().st_size,
    }
