import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from littlebit import field_name, pack_signs, prefixes  # noqa: E402


def choose_rank(width: int, out: int, eff_bit: float, residual: bool) -> int:
    paths = 2 if residual else 1
    target = (eff_bit * width * out / paths - 16 * (width + out)) / (width + out + 16)
    limit = min(width, out)
    return min(limit, max(8, math.floor(min(limit, target) / 8) * 8))


def magnitude_scales(factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left, values, right = torch.linalg.svd(factor.abs(), full_matrices=False)
    scale = values[0].sqrt()
    return left[:, 0].abs() * scale, right[0].abs() * scale


def dual_svid(
    weight: torch.Tensor, rank: int, prefix: str, residual: bool
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    q = min(rank + 8, *weight.shape)
    # Use a randomized truncated SVD so wide projections fit on a CPU.
    left, values, right = torch.svd_lowrank(weight, q=q, niter=4)
    root = values[:rank].sqrt()
    u = left[:, :rank] * root
    v = right[:, :rank] * root
    h, u2 = magnitude_scales(u)
    g, v1 = magnitude_scales(v)
    scales = {
        "u1": h.unsqueeze(0).bfloat16(),
        "u2": u2.unsqueeze(0).bfloat16(),
        "v1": v1.unsqueeze(0).bfloat16(),
        "v2": g.unsqueeze(0).bfloat16(),
    }
    signs_u = torch.where(u < 0, -1.0, 1.0)
    signs_v = torch.where(v.T < 0, -1.0, 1.0).contiguous()
    tensors = {
        "U_packed": torch.from_numpy(pack_signs(signs_u.numpy())),
        "V_packed": torch.from_numpy(pack_signs(signs_v.numpy())),
        "U_shape": torch.tensor(signs_u.shape, dtype=torch.int64),
        "V_shape": torch.tensor(signs_v.shape, dtype=torch.int64),
        **scales,
    }
    latent = scales["u2"].float() * scales["v1"].float()
    estimate = (
        ((signs_u * latent) @ signs_v) * scales["u1"].float().T * scales["v2"].float()
    )
    return {
        field_name(prefix, name, residual): tensor for name, tensor in tensors.items()
    }, estimate


@torch.inference_mode()
def make_checkpoint(
    source: str, output: Path, eff_bit: float, residual: bool, seed: int
) -> None:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(source, dtype="auto")
    config = model.config
    state = model.state_dict()
    del model
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    for prefix in prefixes(config.num_hidden_layers):
        weight = state.pop(prefix + ".weight").float()
        width, out = weight.shape[1], weight.shape[0]
        rank = choose_rank(width, out, eff_bit, residual)
        tensors, estimate = dual_svid(weight, rank, prefix, False)
        state.update(tensors)
        if residual:
            tensors, _ = dual_svid(weight - estimate, rank, prefix, True)
            state.update(tensors)
        print(f"{prefix}: rank {rank}", flush=True)

    for name, tensor in state.items():
        if tensor.is_floating_point() and tensor.element_size() > 2:
            state[name] = tensor.bfloat16()

    metadata = {
        "quant_func": "sign",
        "eff_bit": eff_bit,
        "split_dim": None,
        "residual": residual,
        "kv_factor": 1.0,
        "min_split_dim": 8,
        "quant_mod": "littlebit",
        "use_itq": False,
        "itq_n_iter": 0,
    }
    config_dict = config.to_dict() | metadata | {"use_cache": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".synthetic-", dir=output.parent) as temp:
        checkpoint = Path(temp) / "checkpoint"
        checkpoint.mkdir()
        save_file(
            {name: tensor.contiguous() for name, tensor in state.items()},
            checkpoint / "model.safetensors",
            metadata={"format": "pt"},
        )
        for name, value in (
            ("config.json", config_dict),
            ("littlebit_config.json", metadata),
        ):
            (checkpoint / name).write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )
        AutoTokenizer.from_pretrained(source).save_pretrained(checkpoint)
        tokenizer_model = Path(source) / "tokenizer.model"
        if (
            tokenizer_model.is_file()
            and not (checkpoint / tokenizer_model.name).exists()
        ):
            shutil.copyfile(tokenizer_model, checkpoint / tokenizer_model.name)
        checkpoint.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a synthetic LittleBit checkpoint using Dual-SVID."
    )
    parser.add_argument("source", help="local Hugging Face checkpoint or model ID")
    parser.add_argument("output", type=Path)
    parser.add_argument("--eff-bit", type=float, required=True)
    parser.add_argument(
        "--residual", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.eff_bit) or args.eff_bit <= 0 or args.threads < 1:
        parser.error("--eff-bit and --threads must be positive")
    torch.set_num_threads(args.threads)
    make_checkpoint(args.source, args.output, args.eff_bit, args.residual, args.seed)


if __name__ == "__main__":
    main()
