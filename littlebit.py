from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
FIELDS = ("U_packed", "U_shape", "V_packed", "V_shape", "u1", "u2", "v1", "v2")
BLOCK_SIZE = 128


def prefixes(block_count: int) -> list[str]:
    return [
        f"model.layers.{i}.{name}" for i in range(block_count) for name in PROJECTIONS
    ]


def field_name(prefix: str, field: str, residual: bool) -> str:
    if residual:
        base, sep, tail = field.partition("_")
        field = f"{base}_R{sep}{tail}"
    return f"{prefix}.{field}"


def pack_signs(values: np.ndarray) -> np.ndarray:
    rows, cols = values.shape
    bits = np.zeros((rows, (cols + 31) // 32 * 32), dtype=np.uint8)
    bits[:, :cols] = values < 0
    return np.packbits(bits, axis=1, bitorder="little").view("<i4")


def unpack_signs(words: np.ndarray, cols: int) -> np.ndarray:
    packed = np.ascontiguousarray(words, dtype="<i4").view(np.uint8)
    bits = np.unpackbits(packed, axis=1, bitorder="little")[:, :cols]
    return 1.0 - 2.0 * bits.astype(np.float32)


def q1_blocks(
    words: np.ndarray, scales: np.ndarray, cols: int, used_cols: int
) -> np.ndarray:
    if cols <= 0 or cols % BLOCK_SIZE:
        raise ValueError(f"Q1_0 row length must be a positive multiple of {BLOCK_SIZE}")
    rows = len(scales)
    data = np.zeros((rows, cols // BLOCK_SIZE, 18), dtype=np.uint8)
    data[:, :, :2] = (
        np.ascontiguousarray(scales, dtype="<f2").view(np.uint8).reshape(rows, 1, 2)
    )
    signs = data[:, :, 2:].copy().reshape(rows, -1)
    packed = np.ascontiguousarray(words, dtype="<i4").view(np.uint8)
    signs[: len(words), : packed.shape[1]] = np.bitwise_not(packed)
    complete_bytes, partial = divmod(used_cols, 8)
    if partial:
        signs[:, complete_bytes] &= (1 << partial) - 1
    signs[:, complete_bytes + bool(partial) :] = 0
    data[:, :, 2:] = signs.reshape(rows, -1, 16)
    return data.reshape(rows, -1)


def fp16_scale(values: np.ndarray) -> tuple[np.ndarray, float, int, float]:
    if (
        not np.isfinite(values).all()
        or (np.abs(values) > np.finfo(np.float16).max).any()
    ):
        raise ValueError("scale outside the finite fp16 range")
    rounded = values.astype(np.float16)
    nonzero = values != 0
    flushed = nonzero & (rounded == 0)
    error = np.abs(rounded[nonzero].astype(np.float32) / values[nonzero] - 1)
    return (
        rounded,
        float(error.max(initial=0)),
        int(np.count_nonzero(flushed)),
        float(np.abs(values[flushed]).max(initial=0)),
    )


@dataclass
class PathFactors:
    u: np.ndarray
    v: np.ndarray
    h: np.ndarray
    latent: np.ndarray
    g: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        v = unpack_signs(self.v, len(self.g))
        u = unpack_signs(self.u, len(self.latent))
        return ((((x * self.g) @ v.T) * self.latent) @ u.T) * self.h


def read_path(
    get: Callable[[str], torch.Tensor], prefix: str, residual: bool
) -> PathFactors:
    names = {field: field_name(prefix, field, residual) for field in FIELDS}

    def shape(field: str) -> tuple[int, int]:
        tensor = get(names[field])
        if tensor.dtype != torch.int64 or tensor.shape != (2,):
            raise ValueError(f"{names[field]} must have dtype int64 and shape [2]")
        rows, cols = tensor.tolist()
        if min(rows, cols) <= 0:
            raise ValueError(f"{names[field]} must contain positive dimensions")
        return rows, cols

    def words(field: str, rows: int, cols: int) -> np.ndarray:
        tensor = get(names[field])
        if tensor.dtype != torch.int32 or tensor.shape != (rows, (cols + 31) // 32):
            raise ValueError(f"{names[field]} has invalid packed dimensions or dtype")
        return tensor.numpy()

    def scale(field: str, width: int) -> np.ndarray:
        tensor = get(names[field])
        if tensor.shape != (1, width) or tensor.dtype != torch.bfloat16:
            raise ValueError(
                f"{names[field]} must have dtype bf16 and shape [1, {width}]"
            )
        values = tensor.float().flatten().numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{names[field]} has a non-finite scale")
        return values

    out, rank = shape("U_shape")
    v_rank, width = shape("V_shape")
    if v_rank != rank:
        raise ValueError(f"{prefix}: U and V ranks differ")
    return PathFactors(
        words("U_packed", out, rank),
        words("V_packed", rank, width),
        scale("u1", out),
        scale("u2", rank) * scale("v1", rank),
        scale("v2", width),
    )


def read_legacy(path: Path) -> dict[str, torch.Tensor]:
    file = path / "model_state.pt"
    try:
        state = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used with files saved with" not in str(exc):
            raise
        state = torch.load(file, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("model_state.pt must contain a tensor state dict")
    return state
