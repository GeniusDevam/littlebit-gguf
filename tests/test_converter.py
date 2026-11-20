from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from tiny_model import make_tiny_model

from convert_littlebit_to_gguf import Q, gguf, main
from littlebit import PathFactors, fp16_scale, pack_signs, q1_blocks, unpack_signs
from tools.make_synthetic import make_checkpoint


def test_bit_order_and_reblocking() -> None:
    words = np.array([[1, -2147483648, 257, -1, 2, 0, 0, 0]], dtype=np.int32)
    raw = q1_blocks(words, np.array([-0.5]), 256, 256)
    assert raw.reshape(2, 18)[:, :2].tolist() == [[0, 184], [0, 184]]
    assert raw.reshape(2, 18)[:, 2:].tolist() == [
        [254, 255, 255, 255, 255, 255, 255, 127, 254, 254, 255, 255, 0, 0, 0, 0],
        [
            253,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
        ],
    ]
    np.testing.assert_array_equal(
        gguf.dequantize(raw, Q.Q1_0), -0.5 * unpack_signs(words, 256)
    )


def test_rank_padding_and_layer_exactness() -> None:
    rng = np.random.default_rng(4)
    x = rng.integers(-3, 4, (6, 256), dtype=np.int32).astype(np.float32)
    g = np.ones(256, dtype=np.float32)
    h = np.array([-0.5, 0.0, 2.0], dtype=np.float32)
    rank = 33
    latent = rng.choice([0.125, -0.25, 0.5], rank).astype(np.float32)
    u = pack_signs(rng.choice([-1, 1], (len(h), rank)))
    v = pack_signs(rng.choice([-1, 1], (rank, len(g))))
    factors = PathFactors(u, v, h, latent, g)
    packed_u = q1_blocks(u, h, 128, rank)
    raw_u = packed_u[:, 2:]
    assert np.all(raw_u[:, 4] <= 1)
    assert not raw_u[:, 5:].any()
    packed_v = q1_blocks(v, np.pad(latent, (0, 128 - rank)), 256, 256)
    dequant_v = gguf.dequantize(packed_v, Q.Q1_0)
    assert not dequant_v[rank:].any()
    actual = ((x * g) @ dequant_v.T) @ gguf.dequantize(packed_u, Q.Q1_0).T
    np.testing.assert_array_equal(actual, factors.forward(x))


@pytest.mark.parametrize("value", [np.inf, np.nan, 65536.0, 1e-9])
def test_rejects_out_of_range_scales(value: float) -> None:
    with pytest.raises(ValueError, match="fp16"):
        fp16_scale(np.array([value], dtype=np.float32))


@pytest.mark.parametrize("residual", [False, True])
def test_tiny_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, residual: bool
) -> None:
    torch.set_num_threads(2)
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "model.gguf"
    make_tiny_model(source)
    make_checkpoint(str(source), checkpoint, 0.1, residual, 0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert",
            str(checkpoint),
            "--outfile",
            str(output),
            "--embd-type",
            "q8_0",
            "--validate",
            "2",
        ],
    )
    main()
    reader = gguf.GGUFReader(output)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert tensors["token_embd.weight"].tensor_type == Q.Q8_0
    assert "output.weight" not in tensors
    assert "blk.0.attn_q.weight" not in tensors
    assert tensors["blk.0.attn_q.lb_v"].tensor_type == Q.Q1_0
    assert tuple(tensors["blk.0.attn_q.lb_v"].shape) == (256, 128)
    assert ("blk.0.attn_q.lb_v_r" in tensors) == residual
    for key in ("format_version", "residual", "eff_bits", "use_itq"):
        assert reader.get_field(f"littlebit.{key}") is not None


def test_legacy_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    source, checkpoint, output = (
        tmp_path / name for name in ("source", "lb", "model.gguf")
    )
    make_tiny_model(source)
    make_checkpoint(str(source), checkpoint, 1.0, False, 0)
    tensors = checkpoint / "model.safetensors"
    torch.save(
        load_file(tensors),
        checkpoint / "model_state.pt",
        _use_new_zipfile_serialization=False,
    )
    tensors.unlink()
    (checkpoint / "littlebit_config.json").unlink()
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert",
            str(checkpoint),
            "--outfile",
            str(output),
            "--eff-bit",
            "1.0",
            "--no-residual",
        ],
    )
    main()
    assert gguf.GGUFReader(output).get_field("littlebit.format_version") is not None


def test_rejects_unaligned_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.set_num_threads(2)
    source, checkpoint, output = (
        tmp_path / name for name in ("source", "lb", "model.gguf")
    )
    make_tiny_model(source, 192)
    make_checkpoint(str(source), checkpoint, 0.1, False, 0)
    monkeypatch.setattr(
        "sys.argv", ["convert", str(checkpoint), "--outfile", str(output)]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "input width 192 is not a multiple of 128" in capsys.readouterr().err
    assert not output.exists()
