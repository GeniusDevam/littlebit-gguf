import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from tiny_model import make_tiny_littlebit, make_tiny_model

from convert_littlebit_to_gguf import Q, dense_type, gguf, main
from littlebit import (
    PathFactors,
    fp16_scale,
    pack_signs,
    prefixes,
    q1_blocks,
    unpack_signs,
)
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


@pytest.mark.parametrize("value", [np.inf, np.nan, 65536.0])
def test_rejects_out_of_range_scales(value: float) -> None:
    with pytest.raises(ValueError, match="fp16"):
        fp16_scale(np.array([value], dtype=np.float32))


def test_conversion_flushes_only_scales_that_round_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "model.gguf"
    make_tiny_model(checkpoint)
    paths = prefixes(2)
    weights = checkpoint / "model.safetensors"
    tensors = load_file(weights)
    for prefix in paths:
        rows, width = tensors.pop(f"{prefix}.weight").shape
        rank = 8
        factors = {
            "U_shape": torch.tensor([rows, rank]),
            "V_shape": torch.tensor([rank, width]),
            "U_packed": torch.zeros((rows, 1), dtype=torch.int32),
            "V_packed": torch.zeros((rank, width // 32), dtype=torch.int32),
            "u1": torch.ones((1, rows), dtype=torch.bfloat16),
            "u2": torch.ones((1, rank), dtype=torch.bfloat16),
            "v1": torch.ones((1, rank), dtype=torch.bfloat16),
            "v2": torch.ones((1, width), dtype=torch.bfloat16),
        }
        if prefix == paths[0]:
            factors["u1"][0, :3] = torch.tensor([2**-25, 2**-24, 0])
            factors["u2"][0, :2] = torch.tensor([-(2**-25), -(2**-23)])
            factors["v1"][0, :2] = 0.5
        tensors.update({f"{prefix}.{key}": value for key, value in factors.items()})
    save_file(tensors, weights)
    monkeypatch.setattr(
        "sys.argv",
        [
            "convert",
            str(checkpoint),
            "--outfile",
            str(output),
            "--eff-bit",
            "0.1",
            "--no-residual",
        ],
    )
    main()

    assert (
        "Flushed 2 nonzero scales to fp16 zero (largest magnitude 2.98023e-08)"
        in capsys.readouterr().out
    )
    stored = {tensor.name: tensor for tensor in gguf.GGUFReader(output).tensors}
    u = gguf.dequantize(stored["blk.0.attn_q.lb_u"].data, Q.Q1_0)
    v = gguf.dequantize(stored["blk.0.attn_q.lb_v"].data, Q.Q1_0)
    # Llama interleaves the two halves of each query head.
    np.testing.assert_array_equal(u[[0, 2, 4], 0], [0, 2**-24, 0])
    np.testing.assert_array_equal(v[:2, 0], [0, -(2**-24)])


@pytest.mark.parametrize("residual", [False, True])
def test_tiny_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, residual: bool
) -> None:
    torch.set_num_threads(2)
    monkeypatch.setattr("convert_littlebit_to_gguf.DENSE_CHUNK_ROWS", 7)
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
    embedding = load_file(checkpoint / "model.safetensors")["model.embed_tokens.weight"]
    np.testing.assert_array_equal(
        tensors["token_embd.weight"].data,
        gguf.quantize(embedding.float().numpy(), Q.Q8_0),
    )
    assert "output.weight" not in tensors
    assert "blk.0.attn_q.weight" not in tensors
    assert tensors["blk.0.attn_q.lb_v"].tensor_type == Q.Q1_0
    assert tuple(tensors["blk.0.attn_q.lb_v"].shape) == (256, 128)
    assert ("blk.0.attn_q.lb_v_r" in tensors) == residual
    for key in ("format_version", "residual", "eff_bits", "use_itq"):
        assert reader.get_field(f"littlebit.{key}") is not None


@pytest.mark.parametrize(
    "embd_arg,output_arg,embedding_type,output_type",
    [
        ("q4_k", None, Q.Q5_0, None),
        ("q6_k", "q8_0", Q.Q8_0, None),
        ("q6_k", "q4_k", Q.Q8_0, Q.Q5_0),
    ],
)
def test_k_quant_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    embd_arg: str,
    output_arg: str | None,
    embedding_type: Q,
    output_type: Q | None,
) -> None:
    torch.set_num_threads(2)
    source, checkpoint, output = (
        tmp_path / name for name in ("source", "lb", "model.gguf")
    )
    make_tiny_model(source, width=128)
    make_checkpoint(str(source), checkpoint, 0.1, False, 0)
    args = [
        "convert",
        str(checkpoint),
        "--outfile",
        str(output),
        "--embd-type",
        embd_arg,
    ]
    if output_arg:
        args += ["--output-type", output_arg]
    monkeypatch.setattr("sys.argv", args)
    main()
    notices = capsys.readouterr().out
    assert (
        f"token_embd.weight: width 128 is incompatible with {embd_arg.upper()}; using {embedding_type.name}"
        in notices
    )
    types = {t.name: t.tensor_type for t in gguf.GGUFReader(output).tensors}
    assert types["token_embd.weight"] == embedding_type
    assert types.get("output.weight") == output_type
    if output_type is not None:
        assert output_arg is not None
        assert (
            f"output.weight: width 128 is incompatible with {output_arg.upper()}; using {output_type.name}"
            in notices
        )


@pytest.mark.parametrize("requested", [Q.Q4_K, Q.Q6_K, Q.Q8_0])
def test_unaligned_dense_width_falls_back_to_f16(requested: Q) -> None:
    assert dense_type("output.weight", 129, requested) == Q.F16


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


MIXED_RANKS = (33, 128, 200, 256, 8, 64, 130)


def tensor_digest(path: Path) -> str:
    # The embedding shape depends on the tokenizer trained for the tiny model, so leave it out.
    digest = hashlib.sha256()
    for tensor in gguf.GGUFReader(path).tensors:
        if tensor.name != "token_embd.weight":
            label = f"{tensor.name} {tensor.tensor_type.name} {list(tensor.shape)}"
            digest.update(label.encode() + tensor.data.tobytes())
    return digest.hexdigest()


def convert_tiny(
    checkpoint: Path, output: Path, *flags: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = ["convert", str(checkpoint), "--outfile", str(output), "--validate", "14"]
    monkeypatch.setattr("sys.argv", [*args, *flags])
    main()


def test_unpadded_v_changes_only_v_and_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    torch.set_num_threads(2)
    checkpoint = tmp_path / "checkpoint"
    make_tiny_littlebit(checkpoint, MIXED_RANKS)
    padded, unpadded = tmp_path / "padded.gguf", tmp_path / "unpadded.gguf"
    convert_tiny(checkpoint, padded, monkeypatch=monkeypatch)
    convert_tiny(checkpoint, unpadded, "--unpadded-v", monkeypatch=monkeypatch)
    errors = re.findall(r"max abs ([\d.e+-]+)", capsys.readouterr().out)
    assert len(errors) == 28 and max(map(float, errors)) < 1e-3

    old, new = gguf.GGUFReader(padded), gguf.GGUFReader(unpadded)
    assert old.get_field("littlebit.format_version").contents() == 1
    assert new.get_field("littlebit.format_version").contents() == 2
    assert [t.name for t in old.tensors] == [t.name for t in new.tensors]
    ranks = set()
    for before, after in zip(old.tensors, new.tensors, strict=True):
        if not before.name.endswith(("lb_v", "lb_v_r")):
            assert before.data.tobytes() == after.data.tobytes(), before.name
            continue
        rank = int(after.shape[1])
        ranks.add(rank)
        assert int(before.shape[1]) == -(-rank // 128) * 128
        assert before.data.tobytes().startswith(after.data.tobytes())
    assert ranks == set(MIXED_RANKS)
    changed = {"GGUF.version", "littlebit.format_version"}
    assert old.fields.keys() == new.fields.keys()
    for key in old.fields.keys() - changed:
        assert old.get_field(key).contents() == new.get_field(key).contents(), key


def test_aligned_ranks_write_the_same_file_in_both_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(2)
    checkpoint = tmp_path / "checkpoint"
    make_tiny_littlebit(checkpoint, (128, 256))
    padded, unpadded = tmp_path / "padded.gguf", tmp_path / "unpadded.gguf"
    convert_tiny(checkpoint, padded, monkeypatch=monkeypatch)
    convert_tiny(checkpoint, unpadded, "--unpadded-v", monkeypatch=monkeypatch)
    assert padded.read_bytes() == unpadded.read_bytes()
    reader = gguf.GGUFReader(unpadded)
    assert reader.get_field("littlebit.format_version").contents() == 1


# Digest of every tensor name, type, shape and byte, taken from the converter before format version 2.
DEFAULT_OUTPUT_DIGEST = (
    "3177fe961fd47365943c531fff124709282ce82b84f0dfbe4f4d058b54b7aa5b"
)
DEFAULT_KEYS = (
    "GGUF.version GGUF.tensor_count GGUF.kv_count general.architecture "
    "littlebit.format_version littlebit.residual littlebit.eff_bits littlebit.use_itq "
    "general.type general.name general.size_label llama.block_count "
    "llama.context_length llama.embedding_length llama.feed_forward_length "
    "llama.attention.head_count llama.attention.head_count_kv llama.rope.freq_base "
    "llama.attention.layer_norm_rms_epsilon llama.attention.key_length "
    "llama.attention.value_length general.file_type llama.vocab_size "
    "llama.rope.dimension_count general.quantization_version tokenizer.ggml.model "
    "tokenizer.ggml.pre tokenizer.ggml.tokens tokenizer.ggml.scores "
    "tokenizer.ggml.token_type tokenizer.ggml.bos_token_id tokenizer.ggml.eos_token_id"
).split()


def test_default_output_matches_previous_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.set_num_threads(2)
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "model.gguf"
    make_tiny_littlebit(checkpoint, MIXED_RANKS)
    convert_tiny(checkpoint, output, monkeypatch=monkeypatch)
    assert tensor_digest(output) == DEFAULT_OUTPUT_DIGEST
    reader = gguf.GGUFReader(output)
    assert list(reader.fields) == DEFAULT_KEYS
    assert reader.get_field("littlebit.format_version").contents() == 1

    # The name and size label describe the dense model, not the directory and the packed factors.
    state = load_file(checkpoint / "model.safetensors")
    layers = tuple(f"{p}." for p in prefixes(2))
    dense = sum(v.numel() for n, v in state.items() if not n.startswith(layers))
    dense += sum(
        state[f"{p}.U_shape"][0] * state[f"{p}.V_shape"][1] for p in prefixes(2)
    )
    assert reader.get_field("general.size_label").contents() == gguf.size_label(
        int(dense), int(dense), 0, 0
    )
    assert reader.get_field("general.name").contents().startswith("LittleBit Llama ")


def test_model_name_option(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(2)
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "model.gguf"
    make_tiny_littlebit(checkpoint, (128,))
    convert_tiny(
        checkpoint, output, "--model-name", "Tiny test", monkeypatch=monkeypatch
    )
    assert gguf.GGUFReader(output).get_field("general.name").contents() == "Tiny test"
