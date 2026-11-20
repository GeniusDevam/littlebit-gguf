import argparse
import json
import os
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402
from test_runtime import BUILD, compare_logits, logits, prompt_ids  # noqa: E402
from test_runtime import logits_tool as logits_tool  # noqa: E402
from tiny_model import make_tiny_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config  # noqa: E402

from convert_littlebit_to_gguf import convert  # noqa: E402
from littlebit import prefixes, read_path, unpack_signs  # noqa: E402
from train.qat_model import export_checkpoint, make_student, size_plan, upstream_layer  # noqa: E402
from train.train_qat import Trainer  # noqa: E402


@pytest.fixture(scope="module")
def upstream() -> ModuleType:
    path = Path(os.environ.get("LITTLEBIT_UPSTREAM", "/tmp/littlebit-upstream"))
    if not path.is_dir():
        pytest.skip("set LITTLEBIT_UPSTREAM to the pinned LittleBit training checkout")
    return upstream_layer(path)


def test_train_resume_export(tmp_path: Path, upstream: ModuleType) -> None:
    torch.set_num_threads(2)
    source = tmp_path / "source"
    make_tiny_model(source, architecture="qwen3")
    teacher = AutoModelForCausalLM.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    student = make_student(teacher, upstream, resume=False)
    ids = torch.tensor([[1, 9, 5, 11, 8, 10, 2, 7]])
    trainer = Trainer(student, teacher, 1e-4)
    losses = [sum(trainer.train_step(ids, 4)) for _ in range(4)]
    print(f"Tiny distillation losses: {losses}")
    assert losses[-1] < losses[0]
    assert not student.get_input_embeddings().weight.requires_grad

    resume = tmp_path / "resume.pt"
    signature = {"fixture": "tiny"}
    trainer.save(resume, signature)
    reconstructed = make_student(teacher, upstream, resume=True)
    restarted = Trainer(reconstructed, teacher, 1e-4)
    with pytest.raises(ValueError, match="does not match"):
        restarted.restore(resume, {"fixture": "other"})
    restarted.restore(resume, signature)
    assert restarted.step == 4
    assert restarted.train_step(ids, 4) == trainer.train_step(ids, 4)
    assert restarted.step == trainer.step == 5
    for name, parameter in restarted.parameters.items():
        assert torch.equal(parameter, trainer.parameters[name])

    checkpoint = tmp_path / "littlebit"
    stats = export_checkpoint(reconstructed, tokenizer, checkpoint)
    assert (
        stats["checkpoint_projection_bpw"]
        == size_plan(teacher.config)["checkpoint_projection_bpw"]
    )
    cfg = json.loads((checkpoint / "littlebit_config.json").read_text(encoding="utf-8"))
    assert cfg["use_itq"] and cfg["residual"] and cfg["min_split_dim"] == 128
    state = load_file(checkpoint / "model.safetensors")
    with torch.no_grad():
        for name in prefixes(teacher.config.num_hidden_layers):
            projection = reconstructed.get_submodule(name)
            for residual, factors in enumerate(projection.factors.paths()):
                u, v, h, g, latent = factors.latents()
                read = read_path(state.__getitem__, name, bool(residual))
                np.testing.assert_array_equal(
                    unpack_signs(read.u, len(read.latent)), torch.where(u < 0, -1, 1)
                )
                np.testing.assert_array_equal(
                    unpack_signs(read.v, len(read.g)), torch.where(v.T < 0, -1, 1)
                )
                for recovered, trained in (
                    (read.h, h),
                    (read.g, g),
                    (read.latent, latent),
                ):
                    np.testing.assert_array_equal(
                        recovered, trained.bfloat16().float().numpy()
                    )

    reference = AutoModelForCausalLM.from_config(teacher.config).eval()
    from test_runtime import FloatProjection

    for name in prefixes(teacher.config.num_hidden_layers):
        reference.set_submodule(
            name, FloatProjection(state, name, residual=True, q8_0=False)
        )
    dense = {
        name: value for name, value in state.items() if name in reference.state_dict()
    }
    missing, unexpected = reference.load_state_dict(dense, strict=False)
    assert missing == ["lm_head.weight"] and not unexpected
    with torch.inference_mode():
        torch.testing.assert_close(
            reference(ids).logits, reconstructed.eval()(ids).logits, rtol=0, atol=0
        )


def test_train_full_width(upstream: ModuleType) -> None:
    # Qwen/Qwen3-0.6B config at c1899de, with only the layer count reduced.
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        max_window_layers=28,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000},
        bos_token_id=151643,
        eos_token_id=151645,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        dtype="bfloat16",
    )
    torch.set_num_threads(2)
    torch.manual_seed(7)
    teacher = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    student = make_student(teacher, upstream, resume=False)
    plan = size_plan(config)
    assert [entry["rank"] for entry in plan["per_layer"].values()] == [128] * 4 + [
        256
    ] * 3
    trainer = Trainer(student, teacher, 5e-5)
    ids = torch.tensor([[1, 9, 5, 11]])
    losses = [sum(trainer.train_step(ids, 2)) for _ in range(4)]
    print(f"Full-width random distillation losses: {losses}")
    assert losses[-1] < losses[0]


def test_trained_gguf(tmp_path: Path, upstream: ModuleType, logits_tool: Path) -> None:
    torch.set_num_threads(2)
    source, checkpoint = tmp_path / "source", tmp_path / "littlebit"
    make_tiny_model(source, architecture="qwen3")
    teacher = AutoModelForCausalLM.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    student = make_student(teacher, upstream, resume=False)
    trainer = Trainer(student, teacher, 1e-4)
    ids = torch.tensor([[1, 9, 5, 11, 8, 10, 2, 7]])
    for _ in range(4):
        trainer.train_step(ids, 4)
    export_checkpoint(
        student,
        tokenizer,
        checkpoint,
        tokenizer_model=(source / "tokenizer.model").read_bytes(),
    )
    gguf = tmp_path / "trained.gguf"
    convert(
        argparse.Namespace(
            checkpoint=checkpoint,
            outfile=gguf,
            embd_type="f16",
            output_type=None,
            validate=2,
            eff_bit=None,
            residual=None,
            use_itq=None,
            binary_dir=BUILD / "bin",
        )
    )
    token_ids = prompt_ids(checkpoint)
    student.eval()
    with torch.inference_mode():
        expected = [student(torch.tensor([tokens])).logits[0] for tokens in token_ids]
    for cache in ("f16", "q8_0"):
        actual = [logits(logits_tool, gguf, tokens, 2, cache) for tokens in token_ids]
        top1, error, kl = compare_logits(f"trained {cache}", expected, actual)
        assert top1 >= 0.95
        assert error < 0.1
        assert kl < 0.001
