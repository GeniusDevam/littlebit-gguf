import argparse
import json
import os
import subprocess
import sys
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
from train.data import token_cache, training_blocks  # noqa: E402
from train.qat_model import (  # noqa: E402
    export_checkpoint,
    make_student,
    size_plan,
    upstream_layer,
    upstream_recipe,
)
from train.train_qat import Trainer, legacy_distillation  # noqa: E402


@pytest.fixture(scope="module")
def upstream_path() -> Path:
    path = Path(os.environ.get("LITTLEBIT_UPSTREAM", "/tmp/littlebit-upstream"))
    if not path.is_dir():
        pytest.skip("set LITTLEBIT_UPSTREAM to the pinned LittleBit training checkout")
    return path


@pytest.fixture(scope="module")
def upstream(upstream_path: Path) -> ModuleType:
    return upstream_layer(upstream_path)


@pytest.fixture(scope="module")
def recipe(upstream: ModuleType, upstream_path: Path) -> ModuleType:
    return upstream_recipe(upstream_path)


def test_train_resume_export(
    tmp_path: Path, upstream: ModuleType, recipe: ModuleType
) -> None:
    torch.set_num_threads(2)
    source = tmp_path / "source"
    make_tiny_model(source, architecture="qwen3")
    teacher = AutoModelForCausalLM.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    student = make_student(teacher, upstream, resume=False)
    ids = torch.tensor([[1, 9, 5, 11, 8, 10, 2, 7]])
    batches = torch.cat([ids, ids.flip(-1)])
    settings = {"recipe": recipe, "total_steps": 10, "grad_accum": 2}
    trainer = Trainer(student, teacher, 8e-5, **settings)
    actual = trainer.loss(ids, 4)
    expected = recipe.kd_loss(student, teacher, ids, 10.0, True, 1, 4)
    torch.testing.assert_close(actual, expected[:3], atol=0, rtol=0)

    losses = [trainer.train_step(batches, 4)[0] for _ in range(4)]
    print(f"Tiny distillation losses: {losses}")
    assert losses[-1] < losses[0]
    assert trainer.optim.param_groups[0]["lr"] == recipe.lr_at(3, 10, 8e-5, 0.02)
    assert not student.get_input_embeddings().weight.requires_grad

    resume = tmp_path / "resume.pt"
    signature = {"fixture": "tiny"}
    trainer.save(resume, signature)
    reconstructed = make_student(teacher, upstream, resume=True)
    restarted = Trainer(reconstructed, teacher, 8e-5, **settings)
    with pytest.raises(ValueError, match="does not match"):
        restarted.restore(resume, {"fixture": "other"})
    restarted.restore(resume, signature)
    assert restarted.step == 4
    next_batches = torch.cat([ids.roll(1, -1), ids])
    assert restarted.train_step(next_batches, 4) == trainer.train_step(next_batches, 4)
    assert restarted.optim.param_groups[0]["lr"] == trainer.optim.param_groups[0]["lr"]
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
    trainer = Trainer(student, teacher, 5e-5, recipe=None, total_steps=4, grad_accum=1)
    ids = torch.tensor([[1, 9, 5, 11]])
    kl, mse = legacy_distillation(student, teacher, ids, 2, False)
    torch.testing.assert_close(trainer.loss(ids, 2), (kl + mse, kl, mse))
    losses = [trainer.train_step(ids, 2)[0] for _ in range(4)]
    print(f"Full-width random distillation losses: {losses}")
    assert losses[-1] < losses[0]


def test_trained_gguf(
    tmp_path: Path, upstream: ModuleType, recipe: ModuleType, logits_tool: Path
) -> None:
    torch.set_num_threads(2)
    source, checkpoint = tmp_path / "source", tmp_path / "littlebit"
    make_tiny_model(source, architecture="qwen3")
    teacher = AutoModelForCausalLM.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    student = make_student(teacher, upstream, resume=False)
    trainer = Trainer(
        student, teacher, 8e-5, recipe=recipe, total_steps=10, grad_accum=2
    )
    ids = torch.tensor([[1, 9, 5, 11, 8, 10, 2, 7]])
    for _ in range(4):
        trainer.train_step(torch.cat([ids, ids.flip(-1)]), 4)
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


def test_token_cache(tmp_path: Path) -> None:
    corpus = {
        "train": torch.arange(9, dtype=torch.int32),
        "test": torch.arange(3, dtype=torch.int32),
        "c4": torch.arange(13, dtype=torch.int32),
    }

    def unexpected_tokenization() -> dict[str, torch.Tensor]:
        pytest.fail("a cached corpus should not call the tokenizer")

    signature = {"c4_tokens": 13}
    source = tmp_path / "first.pt"
    first = token_cache(source, None, signature, lambda: corpus)
    destination = tmp_path / "continued.pt"
    second = token_cache(destination, source, signature, unexpected_tokenization)
    torch.testing.assert_close(first, corpus)
    torch.testing.assert_close(second, first)
    assert destination.read_bytes() == source.read_bytes()
    blocks, order = training_blocks(second, 4, seed=7, wiki_repeat=4)
    assert len(blocks) == 11
    torch.testing.assert_close(blocks[8:], corpus["c4"][:12].view(3, 4))
    torch.testing.assert_close(
        order, training_blocks(first, 4, seed=7, wiki_repeat=4)[1]
    )
    with pytest.raises(ValueError, match="cache does not match"):
        token_cache(
            tmp_path / "wrong.pt", source, {"c4_tokens": 0}, unexpected_tokenization
        )
    assert not (tmp_path / "wrong.pt").exists()


def test_notebook_child_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = Path(__file__).resolve().parents[1]
    cells = json.loads((path / "train/littlebit_qat.ipynb").read_text())["cells"]
    (tmp_path / "sitecustomize.py").write_text(
        'raise ModuleNotFoundError("No module named wrapt")\n'
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("LITTLEBIT_REPO", str(path))
    namespace = {"__name__": "__main__"}
    for cell in cells[1:3]:
        exec("".join(cell["source"]), namespace)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        namespace["run_command"](
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('output'); print('real error', file=sys.stderr); sys.exit(3)",
            ]
        )
    assert failure.value.returncode == 3
    output = capsys.readouterr().out
    assert output.splitlines()[-2:] == ["output", "real error"]
    assert "sitecustomize" not in output
