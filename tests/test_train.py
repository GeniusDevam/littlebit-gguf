import argparse
import gzip
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors.torch import load_file  # noqa: E402
from test_runtime import BUILD, compare_logits, logits, prompt_ids  # noqa: E402
from test_runtime import logits_tool as logits_tool  # noqa: E402
from tiny_model import make_tiny_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config  # noqa: E402

from convert_littlebit_to_gguf import convert  # noqa: E402
from littlebit import prefixes, read_path, unpack_signs  # noqa: E402
from train import data, evaluate_gguf  # noqa: E402
from train.data import token_cache, training_blocks  # noqa: E402
from train.qat_model import (  # noqa: E402
    export_checkpoint,
    make_student,
    size_plan,
    trainable_elements,
    upstream_layer,
    upstream_recipe,
)
from train.ranks import Ranks  # noqa: E402
from train.train_qat import Trainer, legacy_distillation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_R2 = "3885d294431862c9db21ca84870a585e7f3452ad"


@pytest.fixture(scope="module")
def stubs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stand-ins for the data packages that the offline tests never call."""
    path = tmp_path_factory.mktemp("stubs")
    for package, module, name in (
        ("datasets", "datasets/__init__.py", "load_dataset"),
        ("pyarrow", "pyarrow/parquet.py", "read_table"),
    ):
        if importlib.util.find_spec(package) is None:
            source = path / module
            source.parent.mkdir()
            (source.parent / "__init__.py").touch()
            source.write_text(f"{name} = None\n")
    return path


@pytest.fixture(scope="module")
def upstream_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = Path(os.environ.get("LITTLEBIT_UPSTREAM", "/tmp/littlebit-upstream"))
    if path.is_dir():
        return path
    if "CI" not in os.environ:
        pytest.skip("set LITTLEBIT_UPSTREAM to the pinned LittleBit training checkout")
    path = tmp_path_factory.mktemp("upstream") / "little-bit"
    for attempt in range(3):
        try:
            upstream_layer(path)
            return path
        except subprocess.CalledProcessError:
            if attempt == 2:
                raise
            time.sleep(10)
    raise AssertionError


@pytest.fixture(scope="module")
def upstream(upstream_path: Path) -> ModuleType:
    return upstream_layer(upstream_path)


@pytest.fixture(scope="module")
def recipe(upstream: ModuleType, upstream_path: Path, stubs: Path) -> ModuleType:
    sys.path.append(str(stubs))
    try:
        return upstream_recipe(upstream_path)
    finally:
        sys.path.remove(str(stubs))


def test_train_resume_export(
    tmp_path: Path, upstream: ModuleType, recipe: ModuleType
) -> None:
    torch.set_num_threads(2)
    source = tmp_path / "source"
    make_tiny_model(source, architecture="qwen3")
    teacher = AutoModelForCausalLM.from_pretrained(source)
    tokenizer = AutoTokenizer.from_pretrained(source)
    student = make_student(teacher, upstream, resume=False)
    assert trainable_elements(teacher.config) == sum(
        p.numel() for p in student.parameters() if p.requires_grad
    )
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
        projection = FloatProjection(state, name, residual=True, q8_0=False)
        # The trainer stores V with the rank last. Some CPUs round a product with a
        # transposed matrix differently, so give the reference the same layout.
        projection.paths = [
            (u, v.T.contiguous().T, *scales) for u, v, *scales in projection.paths
        ]
        reference.set_submodule(name, projection)
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


# Runs the trainer in a subprocess with injected faults. The alarm kills a rank that
# outlives its test, wherever it is blocked.
WORKER = """
import json, os, signal, sys, time
signal.alarm(int(os.environ["TEST_SECONDS"]))
from pathlib import Path
from types import SimpleNamespace
import torch
import train.train_qat as t

faults = json.loads(os.environ["TEST_FAULTS"])
rank = int(os.environ.get("RANK", "0"))
step, loss = t.Trainer.train_step, t.Trainer.loss
stopped = False

def train_step(self, *args):
    global stopped
    try:
        result = step(self, *args)
    except FloatingPointError:
        mark = Path(faults["marks"])
        (mark / f"rank-{rank}").touch()
        while len(list(mark.iterdir())) < 2:
            time.sleep(0.05)
        raise
    if rank == 0 and self.step == faults.get("stop_at"):
        stopped = True
    if rank == 1 and self.step == faults.get("diverge_at"):
        with torch.no_grad():
            next(iter(self.parameters.values())).view(-1)[0] += 1e-6
    return result

def bad_loss(self, *args):
    result = loss(self, *args)
    if rank == 1 and self.step + 1 == faults.get("nan_at"):
        result = (result[0] * float("nan"), *result[1:])
    return result

t.Trainer.train_step, t.Trainer.loss = train_step, bad_loss
t.time = SimpleNamespace(
    time=lambda: time.time() + 1e9 * stopped, monotonic=time.monotonic
)
if faults.get("check_every_step"):
    t.CHECK_INTERVAL = 0
if "old_code" not in faults:
    import train.ranks as r
    r.BUCKET_ELEMENTS = 5000
    r.PROBE_SECONDS = faults.get("probe_seconds", r.PROBE_SECONDS)
    accumulate = t.Trainer.accumulate
    if faults.get("wrong_sequence") and rank == 1:
        t.Trainer.accumulate = lambda self, batches, *args: accumulate(
            self, batches.roll(1, 0), *args
        )
    if "overflow_at" in faults:
        # Run the loss scaler on the CPU and overflow in the second sequence.
        clip = torch.nn.utils.clip_grad_norm_
        torch.nn.utils.clip_grad_norm_ = lambda p, norm, **kwargs: clip(p, norm)
        torch.amp.GradScaler = lambda *a, scaler=torch.amp.GradScaler, **k: scaler(
            "cpu", init_scale=1024
        )
        def overflow(self, batches, chunk, rows):
            result = accumulate(self, batches, chunk, rows)
            if 1 in rows and self.step + 1 in faults["overflow_at"]:
                next(iter(self.parameters.values())).grad.view(-1)[0] = float("inf")
            return result
        t.Trainer.accumulate = overflow
    if faults.get("hang") and rank == 1:
        r.Ranks.check_links = lambda self: time.sleep(1000)
if "perplexity" in faults:
    sys.path.insert(0, str(t.ROOT / "tests"))
    from tiny_model import make_tiny_model
    torch.set_num_threads(1)
    source = Path(faults["perplexity"]) / f"model-{rank}"
    make_tiny_model(source, architecture="qwen3")
    ranks = t.Ranks.launch(cpu=True)
    model = t.AutoModelForCausalLM.from_pretrained(source)
    ids = torch.randint(0, 8, (79,), generator=torch.Generator().manual_seed(1))
    scores = [t.perplexity(model, ids, 16, 4, split) for split in (None, ranks)]
    (source.parent / f"scores-{rank}.json").write_text(json.dumps(scores))
    ranks.close()
else:
    t.main()
"""


class Runner:
    def __init__(self, directory: Path, upstream: Path, stubs: Path) -> None:
        self.directory = directory
        self.upstream = upstream
        self.stubs = stubs

    def __call__(
        self,
        name: str,
        *,
        ranks: int = 1,
        resume: Path | None = None,
        root: Path = ROOT,
        seconds: int = 300,
        **faults: object,
    ) -> subprocess.CompletedProcess:
        worker = self.directory / f"{name}.py"
        worker.write_text(WORKER)
        command = [sys.executable, "-u"]
        if ranks > 1:
            command += ["-m", "torch.distributed.run", "--standalone"]
            command += ["--nproc-per-node", str(ranks)]
        command += [worker, "--tiny", "--cpu", "--seq-len", "16", "--tokens", "192"]
        command += ["--threads", "1", "--log-every", "1", "--upstream", self.upstream]
        command += ["--output", self.directory / name]
        if resume is not None:
            command += ["--resume", resume / "resume.pt"]
        env = os.environ | {
            "PYTHONPATH": os.pathsep.join([str(root), str(self.stubs)]),
            "OMP_NUM_THREADS": "1",
            "TEST_SECONDS": str(seconds),
            "TEST_FAULTS": json.dumps(faults),
        }
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            output, _ = process.communicate(timeout=seconds + 60)
        except subprocess.TimeoutExpired:
            process.kill()
            pytest.fail(f"{name} did not finish:\n{process.communicate()[0]}")
        print(output)
        return subprocess.CompletedProcess(command, process.returncode, output)

    def ok(self, name: str, **settings: object) -> Path:
        result = self(name, **settings)
        assert result.returncode == 0, result.stdout
        (self.directory / name / "log.txt").write_text(result.stdout)
        return self.directory / name


@pytest.fixture(scope="module")
def run_training(
    tmp_path_factory: pytest.TempPathFactory, upstream_path: Path, stubs: Path
) -> Runner:
    return Runner(tmp_path_factory.mktemp("runs"), upstream_path, stubs)


@pytest.fixture(scope="module")
def one_rank(run_training: Runner) -> Path:
    return run_training.ok("one-rank")


def logged_steps(run: Path) -> list[dict]:
    timing = {"tokens_per_second", "token_budget_eta_seconds"}
    entries = [
        json.loads(line)
        for line in (run / "log.txt").read_text().splitlines()
        if line.startswith('{"step"') and "loss" in line
    ]
    return [{k: v for k, v in e.items() if k not in timing} for e in entries]


def assert_identical(actual: object, expected: object, where: str = "resume") -> None:
    assert type(actual) is type(expected), where
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected), where
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), where
        for key, value in expected.items():
            assert_identical(actual[key], value, f"{where}.{key}")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), where
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
            assert_identical(a, b, f"{where}[{i}]")
    else:
        assert actual == expected, where


def assert_same_run(actual: Path, expected: Path) -> None:
    states = [
        torch.load(run / "resume.pt", map_location="cpu", weights_only=True)
        for run in (actual, expected)
    ]
    assert states[1]["step"] == 6 and states[1]["optimizer"]["state"]
    assert_identical(*states)
    assert_identical(
        load_file(actual / "littlebit/model.safetensors"),
        load_file(expected / "littlebit/model.safetensors"),
    )


def summary(run: Path) -> dict:
    return json.loads((run / "summary.json").read_text())


def test_two_ranks_equal_one_process(run_training: Runner, one_rank: Path) -> None:
    two = run_training.ok("two-ranks", ranks=2)
    assert_same_run(two, one_rank)
    steps = logged_steps(two)
    assert [entry["step"] for entry in steps] == [1, 2, 3, 4, 5, 6]
    assert steps == logged_steps(one_rank)

    log = (two / "log.txt").read_text()
    check = json.loads(
        next(line for line in log.splitlines() if "first_step_check" in line)
    )
    assert check == {
        "first_step_check": 1,
        "relative_norm_difference": 0.0,
        "relative_gradient_difference": 0.0,
    }
    assert log.count("Starting at step 0") == 1
    assert log.count("All-reduce check: 2 ranks") == 1
    assert log.count("Gradient all-reduce: ") == 1
    assert "first_step_check" not in (one_rank / "log.txt").read_text()

    results = [summary(run) for run in (two, one_rank)]
    assert [result["ranks"] for result in results] == [2, 1]
    for name in ("teacher", "student"):
        for score in ("perplexity", "c4_validation_perplexity"):
            values = [result[f"{name}_{score}"] for result in results]
            assert values[0] == pytest.approx(values[1], rel=1e-12)
    assert results[0]["c4_validation_documents"] == 2
    assert results[0]["c4_validation_tokens"] > 16
    text = (two / "c4-validation.txt").read_text()
    assert text == "The quick brown fox.\n\nPacking signs is fun.\n\n"


def test_resume_across_ranks(run_training: Runner, one_rank: Path) -> None:
    first = run_training.ok("switch-1", stop_at=2)
    second = run_training.ok("switch-2", ranks=2, resume=first, stop_at=4)
    third = run_training.ok("switch-3", resume=second)
    assert [summary(run)["final_step"] for run in (first, second, third)] == [2, 4, 6]
    assert_same_run(third, one_rank)
    steps = [entry for run in (first, second, third) for entry in logged_steps(run)]
    assert steps == logged_steps(one_rank)


def test_resume_from_train_r2(
    tmp_path: Path, run_training: Runner, one_rank: Path
) -> None:
    def git(*args: str, cwd: Path = ROOT) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True)

    known = subprocess.run(["git", "cat-file", "-e", TRAIN_R2], cwd=ROOT)
    if known.returncode:
        git("fetch", "--depth", "1", "origin", TRAIN_R2)
    archive, old_root = tmp_path / "old.tar", tmp_path / "old"
    git("archive", "--output", str(archive), TRAIN_R2, "train", "tests", "littlebit.py")
    with tarfile.open(archive) as tar:
        tar.extractall(old_root, filter="data")
    # The old trainer records the commit of the directory it runs in.
    identity = ("-c", "user.name=test", "-c", "user.email=test@example.com")
    git("init", "--quiet", cwd=old_root)
    git(*identity, "commit", "--quiet", "--allow-empty", "-m", "old", cwd=old_root)

    old = run_training.ok("old-code", root=old_root, stop_at=2, old_code=True)
    assert not (old / "c4-validation.txt").exists()
    new = run_training.ok("old-code-continued", ranks=2, resume=old)
    assert summary(new)["initial_step"] == 2
    assert_same_run(new, one_rank)
    assert (new / "tokens.pt").read_bytes() == (old / "tokens.pt").read_bytes()


def test_rank_zero_deadline_stops_both(run_training: Runner) -> None:
    run = run_training.ok("deadline", ranks=2, stop_at=3)
    assert summary(run)["final_step"] == 3
    assert summary(run)["stop"] == "time budget"


def test_non_finite_loss_stops_both(tmp_path: Path, run_training: Runner) -> None:
    result = run_training("non-finite", ranks=2, nan_at=2, marks=str(tmp_path))
    assert result.returncode not in (0, None)
    assert "non-finite loss at step 2" in result.stdout
    assert {path.name for path in tmp_path.iterdir()} == {"rank-0", "rank-1"}
    assert not (run_training.directory / "non-finite/summary.json").exists()


def test_overflow_is_shared(run_training: Runner) -> None:
    faults = {"overflow_at": [1, 3], "check_every_step": True}
    one = run_training.ok("overflow-one", **faults)
    first = run_training.ok("overflow-1", ranks=2, stop_at=4, **faults)
    second = run_training.ok("overflow-2", ranks=2, resume=first, **faults)
    assert_same_run(second, one)
    state = torch.load(second / "resume.pt", weights_only=True)
    assert state["scaler"]["scale"] == 256
    log = (first / "log.txt").read_text()
    assert log.count("first-step check moves to the next step") == 1
    assert '"first_step_check": 2,' in log
    for step in (1, 3):
        assert log.count(f"Step {step}: fp16 overflow, reduced loss scale") == 1
    steps = logged_steps(first) + logged_steps(second)
    assert steps == logged_steps(one)


def test_divergent_parameters_stop(run_training: Runner) -> None:
    result = run_training("diverge", ranks=2, diverge_at=2, check_every_step=True)
    assert result.returncode not in (0, None)
    assert "1 tensors differ between ranks, first model.layers.0." in result.stdout
    run = run_training.directory / "diverge"
    saved = torch.load(run / "resume.pt", map_location="cpu", weights_only=True)
    assert saved["step"] == 1
    assert not (run / "summary.json").exists()


def test_wrong_sequence_fails_first_step_check(run_training: Runner) -> None:
    result = run_training("wrong-sequence", ranks=2, wrong_sequence=True)
    assert result.returncode not in (0, None)
    assert "first-step check failed" in result.stdout
    assert not (run_training.directory / "wrong-sequence/resume.pt").exists()


def test_hung_start_aborts(run_training: Runner) -> None:
    result = run_training("hang", ranks=2, hang=True, probe_seconds=20)
    assert result.returncode not in (0, None)
    assert "Rank 0: the small all-reduce did not finish in 20 s" in result.stdout


def test_split_perplexity(tmp_path: Path, run_training: Runner) -> None:
    result = run_training("perplexity", ranks=2, perplexity=str(tmp_path))
    assert result.returncode == 0, result.stdout
    for rank in range(2):
        alone, split = json.loads((tmp_path / f"scores-{rank}.json").read_text())
        assert split == pytest.approx(alone, rel=1e-12)


def test_sequences_must_divide_over_ranks(tmp_path: Path) -> None:
    make_tiny_model(tmp_path / "source", architecture="qwen3")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "source")
    ranks = Ranks(torch.device("cpu"), rank=0, size=2)
    with pytest.raises(ValueError, match="do not divide"):
        Trainer(
            model,
            model,
            8e-5,
            recipe=None,
            total_steps=4,
            grad_accum=3,
            ranks=ranks,
        )


def test_c4_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_tiny_model(tmp_path / "source", architecture="qwen3")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "source")
    shard = tmp_path / "shard.json.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as output:
        for i in range(5):
            output.write(json.dumps({"text": f"lazy dog {i}"}) + "\n")
    requested = []

    def download(*args: object, **kwargs: object) -> str:
        requested.append((args, kwargs["revision"]))
        return str(shard)

    monkeypatch.setattr(data, "hf_hub_download", download)
    monkeypatch.setattr(data, "C4_VALIDATION_DOCUMENTS", 3)
    documents = data.c4_validation_documents()
    assert documents == ["lazy dog 0", "lazy dog 1", "lazy dog 2"]
    name = "en/c4-validation.00000-of-00008.json.gz"
    assert requested == [(("allenai/c4", name), data.C4_REV)]
    data.write_c4_validation(tmp_path / "text", documents)
    text = "lazy dog 0\n\nlazy dog 1\n\nlazy dog 2\n\n"
    assert (tmp_path / "text").read_bytes() == text.encode()

    def ids(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False).input_ids

    expected = [token for d in documents for token in ids(d) + ids("\n\n")]
    assert data.join_documents(tokenizer, documents).tolist() == expected
    assert data.join_documents(tokenizer, documents, 5).tolist() == expected[:5]
    monkeypatch.setattr(data, "C4_VALIDATION_DOCUMENTS", 6)
    with pytest.raises(ValueError, match="expected 6"):
        data.c4_validation_documents()


def test_evaluate_scores_c4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    results, output = tmp_path / "results", tmp_path / "evaluation"
    results.mkdir()
    (results / "summary.json").write_text('{"training_commit": "abc"}')
    commands = []

    def record(args: tuple, **kwargs: object) -> None:
        commands.append([str(arg) for arg in args])
        if "--outfile" in args:
            args[args.index("--outfile") + 1].touch()

    monkeypatch.setattr(evaluate_gguf.subprocess, "check_output", lambda *a, **k: "abc")
    monkeypatch.setattr(evaluate_gguf.subprocess, "run", record)
    monkeypatch.setattr(evaluate_gguf, "snapshot_download", lambda *a, **k: "dense")
    monkeypatch.setattr(evaluate_gguf, "write_wiki_test", lambda path: path.touch())
    with pytest.raises(FileNotFoundError, match="c4-validation.txt"):
        evaluate_gguf.evaluate(results, output, 1)
    (results / "c4-validation.txt").touch()
    evaluate_gguf.evaluate(results, output, 1)
    scored = {
        (Path(command[2]).name, Path(command[4]).name)
        for command in commands
        if command[0].endswith("llama-perplexity")
    }
    models = ("qwen3-bf16.gguf", "littlebit-q8_0.gguf", "littlebit-q4_k.gguf")
    texts = ("wikitext-2-test.txt", "c4-validation.txt")
    assert scored == {(model, text) for model in models for text in texts}


class Notebook:
    """Runs notebook cells with recorded commands and paths below a directory."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.gpus: int | None = 2
        self.commands: list[list[str]] = []
        for name in ("LITTLEBIT_REPO", "LITTLEBIT_TINY", "LITTLEBIT_OUTPUT"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setattr(subprocess, "check_output", self.check_output)
        (root / "working").mkdir()
        (root / "input").mkdir()

    def popen(self, command: list, **kwargs: object):
        self.commands.append([str(part) for part in command])
        notebook = self

        class Process:
            stdout: list[str] = []

            def __enter__(self) -> "Process":
                return self

            def __exit__(self, *error: object) -> None:
                pass

            def wait(self) -> int:
                return notebook.finish(command)

        return Process()

    def finish(self, command: list) -> int:
        if "train.train_qat" in command:
            output = Path(command[command.index("--output") + 1])
            for name in ("resume.pt", "tokens.pt", "littlebit/littlebit_config.json"):
                (output / name).parent.mkdir(parents=True, exist_ok=True)
                (output / name).touch()
        return 0

    def check_output(self, command: list, **kwargs: object) -> str:
        if command[0] != "nvidia-smi":
            return "abc\n"
        if self.gpus is None:
            raise FileNotFoundError("nvidia-smi")
        return "".join(f"GPU {i}: Tesla T4 (UUID: {i})\n" for i in range(self.gpus))

    def run(self, cells: int = 2, **settings: object) -> list[list[str]]:
        self.commands = []
        notebook = json.loads((ROOT / "train/littlebit_qat.ipynb").read_text())
        namespace = {"__name__": "__main__"}
        for cell in notebook["cells"][1 : 1 + cells]:
            source = "".join(cell["source"]).replace("/kaggle", self.root.as_posix())
            lines = source.splitlines()
            for name, value in settings.items():
                matches = [i for i, x in enumerate(lines) if x.startswith(f"{name} = ")]
                if matches:
                    lines[matches[0]] = f"{name} = {value!r}"
            exec("\n".join(lines), namespace)
        return self.commands

    def fetched(self, **settings: object) -> str:
        fetch = [c for c in self.run(**settings) if c[:2] == ["git", "fetch"]]
        assert len(fetch) == 1
        return fetch[0][-1]


@pytest.fixture
def notebook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Notebook:
    return Notebook(tmp_path, monkeypatch)


@pytest.mark.parametrize("gpus", [None, 0, 1, 3])
def test_notebook_needs_two_gpus(notebook: Notebook, gpus: int | None) -> None:
    notebook.gpus = gpus
    with pytest.raises(RuntimeError, match="GPU T4 x2"):
        notebook.run()
    assert notebook.commands == []
    assert not (notebook.root / "temp").exists()
    # The CPU evaluation does not need a GPU.
    assert notebook.run(RUN_TRAIN=False)


def test_notebook_tiny_needs_no_gpu(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook.gpus = None
    monkeypatch.setenv("LITTLEBIT_TINY", "1")
    assert notebook.run()


def test_notebook_launches_two_ranks(
    notebook: Notebook, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    environments = []
    popen = notebook.popen
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: (
            environments.append(kwargs["env"]) or popen(command, **kwargs)
        ),
    )
    training = notebook.run(cells=3)[-1]
    start = training.index("torch.distributed.run")
    assert training[start - 1 : start + 6] == [
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        "2",
        "-m",
        "train.train_qat",
    ]
    assert "--resume" not in training
    assert all("CUDA_VISIBLE_DEVICES" not in env for env in environments)
    with pytest.raises(ValueError, match="even number"):
        notebook.run(LEGACY_RECIPE=True)
    assert notebook.run(LEGACY_RECIPE=True, GRAD_ACCUM=2)


def test_notebook_finds_resume(notebook: Notebook) -> None:
    with pytest.raises(FileNotFoundError, match="found 0"):
        notebook.run(RESUME="attached")
    first = notebook.root / "input/notebooks/owner/first/littlebit-train/resume.pt"
    first.parent.mkdir(parents=True)
    first.touch()
    training = notebook.run(cells=3, RESUME="attached")[-1]
    assert training[training.index("--resume") + 1] == str(first)
    second = notebook.root / "input/second/littlebit-train/resume.pt"
    second.parent.mkdir(parents=True)
    second.touch()
    with pytest.raises(FileNotFoundError, match="found 2"):
        notebook.run(RESUME="attached")
    training = notebook.run(cells=3, RESUME=str(second))[-1]
    assert training[training.index("--resume") + 1] == str(second)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        notebook.run(RESUME=str(second.with_name("other.pt")))


def test_notebook_source_commit(notebook: Notebook) -> None:
    results = notebook.root / "input/session/littlebit-train"
    results.mkdir(parents=True)
    (results / "resume.pt").touch()
    (results / "summary.json").write_text('{"training_commit": "trained"}')
    resume, attached = str(results / "resume.pt"), str(results)
    assert notebook.fetched() == "train-r3"
    assert notebook.fetched(RESUME=resume) == "train-r3"
    assert notebook.fetched(RESUME="attached") == "train-r3"
    assert notebook.fetched(RESUME=resume, TRAINING_COMMIT="chosen") == "chosen"
    evaluate = {"RUN_TRAIN": False, "RESULTS": attached}
    assert notebook.fetched(**evaluate) == "trained"
    assert notebook.fetched(**evaluate, TRAINING_COMMIT="chosen") == "chosen"
    with pytest.raises(ValueError, match="not both"):
        notebook.run(RESUME=resume, RESULTS=attached)
