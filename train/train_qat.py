import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)

if __package__ in ("", None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.data import (
    C4_REV,
    C4_SHARD,
    MODEL,
    MODEL_REV,
    WIKI_REV,
    c4_validation_documents,
    join_documents,
    token_cache,
    tokenize_corpus,
    training_blocks,
    write_c4_validation,
)
from train.qat_model import (
    UPSTREAM_REV,
    export_checkpoint,
    make_student,
    size_plan,
    trainable_elements,
    upstream_layer,
    upstream_recipe,
)
from train.ranks import Ranks

ROOT = Path(__file__).resolve().parents[1]
PEAK_LR = 8e-5
WARMUP = 0.02
MSE_WEIGHT = 10.0
CHECK_INTERVAL = 30 * 60
PROGRESS_WINDOWS = 4
NORM_TOLERANCE = 1e-2
GRADIENT_TOLERANCE = 1e-1


def logit_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    head: torch.Tensor,
    teacher_head: torch.Tensor,
) -> torch.Tensor:
    prediction = F.linear(student, head).float().log_softmax(-1)
    target = F.linear(teacher, teacher_head).float().log_softmax(-1)
    return F.kl_div(prediction, target, reduction="sum", log_target=True)


def legacy_distillation(
    student: PreTrainedModel,
    teacher: PreTrainedModel,
    ids: torch.Tensor,
    logit_chunk: int,
    mixed_precision: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(
        ids.device.type,
        dtype=torch.float16,
        enabled=mixed_precision,
        cache_enabled=False,
    ):
        with torch.no_grad():
            target = teacher.model(ids, use_cache=False, output_hidden_states=True)
        prediction = student.model(ids, use_cache=False, output_hidden_states=True)
        hidden_mse = torch.stack(
            [
                F.mse_loss(got.float(), expected.float())
                for got, expected in zip(
                    prediction.hidden_states[1:], target.hidden_states[1:], strict=True
                )
            ]
        ).mean()
        total_kl = torch.zeros((), device=ids.device)
        head = student.lm_head.weight
        if mixed_precision:
            head = head.half()
        for got, expected in zip(
            prediction.last_hidden_state[0].split(logit_chunk),
            target.last_hidden_state[0].split(logit_chunk),
            strict=True,
        ):
            total_kl = total_kl + checkpoint(
                logit_kl,
                got,
                expected,
                head,
                teacher.lm_head.weight,
                use_reentrant=False,
            )
    return total_kl / ids.numel(), hidden_mse


class Trainer:
    def __init__(
        self,
        student: PreTrainedModel,
        teacher: PreTrainedModel,
        learning_rate: float,
        *,
        recipe: ModuleType | None,
        total_steps: int,
        grad_accum: int,
        ranks: Ranks | None = None,
    ) -> None:
        self.ranks = Ranks(student.device) if ranks is None else ranks
        if grad_accum % self.ranks.size:
            raise ValueError(
                f"{grad_accum} sequences per step do not divide over "
                f"{self.ranks.size} ranks"
            )
        self.student = student.train()
        self.teacher = teacher.eval().requires_grad_(False)
        self.student.config.use_cache = self.teacher.config.use_cache = False
        self.recipe = recipe
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.grad_accum = grad_accum
        self.parameters = {
            name: p for name, p in student.named_parameters() if p.requires_grad
        }
        self.optim = torch.optim.Adam(
            list(self.parameters.values()), lr=learning_rate, foreach=False
        )
        self.mixed_precision = student.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.step = 0
        self.check_first_step = self.ranks.size > 1

    def loss(
        self, ids: torch.Tensor, logit_chunk: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.recipe is None:
            kl, mse = legacy_distillation(
                self.student, self.teacher, ids, logit_chunk, self.mixed_precision
            )
            return kl + mse, kl, mse
        with torch.autocast(
            ids.device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision,
            cache_enabled=False,
        ):
            loss, kl, mse, _ = self.recipe.kd_loss(
                self.student,
                self.teacher,
                ids,
                lam=MSE_WEIGHT,
                mse_norm=True,
                mse_skip=1,
                chunk=logit_chunk,
            )
        return loss, kl, mse

    def accumulate(
        self, batches: torch.Tensor, logit_chunk: int, rows: range
    ) -> torch.Tensor:
        """Add the scaled gradients of these rows and return their loss terms."""
        terms = torch.zeros(len(batches), 3, dtype=torch.float64)
        for row in rows:
            ids = batches[row : row + 1].to(
                device=self.student.device, dtype=torch.long
            )
            loss, kl, mse = self.loss(ids, logit_chunk)
            self.scaler.scale(loss / self.grad_accum).backward()
            for column, value in enumerate((loss, kl, mse)):
                terms[row, column] = value.item()
        return terms

    def compare_gradients(self, batches: torch.Tensor, logit_chunk: int) -> None:
        """Compare the summed gradients with rank 0 computing every sequence."""
        parameters = list(self.parameters.values())
        device = self.student.device
        norms = torch.zeros(3, dtype=torch.float64, device=device)
        if self.ranks.rank == 0:
            reduced = [p.grad for p in parameters]
            self.optim.zero_grad(set_to_none=True)
            devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                self.accumulate(batches, logit_chunk, range(self.grad_accum))
            for p, gradient in zip(parameters, reduced, strict=True):
                alone, gradient = p.grad.double(), gradient.double()
                norms[0] += alone.square().sum()
                norms[1] += gradient.square().sum()
                norms[2] += (alone - gradient).square().sum()
            norms.sqrt_()
            for p, gradient in zip(parameters, reduced, strict=True):
                p.grad = gradient
        self.ranks.broadcast_all([norms])
        alone, total, difference = norms.tolist()
        if not math.isfinite(alone) and not math.isfinite(total):
            print(
                f"Step {self.step + 1}: fp16 overflow, the first-step check moves "
                "to the next step",
                flush=True,
            )
            return
        self.check_first_step = False
        norm_error = abs(alone - total) / alone if alone else total
        gradient_error = difference / alone if alone else difference
        print(
            json.dumps(
                {
                    "first_step_check": self.step + 1,
                    "relative_norm_difference": norm_error,
                    "relative_gradient_difference": gradient_error,
                }
            ),
            flush=True,
        )
        if not (norm_error <= NORM_TOLERANCE and gradient_error <= GRADIENT_TOLERANCE):
            raise RuntimeError(
                f"first-step check failed: {self.ranks.size} ranks give gradient "
                f"norm {total}, rank 0 alone gives {alone}"
            )

    def train_step(
        self, batches: torch.Tensor, logit_chunk: int
    ) -> tuple[float, float, float]:
        if len(batches) != self.grad_accum:
            raise ValueError("optimizer step needs one sequence per accumulation pass")
        self.student.train()
        if self.recipe is not None:
            self.optim.param_groups[0]["lr"] = self.recipe.lr_at(
                self.step, self.total_steps, self.learning_rate, WARMUP
            )
        self.optim.zero_grad(set_to_none=True)
        rows = range(self.ranks.rank, self.grad_accum, self.ranks.size)
        terms = self.accumulate(batches, logit_chunk, rows).to(self.ranks.device)
        total_loss = total_kl = total_mse = 0.0
        for loss, kl, mse in self.ranks.sum(terms).tolist():
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {self.step + 1}")
            total_loss += loss
            total_kl += kl
            total_mse += mse
        # Every rank unscales the same sum, so the overflow decision is shared too.
        self.ranks.sum_all([p.grad for p in self.parameters.values()])
        if self.check_first_step:
            self.compare_gradients(batches, logit_chunk)
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(
            list(self.parameters.values()),
            1.0,
            error_if_nonfinite=not self.mixed_precision,
        )
        self.scaler.step(self.optim)
        scale = self.scaler.get_scale()
        self.scaler.update()
        if self.scaler.get_scale() < scale:
            print(
                f"Step {self.step + 1}: fp16 overflow, reduced loss scale", flush=True
            )
        self.step += 1
        self.optim.zero_grad(set_to_none=True)
        return (
            total_loss / self.grad_accum,
            total_kl / self.grad_accum,
            total_mse / self.grad_accum,
        )

    def synchronize(self) -> None:
        """Copy rank 0's complete training state to the other ranks."""
        if self.ranks.size == 1:
            return
        self.ranks.wait()
        optimizer = self.optim.state_dict()
        header = self.ranks.share(
            {
                "step": self.step,
                "scaler": self.scaler.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": self.cuda_rng(),
                "groups": optimizer["param_groups"],
                "steps": [entry["step"] for entry in optimizer["state"].values()],
            }
        )
        self.ranks.broadcast_all(
            [p.data for p in self.student.parameters()] + list(self.student.buffers())
        )
        moments: dict[int, dict[str, torch.Tensor]] = {}
        for index, (step, p) in enumerate(
            zip(header["steps"], self.parameters.values(), strict=False)
        ):
            entry = optimizer["state"].get(index, {})
            moments[index] = {"step": step} | {
                name: entry[name] if entry else torch.zeros_like(p)
                for name in ("exp_avg", "exp_avg_sq")
            }
        for name in ("exp_avg", "exp_avg_sq"):
            self.ranks.broadcast_all([entry[name] for entry in moments.values()])
        if self.ranks.rank:
            self.step = header["step"]
            self.optim.load_state_dict(
                {"state": moments, "param_groups": header["groups"]}
            )
            self.scaler.load_state_dict(header["scaler"])
            self.set_rng(header["rng"], header["cuda_rng"])

    def check_agreement(self) -> None:
        self.ranks.check_equal(self.parameters, self.step)

    def cuda_rng(self) -> torch.Tensor | None:
        if not self.mixed_precision:
            return None
        return torch.cuda.get_rng_state(self.student.device)

    def set_rng(self, rng: torch.Tensor, cuda_rng: torch.Tensor | None) -> None:
        torch.set_rng_state(rng)
        if self.mixed_precision:
            torch.cuda.set_rng_state(cuda_rng, self.student.device)

    def save(self, path: Path, signature: dict) -> None:
        state = {
            "signature": signature,
            "step": self.step,
            "parameters": {name: p.detach() for name, p in self.parameters.items()},
            "optimizer": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": self.cuda_rng(),
        }
        with tempfile.TemporaryDirectory(prefix=".resume-", dir=path.parent) as temp:
            pending = Path(temp) / "resume.pt"
            torch.save(state, pending)
            pending.replace(path)

    def restore(self, path: Path, signature: dict) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (
            state["signature"] != signature
            or state["parameters"].keys() != self.parameters.keys()
        ):
            raise ValueError(
                "resume checkpoint does not match the model or training settings"
            )
        with torch.no_grad():
            for name, p in self.parameters.items():
                p.copy_(state["parameters"][name])
        self.optim.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.set_rng(state["rng"], state["cuda_rng"])
        self.step = state["step"]


@torch.inference_mode()
def perplexity(
    model: PreTrainedModel,
    ids: torch.Tensor,
    seq_len: int,
    logit_chunk: int,
    ranks: Ranks | None = None,
) -> float:
    model.eval()
    device = model.device
    mixed_precision = device.type == "cuda"
    total = torch.zeros((), dtype=torch.float64, device=device)
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=mixed_precision, cache_enabled=False
    ):
        head = model.lm_head.weight.half() if mixed_precision else model.lm_head.weight
        offsets = range(0, len(ids) - 1, seq_len - 1)
        if ranks is not None:
            offsets = offsets[ranks.rank :: ranks.size]
        for offset in offsets:
            block = ids[offset : offset + seq_len].to(device=device, dtype=torch.long)
            hidden = model.model(
                block[:-1][None, :], use_cache=False
            ).last_hidden_state[0]
            for features, expected in zip(
                hidden.split(logit_chunk), block[1:].split(logit_chunk), strict=True
            ):
                logits = F.linear(features, head).float()
                total += F.cross_entropy(logits, expected, reduction="sum").double()
    if ranks is not None:
        ranks.sum(total)
    return math.exp(total.item() / (len(ids) - 1))


def run(args: argparse.Namespace) -> None:
    ranks = Ranks.launch(args.cpu)
    with contextlib.ExitStack() as stack:
        if ranks.rank:
            quiet = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
            stack.enter_context(contextlib.redirect_stdout(quiet))
        try:
            train(args, ranks)
        except BaseException:
            if ranks.size == 1:
                raise
            # A normal exit could wait for a collective the other rank never joins.
            traceback.print_exc()
            sys.stderr.flush()
            os._exit(1)
        ranks.close()


def train(args: argparse.Namespace, ranks: Ranks) -> None:
    started = time.time() if args.session_start is None else args.session_start
    deadline = started + args.minutes * 60
    device = ranks.device
    if args.grad_accum % ranks.size:
        raise ValueError(
            f"--grad-accum {args.grad_accum} must be divisible by {ranks.size} ranks"
        )
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    ranks.check_links()
    output = args.output.resolve()
    if ranks.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if any(
            (output / name).exists()
            for name in ("resume.pt", "littlebit", "summary.json")
        ):
            raise ValueError("use an output directory without a previous run's files")
    with ranks.first():
        upstream = upstream_layer(args.upstream)
        recipe = None if args.legacy_recipe else upstream_recipe(args.upstream)
    tokenizer_model = None
    cache_signature = {
        "model": MODEL_REV,
        "datasets": [C4_REV, WIKI_REV],
        "shard": C4_SHARD,
        "c4_tokens": args.c4_tokens,
        "tiny": {"seed": args.seed, "seq_len": args.seq_len} if args.tiny else None,
    }
    if args.tiny:
        sys.path.insert(0, str(ROOT / "tests"))
        from tiny_model import make_tiny_model

        with tempfile.TemporaryDirectory(dir=output) as temp:
            source = Path(ranks.share(temp)) / "tiny"
            if ranks.rank == 0:
                make_tiny_model(source, architecture="qwen3")
            ranks.wait()
            teacher = AutoModelForCausalLM.from_pretrained(source)
            tokenizer = AutoTokenizer.from_pretrained(source)
            tokenizer_model = (source / "tokenizer.model").read_bytes()
            ranks.wait()
        config = teacher.config
        validation_documents = ["The quick brown fox.", "Packing signs is fun."]
    else:
        with ranks.first():
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL, revision=MODEL_REV, token=False
            )
            config = AutoConfig.from_pretrained(MODEL, revision=MODEL_REV, token=False)
            validation_documents = c4_validation_documents()
    ranks.time_reduction(trainable_elements(config))

    def corpus() -> dict[str, torch.Tensor]:
        if not args.tiny:
            return tokenize_corpus(tokenizer, args.c4_tokens)
        generator = torch.Generator().manual_seed(args.seed)
        ids = torch.randint(
            0,
            teacher.config.vocab_size,
            (4, args.seq_len),
            generator=generator,
            dtype=torch.int32,
        )
        return {
            "train": ids.flatten(),
            "test": ids[0].clone(),
            "c4": torch.empty(0, dtype=torch.int32),
        }

    cache_source = args.token_cache
    if cache_source is None and args.resume is not None:
        sibling = args.resume.with_name("tokens.pt")
        if sibling.is_file():
            cache_source = sibling
    with ranks.first():
        tokens = token_cache(
            output / "tokens.pt",
            None if ranks.rank else cache_source,
            cache_signature,
            corpus,
        )
    wiki_repeat = 1 if args.legacy_recipe or args.tiny else 4
    blocks, order = training_blocks(tokens, args.seq_len, args.seed, wiki_repeat)
    if args.tiny:
        order = torch.arange(len(blocks))
    else:
        with ranks.first():
            teacher = AutoModelForCausalLM.from_pretrained(
                MODEL, revision=MODEL_REV, token=False, dtype="auto"
            )
    test_ids = tokens["test"]
    c4_ids = join_documents(tokenizer, validation_documents)
    if ranks.rank == 0:
        write_c4_validation(output / "c4-validation.txt", validation_documents)
    print(
        f"C4 validation: {len(validation_documents)} documents, {len(c4_ids)} tokens",
        flush=True,
    )
    tokens_per_step = args.seq_len * args.grad_accum
    total_steps = args.tokens // tokens_per_step
    signature = {
        "corpus": cache_signature,
        "upstream": UPSTREAM_REV,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "wiki_repeat": wiki_repeat,
        "grad_accum": args.grad_accum,
        "recipe": "legacy" if args.legacy_recipe else "upstream",
        "schedule_steps": None if args.legacy_recipe else total_steps,
        "learning_rate": args.learning_rate,
        "device_type": device.type,
        "plan": size_plan(teacher.config),
    }
    print(json.dumps(signature["plan"], indent=2), flush=True)
    print("Preparing LittleBit-2 factors", flush=True)
    teacher.to(device)
    # Other ranks receive rank 0's factors, so they skip the initialization.
    student = make_student(
        teacher, upstream, resume=args.resume is not None or ranks.rank > 0
    )
    teacher.to(dtype=torch.float32 if args.cpu else torch.float16)
    student_use_cache = student.config.use_cache
    trainer = Trainer(
        student,
        teacher,
        args.learning_rate,
        recipe=recipe,
        total_steps=total_steps,
        grad_accum=args.grad_accum,
        ranks=ranks,
    )
    if args.resume and ranks.rank == 0:
        trainer.restore(args.resume, signature)
    trainer.synchronize()
    start_step = trainer.step
    if total_steps < start_step:
        raise ValueError(
            "--tokens must cover the tokens already consumed in the resume file"
        )
    names = ranks.gather("CPU" if args.cpu else torch.cuda.get_device_name(device))
    print(
        f"Starting at step {start_step} with {total_steps} total steps, "
        f"{tokens_per_step} tokens per step, setup {time.time() - started:.1f}s, "
        f"{'device' if ranks.size == 1 else f'{ranks.size} ranks'} {', '.join(names)}",
        flush=True,
    )

    progress_ids = test_ids[: 1 + PROGRESS_WINDOWS * (args.seq_len - 1)]
    progress_windows = math.ceil((len(progress_ids) - 1) / (args.seq_len - 1))

    def check_progress(save: bool) -> None:
        trainer.check_agreement()
        if save and ranks.rank == 0:
            trainer.save(output / "resume.pt", signature)
            print(f"Saved resume.pt at step {trainer.step}", flush=True)
        score = perplexity(student, progress_ids, args.seq_len, args.logit_chunk, ranks)
        print(
            json.dumps(
                {
                    "step": trainer.step,
                    "tokens": trainer.step * tokens_per_step,
                    "progress_windows": progress_windows,
                    "student_progress_perplexity": score,
                }
            ),
            flush=True,
        )

    if trainer.step < total_steps and ranks.agree(time.time() < deadline):
        check_progress(save=False)
    train_start = time.monotonic()
    next_check = train_start + CHECK_INTERVAL
    last_step_seconds = 0.0
    latest: dict = {}
    while trainer.step < total_steps and ranks.agree(
        time.time() + last_step_seconds < deadline
    ):
        step_start = time.monotonic()
        start = trainer.step * args.grad_accum
        positions = torch.arange(start, start + args.grad_accum) % len(order)
        loss, kl, mse = trainer.train_step(blocks[order[positions]], args.logit_chunk)
        last_step_seconds = time.monotonic() - step_start
        # The step number alone decides, so every rank takes part in the same check.
        if trainer.step == start_step + 1 or trainer.step % args.log_every == 0:
            trainer.check_agreement()
        elapsed = time.monotonic() - train_start
        rate = (trainer.step - start_step) * tokens_per_step / elapsed
        latest = {
            "step": trainer.step,
            "tokens": trainer.step * tokens_per_step,
            "loss": loss,
            "kl": kl,
            "mse": mse,
            "learning_rate": trainer.optim.param_groups[0]["lr"],
            "tokens_per_second": rate,
            "token_budget_eta_seconds": (total_steps - trainer.step)
            * tokens_per_step
            / rate,
        }
        if (
            trainer.step in (start_step + 1, total_steps)
            or trainer.step % args.log_every == 0
        ):
            print(json.dumps(latest), flush=True)
        if ranks.agree(time.monotonic() >= next_check):
            check_progress(save=True)
            next_check += CHECK_INTERVAL

    trainer.check_agreement()
    student.config.use_cache = student_use_cache
    export = {}
    if ranks.rank == 0:
        trainer.save(output / "resume.pt", signature)
        export = export_checkpoint(
            student, tokenizer, output / "littlebit", tokenizer_model=tokenizer_model
        )
    ranks.wait()
    scores = {
        f"{name}{label}_perplexity": perplexity(
            model, ids, args.seq_len, args.logit_chunk, ranks
        )
        for label, ids in (("", test_ids), ("_c4_validation", c4_ids))
        for name, model in (("teacher", teacher), ("student", student))
    }
    peaks = ranks.gather(None if args.cpu else torch.cuda.max_memory_allocated(device))
    if ranks.rank:
        return
    result = {
        "training_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "initial_step": start_step,
        "final_step": trainer.step,
        "training_tokens": trainer.step * tokens_per_step,
        "tokens_per_step": tokens_per_step,
        "recipe": signature["recipe"],
        "schedule_steps": signature["schedule_steps"],
        "stop": "token budget" if trainer.step == total_steps else "time budget",
        "data": "tiny random fixture" if args.tiny else "WikiText-2 test",
        "last_step": latest,
        "c4_validation_documents": len(validation_documents),
        "c4_validation_tokens": len(c4_ids),
        **scores,
        "plan": signature["plan"],
        **export,
        "elapsed_seconds": time.time() - started,
        "ranks": ranks.size,
    }
    if not args.cpu:
        result["gpu"] = names[0]
        result["gpu_peak_allocated_bytes"] = peaks[0]
        result["gpus"] = names
        result["gpu_peak_allocated_bytes_per_rank"] = peaks
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a format-exact LittleBit-2 Qwen3 student."
    )
    parser.add_argument("--output", type=Path, default=Path("run"))
    parser.add_argument(
        "--upstream", type=Path, default=Path("/tmp/littlebit-upstream")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--legacy-recipe", action="store_true")
    parser.add_argument("--tokens", type=int, help="cumulative input token target")
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--minutes", type=float, default=630.0)
    parser.add_argument("--session-start", type=float)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--c4-tokens", type=int, help="0 uses the entire shard")
    parser.add_argument("--logit-chunk", type=int, default=128)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--tiny", action="store_true", help="use a random, offline model and corpus"
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print ranks and projection sizes from config only",
    )
    args = parser.parse_args()
    if args.plan:
        config = AutoConfig.from_pretrained(MODEL, revision=MODEL_REV, token=False)
        print(json.dumps(size_plan(config), indent=2))
        return
    defaults = {
        "tokens": 2_097_152 if args.legacy_recipe else 100_000_000,
        "grad_accum": 1 if args.legacy_recipe else 2,
        "c4_tokens": 8_388_608 if args.legacy_recipe else 0,
        "learning_rate": 3e-5 if args.legacy_recipe else PEAK_LR,
        "seed": 7 if args.legacy_recipe else 0,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if not 2 <= args.seq_len <= (256 if args.tiny else 40960):
        parser.error("sequence length is outside the model's supported range")
    if (
        args.grad_accum < 1
        or args.tokens < args.seq_len * args.grad_accum
        or not 0 < args.minutes <= 660
        or args.c4_tokens < 0
        or min(args.threads, args.logit_chunk, args.log_every) < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error("invalid token, time, or training settings")
    run(args)


if __name__ == "__main__":
    main()
