import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
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
    token_cache,
    tokenize_corpus,
    training_blocks,
)
from train.qat_model import (
    UPSTREAM_REV,
    export_checkpoint,
    make_student,
    size_plan,
    upstream_layer,
    upstream_recipe,
)

ROOT = Path(__file__).resolve().parents[1]
PEAK_LR = 8e-5
WARMUP = 0.02
MSE_WEIGHT = 10.0
CHECK_INTERVAL = 30 * 60
PROGRESS_WINDOWS = 4


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
    ) -> None:
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
        total_loss = total_kl = total_mse = 0.0
        for batch in batches.split(1):
            ids = batch.to(device=self.student.device, dtype=torch.long)
            loss, kl, mse = self.loss(ids, logit_chunk)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"non-finite loss at step {self.step + 1}")
            self.scaler.scale(loss / self.grad_accum).backward()
            total_loss += loss.item()
            total_kl += kl.item()
            total_mse += mse.item()
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

    def save(self, path: Path, signature: dict) -> None:
        state = {
            "signature": signature,
            "step": self.step,
            "parameters": {name: p.detach() for name, p in self.parameters.items()},
            "optimizer": self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if self.mixed_precision else None,
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
        torch.set_rng_state(state["rng"])
        if self.mixed_precision:
            torch.cuda.set_rng_state(state["cuda_rng"])
        self.step = state["step"]


@torch.inference_mode()
def perplexity(
    model: PreTrainedModel, ids: torch.Tensor, seq_len: int, logit_chunk: int
) -> float:
    model.eval()
    device = model.device
    mixed_precision = device.type == "cuda"
    total = torch.zeros((), dtype=torch.float64, device=device)
    with torch.autocast(
        device.type, dtype=torch.float16, enabled=mixed_precision, cache_enabled=False
    ):
        head = model.lm_head.weight.half() if mixed_precision else model.lm_head.weight
        for offset in range(0, len(ids) - 1, seq_len - 1):
            block = ids[offset : offset + seq_len].to(device=device, dtype=torch.long)
            hidden = model.model(
                block[:-1][None, :], use_cache=False
            ).last_hidden_state[0]
            for features, expected in zip(
                hidden.split(logit_chunk), block[1:].split(logit_chunk), strict=True
            ):
                logits = F.linear(features, head).float()
                total += F.cross_entropy(logits, expected, reduction="sum").double()
    return math.exp(total.item() / (len(ids) - 1))


def run(args: argparse.Namespace) -> None:
    started = time.time() if args.session_start is None else args.session_start
    deadline = started + args.minutes * 60
    device = torch.device("cpu" if args.cpu else "cuda:0")
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("enable a CUDA GPU for training or use --cpu --tiny")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists() for name in ("resume.pt", "littlebit", "summary.json")
    ):
        raise ValueError("use an output directory without a previous run's files")
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
            source = Path(temp) / "tiny"
            make_tiny_model(source, architecture="qwen3")
            teacher = AutoModelForCausalLM.from_pretrained(source)
            tokenizer = AutoTokenizer.from_pretrained(source)
            tokenizer_model = (source / "tokenizer.model").read_bytes()
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=MODEL_REV, token=False
        )

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
    tokens = token_cache(output / "tokens.pt", cache_source, cache_signature, corpus)
    wiki_repeat = 1 if args.legacy_recipe or args.tiny else 4
    blocks, order = training_blocks(tokens, args.seq_len, args.seed, wiki_repeat)
    if args.tiny:
        order = torch.arange(len(blocks))
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=MODEL_REV, token=False, dtype="auto"
        )
    test_ids = tokens["test"]
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
    student = make_student(teacher, upstream, resume=args.resume is not None)
    teacher.to(dtype=torch.float32 if args.cpu else torch.float16)
    student_use_cache = student.config.use_cache
    trainer = Trainer(
        student,
        teacher,
        args.learning_rate,
        recipe=recipe,
        total_steps=total_steps,
        grad_accum=args.grad_accum,
    )
    if args.resume:
        trainer.restore(args.resume, signature)
    start_step = trainer.step
    if total_steps < start_step:
        raise ValueError(
            "--tokens must cover the tokens already consumed in the resume file"
        )
    print(
        f"Starting at step {start_step} with {total_steps} total steps, "
        f"{tokens_per_step} tokens per step, setup {time.time() - started:.1f}s, device "
        f"{'CPU' if args.cpu else torch.cuda.get_device_name(device)}",
        flush=True,
    )

    progress_ids = test_ids[: 1 + PROGRESS_WINDOWS * (args.seq_len - 1)]
    progress_windows = math.ceil((len(progress_ids) - 1) / (args.seq_len - 1))

    def check_progress() -> None:
        print(
            json.dumps(
                {
                    "step": trainer.step,
                    "tokens": trainer.step * tokens_per_step,
                    "progress_windows": progress_windows,
                    "student_progress_perplexity": perplexity(
                        student, progress_ids, args.seq_len, args.logit_chunk
                    ),
                }
            ),
            flush=True,
        )

    if trainer.step < total_steps and time.time() < deadline:
        check_progress()
    train_start = time.monotonic()
    next_check = train_start + CHECK_INTERVAL
    last_step_seconds = 0.0
    latest: dict = {}
    while trainer.step < total_steps and time.time() + last_step_seconds < deadline:
        step_start = time.monotonic()
        start = trainer.step * args.grad_accum
        positions = torch.arange(start, start + args.grad_accum) % len(order)
        loss, kl, mse = trainer.train_step(blocks[order[positions]], args.logit_chunk)
        last_step_seconds = time.monotonic() - step_start
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
        if time.monotonic() >= next_check:
            trainer.save(output / "resume.pt", signature)
            print(f"Saved resume.pt at step {trainer.step}", flush=True)
            check_progress()
            next_check += CHECK_INTERVAL

    trainer.save(output / "resume.pt", signature)
    student.config.use_cache = student_use_cache
    export = export_checkpoint(
        student, tokenizer, output / "littlebit", tokenizer_model=tokenizer_model
    )
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
        "teacher_perplexity": perplexity(
            teacher, test_ids, args.seq_len, args.logit_chunk
        ),
        "student_perplexity": perplexity(
            student, test_ids, args.seq_len, args.logit_chunk
        ),
        "plan": signature["plan"],
        **export,
        "elapsed_seconds": time.time() - started,
    }
    if not args.cpu:
        result["gpu"] = torch.cuda.get_device_name(device)
        result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
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
