import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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

from train.data import C4_REV, MODEL, MODEL_REV, WIKI_REV, training_blocks, wikitext
from train.qat_model import (
    UPSTREAM_REV,
    export_checkpoint,
    make_student,
    size_plan,
    upstream_layer,
)

ROOT = Path(__file__).resolve().parents[1]


def logit_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    head: torch.Tensor,
    teacher_head: torch.Tensor,
) -> torch.Tensor:
    prediction = F.linear(student, head).float().log_softmax(-1)
    target = F.linear(teacher, teacher_head).float().log_softmax(-1)
    return F.kl_div(prediction, target, reduction="sum", log_target=True)


def distillation(
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
        self, student: PreTrainedModel, teacher: PreTrainedModel, learning_rate: float
    ) -> None:
        self.student = student.train()
        self.teacher = teacher.eval().requires_grad_(False)
        self.parameters = {
            name: p for name, p in student.named_parameters() if p.requires_grad
        }
        self.optim = torch.optim.Adam(
            list(self.parameters.values()), lr=learning_rate, foreach=False
        )
        self.mixed_precision = student.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.step = 0

    def train_step(self, ids: torch.Tensor, logit_chunk: int) -> tuple[float, float]:
        self.optim.zero_grad(set_to_none=True)
        kl, mse = distillation(
            self.student, self.teacher, ids, logit_chunk, self.mixed_precision
        )
        loss = kl + mse
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"non-finite loss at step {self.step + 1}")
        self.scaler.scale(loss).backward()
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
        return kl.item(), mse.item()

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
            block = ids[offset : offset + seq_len].to(device)
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
    tokenizer_model = None
    if args.tiny:
        sys.path.insert(0, str(ROOT / "tests"))
        from tiny_model import make_tiny_model

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "tiny"
            make_tiny_model(source, architecture="qwen3")
            teacher = AutoModelForCausalLM.from_pretrained(source)
            tokenizer = AutoTokenizer.from_pretrained(source)
            tokenizer_model = (source / "tokenizer.model").read_bytes()
        generator = torch.Generator().manual_seed(args.seed)
        blocks = torch.randint(
            0, teacher.config.vocab_size, (4, args.seq_len), generator=generator
        )
        order = torch.arange(len(blocks))
        test_ids = blocks[0].long()
    else:
        teacher = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=MODEL_REV, token=False, dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=MODEL_REV, token=False
        )
        blocks, order = training_blocks(
            tokenizer, args.seq_len, args.c4_tokens, args.seed
        )
        test_ids = tokenizer(
            wikitext("test"),
            add_special_tokens=False,
            return_tensors="pt",
            verbose=False,
        ).input_ids.flatten()

    total_steps = args.tokens // args.seq_len
    signature = {
        "tiny": args.tiny,
        "model": MODEL_REV,
        "datasets": [C4_REV, WIKI_REV],
        "upstream": UPSTREAM_REV,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "c4_tokens": args.c4_tokens,
        "learning_rate": args.learning_rate,
        "device_type": device.type,
        "plan": size_plan(teacher.config),
    }
    print(json.dumps(signature["plan"], indent=2), flush=True)
    print("Preparing LittleBit-2 factors", flush=True)
    teacher.to(device)
    student = make_student(teacher, upstream, resume=args.resume is not None)
    teacher.to(dtype=torch.float32 if args.cpu else torch.float16)
    trainer = Trainer(student, teacher, args.learning_rate)
    if args.resume:
        trainer.restore(args.resume, signature)
    start_step = trainer.step
    if total_steps < start_step:
        raise ValueError(
            "--tokens must cover the tokens already consumed in the resume file"
        )
    print(
        f"Starting at step {start_step} with {total_steps} total steps, "
        f"setup {time.time() - started:.1f}s, device "
        f"{'CPU' if args.cpu else torch.cuda.get_device_name(device)}",
        flush=True,
    )

    train_start = time.monotonic()
    next_save = train_start + 30 * 60
    last_step_seconds = 0.0
    latest: dict = {}
    while trainer.step < total_steps and time.time() + last_step_seconds < deadline:
        step_start = time.monotonic()
        index = order[trainer.step % len(order)]
        ids = blocks[index][None, :].to(device=device, dtype=torch.long)
        kl, mse = trainer.train_step(ids, args.logit_chunk)
        last_step_seconds = time.monotonic() - step_start
        elapsed = time.monotonic() - train_start
        rate = (trainer.step - start_step) * args.seq_len / elapsed
        latest = {
            "step": trainer.step,
            "tokens": trainer.step * args.seq_len,
            "loss": kl + mse,
            "kl": kl,
            "mse": mse,
            "tokens_per_second": rate,
            "token_budget_eta_seconds": (total_steps - trainer.step)
            * args.seq_len
            / rate,
        }
        if trainer.step == start_step + 1 or trainer.step % args.log_every == 0:
            print(json.dumps(latest), flush=True)
        if time.monotonic() >= next_save:
            trainer.save(output / "resume.pt", signature)
            print(f"Saved resume.pt at step {trainer.step}", flush=True)
            next_save = time.monotonic() + 30 * 60

    trainer.save(output / "resume.pt", signature)
    export = export_checkpoint(
        student, tokenizer, output / "littlebit", tokenizer_model=tokenizer_model
    )
    result = {
        "training_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "initial_step": start_step,
        "final_step": trainer.step,
        "training_tokens": trainer.step * args.seq_len,
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
    parser.add_argument("--tokens", type=int, default=2_097_152)
    parser.add_argument("--minutes", type=float, default=630.0)
    parser.add_argument("--session-start", type=float)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument(
        "--c4-tokens", type=int, default=8_388_608, help="0 uses the entire shard"
    )
    parser.add_argument("--logit-chunk", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=7)
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
    if not 2 <= args.seq_len <= (256 if args.tiny else 40960):
        parser.error("sequence length is outside the model's supported range")
    if (
        args.tokens < args.seq_len
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
