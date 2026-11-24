import os
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

SETUP_TIMEOUT = timedelta(hours=2)
STEP_TIMEOUT = timedelta(minutes=10)
PROBE_SECONDS = 120.0
BUCKET_ELEMENTS = 16 * 1024 * 1024


def buckets(tensors: Iterable[torch.Tensor]) -> Iterator[list[torch.Tensor]]:
    pending: list[torch.Tensor] = []
    elements = 0
    for tensor in tensors:
        if pending and (
            tensor.dtype != pending[0].dtype
            or elements + tensor.numel() > BUCKET_ELEMENTS
        ):
            yield pending
            pending, elements = [], 0
        pending.append(tensor)
        elements += tensor.numel()
    if pending:
        yield pending


@contextmanager
def abort_after(seconds: float, message: str) -> Iterator[None]:
    """Kill this process if the block hangs, even inside a blocked collective."""

    def abort() -> None:
        print(message, file=sys.stderr, flush=True)
        os._exit(1)

    timer = threading.Timer(seconds, abort)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


class Ranks:
    """One process per GPU. With a single process every method is a no-op."""

    def __init__(self, device: torch.device, rank: int = 0, size: int = 1) -> None:
        self.device = device
        self.rank = rank
        self.size = size
        self.setup_group: dist.ProcessGroup | None = None

    @classmethod
    def launch(cls, cpu: bool) -> "Ranks":
        rank = int(os.environ.get("RANK", "0"))
        size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cpu" if cpu else f"cuda:{local_rank}")
        if not cpu:
            if not torch.cuda.is_available():
                raise RuntimeError("enable a CUDA GPU for training or use --cpu --tiny")
            if torch.cuda.device_count() <= local_rank:
                raise RuntimeError(
                    f"rank {rank} needs GPU {local_rank}, but only "
                    f"{torch.cuda.device_count()} visible"
                )
            torch.cuda.set_device(device)
        ranks = cls(device, rank, size)
        if size > 1:
            if sys.platform == "linux":
                os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            with ranks.probe("joining the process group"):
                dist.init_process_group("gloo" if cpu else "nccl", timeout=STEP_TIMEOUT)
                ranks.setup_group = dist.new_group(
                    backend="gloo", timeout=SETUP_TIMEOUT
                )
        return ranks

    def close(self) -> None:
        if self.size > 1:
            dist.destroy_process_group()

    def probe(self, action: str):
        return abort_after(
            PROBE_SECONDS,
            f"Rank {self.rank}: {action} did not finish in {PROBE_SECONDS:.0f} s",
        )

    def check_links(self) -> None:
        if self.size == 1:
            return
        with self.probe("the small all-reduce"):
            started = time.monotonic()
            total = self.sum(torch.ones(4, device=self.device)).tolist()
            self.wait()
            seconds = time.monotonic() - started
        if total != [float(self.size)] * 4:
            raise RuntimeError(f"all-reduce over {self.size} ranks returned {total}")
        print(f"All-reduce check: {self.size} ranks, {seconds:.3f} s", flush=True)

    def time_reduction(self, elements: int) -> None:
        if self.size == 1:
            return
        sizes = [BUCKET_ELEMENTS] * (elements // BUCKET_ELEMENTS)
        if elements % BUCKET_ELEMENTS:
            sizes.append(elements % BUCKET_ELEMENTS)
        self.wait()
        with self.probe(f"an all-reduce of {elements} gradient elements"):
            gradients = [torch.zeros(size, device=self.device) for size in sizes]
            self.sync_device()
            started = time.monotonic()
            self.sum_all(gradients)
            self.sync_device()
            seconds = time.monotonic() - started
        size = elements * 4
        print(
            f"Gradient all-reduce: {size} bytes in {seconds:.3f} s "
            f"({size / seconds / 1e9:.2f} GB/s)",
            flush=True,
        )

    def sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def wait(self) -> None:
        """Wait for work that only one rank does, such as tokenizing."""
        if self.size > 1:
            dist.barrier(group=self.setup_group)

    @contextmanager
    def first(self) -> Iterator[None]:
        """Run the block on rank 0, then on the other ranks."""
        if self.rank:
            self.wait()
        yield
        if not self.rank:
            self.wait()

    def share(self, value: Any) -> Any:
        """Give every rank the object rank 0 holds."""
        if self.size == 1:
            return value
        holder = [value]
        dist.broadcast_object_list(
            holder, src=0, group=self.setup_group, device=torch.device("cpu")
        )
        return holder[0]

    def gather(self, value: Any) -> list[Any]:
        if self.size == 1:
            return [value]
        values: list[Any] = [None] * self.size
        dist.all_gather_object(values, value, group=self.setup_group)
        return values

    def agree(self, decision: bool) -> bool:
        """Rank 0 decides for every rank."""
        if self.size == 1:
            return decision
        flag = torch.tensor(int(decision), device=self.device)
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.size > 1:
            dist.all_reduce(tensor)
        return tensor

    def sum_all(self, tensors: list[torch.Tensor]) -> None:
        if self.size > 1:
            self._in_buckets(tensors, dist.all_reduce)

    def broadcast_all(self, tensors: list[torch.Tensor]) -> None:
        if self.size > 1:
            self._in_buckets(tensors, lambda flat: dist.broadcast(flat, src=0))

    @torch.no_grad()
    def _in_buckets(self, tensors: list[torch.Tensor], collective) -> None:
        for bucket in buckets(tensors):
            flat = torch.cat([tensor.reshape(-1) for tensor in bucket])
            collective(flat)
            for tensor, part in zip(
                bucket, flat.split([t.numel() for t in bucket]), strict=True
            ):
                tensor.copy_(part.view_as(tensor))

    @torch.no_grad()
    def check_equal(self, tensors: dict[str, torch.Tensor], step: int) -> None:
        """Stop every rank if the same-named fp32 tensors differ in any rank."""
        if self.size == 1:
            return
        checksums = []
        for tensor in tensors.values():
            bits = tensor.detach().reshape(-1).view(torch.int32).long()
            position = torch.arange(1, len(bits) + 1, device=bits.device)
            checksums.append((bits * position).sum())
        low = torch.stack(checksums).to(self.device)
        high = low.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        differing = [
            name
            for name, same in zip(tensors, (low == high).tolist(), strict=True)
            if not same
        ]
        if differing:
            raise RuntimeError(
                f"{len(differing)} tensors differ between ranks at step {step}, "
                f"first {differing[0]}"
            )
