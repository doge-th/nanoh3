"""Block-wise CPU/GPU offload -- the mechanism that actually makes 8 GB work.

A 300 M parameter denoiser in int8 is 300 MB, so weights are never the binding
constraint at our sizes; *activations* are, and the second binding constraint is
the VAE decoder, which runs at full pixel resolution. What this module buys is
the ability to keep only a few transformer blocks resident at a time, so peak
VRAM is set by ``resident_blocks`` rather than by model depth.

The wrapper is transparent: it keeps the wrapped block's ``forward`` signature, so
:class:`nanoh3.models.dit.NanoDiT` is unaware of it. Copies to the accelerator are
prefetched one block ahead on a worker thread and overlap with compute.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import Tensor, nn


@dataclass
class OffloadStats:
    moves: int = 0
    bytes_to_device: int = 0
    bytes_to_host: int = 0
    active: bool = False

    def summary(self) -> str:
        return (
            f"offload(active={self.active}, moves={self.moves}, "
            f"h2d={self.bytes_to_device / 2**20:.1f}MiB, d2h={self.bytes_to_host / 2**20:.1f}MiB)"
        )


def _tensors(module: nn.Module) -> Iterable[Tensor]:
    seen = set()
    for t in list(module.parameters()) + list(module.buffers()):
        if id(t) not in seen:
            seen.add(id(t))
            yield t


def module_bytes(module: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in _tensors(module))


class OffloadedBlock(nn.Module):
    """Runs ``block`` with its weights streamed in and out around each call.

    ``cpu_is_noop`` exists so the CPU can be treated as a real target. In
    production it stays ``True`` (copying host memory to host memory is pure
    waste); the test suite sets it ``False`` so the streaming path is actually
    executed on machines that have no accelerator.
    """

    def __init__(
        self,
        block: nn.Module,
        device: torch.device,
        enabled: bool = True,
        executor: Optional[ThreadPoolExecutor] = None,
        stats: Optional[OffloadStats] = None,
        cpu_is_noop: bool = True,
    ) -> None:
        super().__init__()
        self.block = block
        self.device = device
        self.enabled = enabled and not (cpu_is_noop and device.type == "cpu")
        self.executor = executor
        self.stats = stats or OffloadStats(active=self.enabled)
        self.stats.active = self.stats.active or self.enabled
        self._resident = not self.enabled
        if self.enabled:
            self.to_host()
        self._prefetch: Optional[Future] = None

    # -- residency ------------------------------------------------------
    def to_host(self) -> None:
        for t in _tensors(self.block):
            t.data = t.data.to("cpu", non_blocking=False)
        self._resident = False

    def _prefetch_to_device(self) -> None:
        if not self.enabled:
            return
        nbytes = module_bytes(self.block)
        self._prefetch = self.executor.submit(self._copy_to_device) if self.executor else None
        if self._prefetch is None:
            self._copy_to_device()
        self.stats.moves += 1
        self.stats.bytes_to_device += nbytes

    def _copy_to_device(self) -> None:
        for t in _tensors(self.block):
            t.data = t.data.to(self.device, non_blocking=True)

    def _join_prefetch(self) -> None:
        if self._prefetch is not None:
            self._prefetch.result()
            self._prefetch = None

    # -- module protocol ------------------------------------------------
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self.enabled:
            return self.block(*args, **kwargs)
        self._prefetch_to_device()
        self._join_prefetch()
        out = self.block(*args, **kwargs)
        nbytes = module_bytes(self.block)
        for t in _tensors(self.block):
            t.data = t.data.to("cpu", non_blocking=False)
        self.stats.moves += 1
        self.stats.bytes_to_host += nbytes
        return out

    def extra_repr(self) -> str:
        return f"offloaded={self.enabled} bytes={module_bytes(self.block)}"


class SequentialOffload:
    """Context manager that streams a ModuleList of blocks and reports what it cost.

    The blocks are *replaced* in-place by :class:`OffloadedBlock` wrappers, so the
    parent module still owns exactly the same parameter objects (checkpoints and
    optimizers are unaffected) and no submodule cycle is introduced.

    Usage::

        with SequentialOffload(model.blocks, device) as offload:
            out = model(...)          # blocks stream in and out as they are used
        print(offload.stats.summary())
    """

    def __init__(
        self,
        container: nn.ModuleList,
        device: torch.device,
        enabled: bool = True,
        cpu_is_noop: bool = True,
    ) -> None:
        self.container = container
        self.device = device
        self.enabled = enabled and not (cpu_is_noop and device.type == "cpu")
        self.stats = OffloadStats(active=self.enabled)
        self._executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self) -> "SequentialOffload":
        if self.enabled:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nanoh3-offload")
            for i in range(len(self.container)):
                # cpu_is_noop=False: this context manager has already made the
                # decision, so the wrappers must not second-guess it.
                self.container[i] = OffloadedBlock(
                    self.container[i],
                    self.device,
                    True,
                    self._executor,
                    self.stats,
                    cpu_is_noop=False,
                )
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.enabled:
            for i in range(len(self.container)):
                wrapper = self.container[i]
                assert isinstance(wrapper, OffloadedBlock)
                wrapper._copy_to_device()  # leave the block resident for later direct use
                self.container[i] = wrapper.block
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None


def pin_host_memory(module: nn.Module) -> int:
    """Move weights into pinned CPU memory (CUDA only). Returns bytes pinned."""
    if not torch.cuda.is_available():
        return 0
    pinned = 0
    for t in _tensors(module):
        if not t.is_pinned():
            t.data = t.data.pin_memory()
            pinned += t.numel() * t.element_size()
    return pinned


def host_ram_required(module: nn.Module, multiplier: float = 1.0) -> int:
    return int(module_bytes(module) * multiplier)


__all__ = [
    "OffloadStats",
    "OffloadedBlock",
    "SequentialOffload",
    "module_bytes",
    "pin_host_memory",
    "host_ram_required",
]
