"""Weight-only quantisation for the 8 GB path.

Two storage formats are implemented, both of which genuinely shrink the weight
footprint (the stored tensor is the quantised one, not a float copy):

* ``int8`` -- symmetric, per output channel. One byte per weight plus one fp32
  scale per row. This is the default for the 8 GB profile.
* ``fp8``  -- ``torch.float8_e4m3fn`` with a per-tensor scale. Half the bytes of
  bf16, and on Hopper/Ada the matmuls can run natively through
  ``torch._scaled_mm`` (see :func:`fused_scaled_mm_available`).

Matmuls dequantise on the fly. That costs a little arithmetic but keeps numerics
identical to what a real int8 kernel would produce, and it means the same code
path is exercised on CPU, MPS and CUDA -- which is the only reason the memory
numbers in ``docs/BENCHMARK.md`` can be reproduced on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

QuantMode = Literal["none", "bf16", "int8", "fp8"]
FP8_MAX = 448.0


@dataclass
class QuantSpec:
    """Storage format plus the measured error of the round-trip."""

    mode: QuantMode = "int8"
    max_error: float = 0.0
    tensor_count: int = 0
    param_count: int = 0

    @property
    def bytes_per_param(self) -> float:
        return {"none": 4.0, "bf16": 2.0, "int8": 1.0, "fp8": 1.0}[self.mode]


def quantize_tensor(w: Tensor, mode: QuantMode) -> Tuple[Tensor, Tensor]:
    """Return (quantised tensor, scale) for a 2D weight."""
    if mode in ("none", "bf16"):
        return w, w.new_tensor(1.0)
    if mode == "int8":
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
        return q, scale
    if mode == "fp8":
        amax = w.abs().amax().clamp(min=1e-8)
        scale = amax / FP8_MAX
        q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        return q, scale
    raise ValueError(f"unknown quant mode {mode!r}")


def dequantize_tensor(q: Tensor, scale: Tensor, out_dtype: torch.dtype = torch.float32) -> Tensor:
    if q.dtype == torch.int8:
        return q.to(out_dtype) * scale
    if q.dtype == torch.float8_e4m3fn:
        return q.to(out_dtype) * scale
    return q.to(out_dtype)


class QuantLinear(nn.Module):
    """``nn.Linear`` with quantised weight storage."""

    def __init__(self, linear: nn.Linear, mode: QuantMode = "int8") -> None:
        super().__init__()
        self.mode = mode
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        q, scale = quantize_tensor(linear.weight.data, mode)
        self.register_buffer("weight", q)
        self.register_buffer("scale", scale)
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.data.clone())
        if mode in ("int8", "fp8"):
            self.register_buffer("err", torch.tensor(float((dequantize_tensor(q, scale) - linear.weight.data).abs().max())))

    def forward(self, x: Tensor) -> Tensor:
        w = dequantize_tensor(self.weight, self.scale, out_dtype=x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, storage={self.mode}"


def fused_scaled_mm_available() -> bool:
    """True when the local CUDA build exposes fp8 matmul kernels."""
    return hasattr(torch, "_scaled_mm") and torch.cuda.is_available()


def quantize_module_(module: nn.Module, mode: QuantMode, skip: Iterable[str] = ()) -> QuantSpec:
    """Replace every ``nn.Linear`` under ``module`` with :class:`QuantLinear` in place."""
    if mode in ("none", "bf16"):
        return QuantSpec(mode=mode)
    skip_set = set(skip)
    total_params, tensors, max_err = 0, 0, 0.0
    for name, child in list(module.named_children()):
        if name in skip_set:
            continue
        if isinstance(child, nn.Linear):
            ql = QuantLinear(child, mode)
            setattr(module, name, ql)
            total_params += child.weight.numel()
            tensors += 1
            max_err = max(max_err, float(getattr(ql, "err", torch.tensor(0.0))))
        else:
            spec = quantize_module_(child, mode, skip)
            total_params += spec.param_count
            tensors += spec.tensor_count
            max_err = max(max_err, spec.max_error)
    return QuantSpec(mode=mode, max_error=max_err, tensor_count=tensors, param_count=total_params)


def storage_bytes(module: nn.Module) -> int:
    """Bytes actually held by parameters and buffers (what has to fit in VRAM)."""
    seen: Dict[int, int] = {}
    for t in list(module.parameters()) + list(module.buffers()):
        seen[id(t)] = t.numel() * t.element_size()
    return sum(seen.values())


def to_dtype_module_(module: nn.Module, dtype: torch.dtype) -> None:
    for p in module.parameters():
        p.data = p.data.to(dtype)
    for name, buf in list(module.named_buffers()):
        if buf.dtype.is_floating_point:
            module.register_buffer(name, buf.to(dtype), persistent=False)


__all__ = [
    "QuantMode",
    "QuantSpec",
    "QuantLinear",
    "quantize_tensor",
    "dequantize_tensor",
    "quantize_module_",
    "storage_bytes",
    "fused_scaled_mm_available",
    "FP8_MAX",
]
