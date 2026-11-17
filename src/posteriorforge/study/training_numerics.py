"""Named numerical comparisons for explicit training-engineering qualification.

These diagnostics do not run a model, select tolerances, or establish scientific
qualification. Callers must declare their tolerances before comparing a frozen
reference with an execution-only candidate. Missing/nonfinite gradients fail,
including when both executions omit the same trainable parameter's gradient.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch


def compare_named_tensors(reference: Mapping[str, torch.Tensor | None],
                          candidate: Mapping[str, torch.Tensor | None], *,
                          atol: float, rtol: float) -> dict:
    """Return a JSON-compatible comparison with strict name/shape/dtype checks.

    Copies and FP64 diagnostic arithmetic happen outside any measured training
    region. ``max_relative_error`` uses max(abs(reference), 1e-30) only as a
    reporting denominator; pass/fail uses torch's declared atol + rtol rule.
    Nonfinite inputs fail without masking, replacement, or parameter deletion.
    """
    if any(not math.isfinite(value) or value < 0 for value in (atol, rtol)):
        raise ValueError("finite nonnegative comparison tolerances are required")
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    records = {}
    for name in sorted(set(reference) & set(candidate)):
        left, right = reference[name], candidate[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            records[name] = {"passed": False, "reason": "missing tensor/gradient"}
            continue
        if left.shape != right.shape or left.dtype != right.dtype:
            records[name] = {"passed": False, "reason": "shape/dtype mismatch",
                "reference_shape": list(left.shape), "candidate_shape": list(right.shape),
                "reference_dtype": str(left.dtype), "candidate_dtype": str(right.dtype)}
            continue
        left = left.detach().to(device="cpu")
        right = right.detach().to(device="cpu")
        if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
            records[name] = {"passed": False, "reason": "nonfinite tensor"}
            continue
        exact = torch.equal(left, right)
        left64, right64 = left.to(torch.float64), right.to(torch.float64)
        difference = (left64 - right64).abs()
        maximum = float(difference.max()) if difference.numel() else 0.0
        relative = difference / left64.abs().clamp_min(1e-30)
        maximum_relative = float(relative.max()) if relative.numel() else 0.0
        # Diagnostic overflow must not introduce nonstandard JSON Infinity.
        records[name] = {"passed": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
            "exact": exact, "elements": left.numel(),
            "max_absolute_error": maximum if math.isfinite(maximum) else None,
            "max_relative_error": maximum_relative if math.isfinite(maximum_relative) else None}
    failures = [name for name, record in records.items() if not record["passed"]]
    return {"passed": bool(reference) and not missing and not unexpected and not failures,
        "atol": atol, "rtol": rtol, "tensor_count": len(reference),
        "missing": missing, "unexpected": unexpected, "failed": failures,
        "max_absolute_error": max((r.get("max_absolute_error") or 0.0 for r in records.values()), default=0.0),
        "tensors": records}


def trainable_gradient_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    """Keep every trainable name, retaining absent gradients as explicit failures."""
    return {name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad}
