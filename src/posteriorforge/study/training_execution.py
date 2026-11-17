"""Opt-in differentiable execution reuse for bounded training qualification.

No inference merge, detached tensor, persistent neural cache or scientific loss
change is introduced here. All posterior evaluations retain 50,000 full vectors.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial, wraps
from typing import Iterator

import torch
from torch import nn
from torch.nn.utils import parametrize

from .model import ModelQualificationError

CACHE_POLICY = "per_layer_forward_and_checkpoint_recompute.v1"
_EXECUTION_ACTIVE = False


def _require_empty_parametrization_cache() -> None:
    # Pinned PyTorch has no public query for its process-global cache counter.
    active = getattr(parametrize, "_cache_enabled", None)
    if type(active) is not int or active < 0:
        raise ModelQualificationError("cannot establish an isolated PyTorch parametrization-cache scope")
    if active:
        raise ModelQualificationError("an already-active parametrization cache could outlive one layer invocation")


def _cache_targets(model: nn.Module) -> tuple[nn.Module, ...]:
    """Validate the pinned checkpoint boundary before installing any wrapper."""
    modules = tuple(model.modules())
    stacks = tuple(module for module in modules if hasattr(module, "recompute_each_layer"))
    for module in modules:
        if isinstance(module.forward, partial) or "forward" in module.__dict__:
            raise ModelQualificationError("preexisting instance forward wrappers or nested checkpointing are unsupported")
        if getattr(module, "recompute_attn", False) or getattr(module, "recompute", False):
            raise ModelQualificationError("nested attention/MLP checkpointing is unsupported by the layer cache")
    if not stacks:
        # Small ordinary modules may be checked without a TabPFN LayerStack.
        # Their root forward receives a fresh cache on every invocation.
        if any(module.__class__.__name__ in ("LayerStack", "PerFeatureEncoderLayer") for module in modules):
            raise ModelQualificationError("unknown TabPFN checkpoint layer layout")
        return (model,)

    from importlib.metadata import version
    from tabpfn.model.layer import PerFeatureEncoderLayer
    from tabpfn.model.transformer import LayerStack
    if version("tabpfn") != "2.0.9":
        raise ModelQualificationError("the layer cache requires the pinned TabPFN 2.0.9 layout")
    targets = []
    for stack in stacks:
        if type(stack) is not LayerStack or type(stack.recompute_each_layer) is not bool or type(stack.layers) is not nn.ModuleList or len(stack.layers) != stack.num_layers:
            raise ModelQualificationError("unknown TabPFN checkpoint layer layout")
        for layer in stack.layers:
            if type(layer) is not PerFeatureEncoderLayer or type(layer.recompute_attn) is not bool or layer.recompute_attn:
                raise ModelQualificationError("unknown or nested TabPFN checkpoint layer layout")
            if not any(layer is prior for prior in targets):
                targets.append(layer)
    if not targets or {id(module) for module in modules if isinstance(module, PerFeatureEncoderLayer)} != {id(layer) for layer in targets}:
        raise ModelQualificationError("incomplete TabPFN checkpoint layer layout")
    return tuple(targets)


def _cached_forward(original):
    @wraps(original)
    def forward(*args, **kwargs):
        _require_empty_parametrization_cache()
        with parametrize.cached():
            return original(*args, **kwargs)
    return forward


@contextmanager
def conditional_forward_cache(model: nn.Module, *, enabled: bool = False) -> Iterator[None]:
    """Scope temporary layer wrappers through one conditional's backward.

    The enabled context MUST include its density forward, probability scoring
    and backward, and end before gradient guards, the next conditional or the
    optimizer step. Each pinned TabPFN encoder layer opens a fresh public
    parametrization cache for just that invocation. Its cached weights retain
    autograd and are discarded on layer return. The same instance wrapper is
    invoked by stock non-reentrant checkpoint recomputation during backward,
    preserving the original/recomputed tensor-save sequence. Upstream checkpoint
    calls, flags, RNG behavior and early stopping are unchanged.

    Only the temporary wrappers live through backward: cached tensors never
    survive a layer return. Original forward attributes are restored in finally
    on success or failure, with no persistent hooks or checkpoint state. Ordinary
    noncheckpointed fixture modules use the same per-invocation root wrapper.

    PyTorch's cache and this single-worker scope are process-global. Existing
    helper/cache scopes, instance forward overrides, nested attention/MLP
    checkpoints and unknown TabPFN layouts fail closed before modification.
    Disabled execution is the existing path without extra model guards.
    This optimization does not qualify a production or scientific campaign.
    """
    global _EXECUTION_ACTIVE
    if type(enabled) is not bool:
        raise TypeError("enabled must be an explicit boolean")
    if not enabled:
        yield
        return
    if not isinstance(model, nn.Module):
        raise TypeError("a trainable torch module is required")
    if torch.is_inference_mode_enabled() or not torch.is_grad_enabled():
        raise ModelQualificationError("conditional training cache requires gradient-enabled execution")
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise ModelQualificationError("conditional training cache requires trainable parameters")
    if _EXECUTION_ACTIVE:
        raise ModelQualificationError("an already-active training cache scope could outlive one conditional")
    _require_empty_parametrization_cache()
    targets = _cache_targets(model)
    installed = []
    _EXECUTION_ACTIVE = True
    try:
        for target in targets:
            target.forward = _cached_forward(target.forward)
            installed.append(target)
        yield
    finally:
        for target in reversed(installed):
            delattr(target, "forward")
        _EXECUTION_ACTIVE = False
