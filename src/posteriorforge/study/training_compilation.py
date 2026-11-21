"""Opt-in in-place compilation of the pinned TabPFN training MLPs."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from importlib.metadata import version
import os
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from .model import BASE_CHECKPOINT_SHA256, ModelQualificationError, TabPFNBackbone

MLP_COMPILE_POLICY = "tabpfn_mlp_torch_compile.v1"
_COMPILED_CALL = "_compiled_call_impl"
_ABSENT = object()


def _qualified_mlps(model: TabPFNBackbone) -> tuple[tuple[str, nn.Module], ...]:
    from tabpfn.model.layer import PerFeatureEncoderLayer
    from tabpfn.model.mlp import MLP
    from tabpfn.model.transformer import LayerStack, PerFeatureTransformer

    if type(model) is not TabPFNBackbone or model.checkpoint_sha256 != BASE_CHECKPOINT_SHA256:
        raise ModelQualificationError("MLP compilation requires the pinned TabPFN backbone")
    base = model.backbone
    if type(base) is not PerFeatureTransformer:
        from peft import PeftModel

        if type(base) is not PeftModel or version("peft") != "0.21.3":
            raise ModelQualificationError("MLP compilation requires the pinned PEFT wrapper")
        base = base.get_base_model()
    if type(base) is not PerFeatureTransformer:
        raise ModelQualificationError("unsupported compiled TabPFN transformer")
    stack = base.transformer_encoder
    if type(stack) is not LayerStack or len(stack.layers) != 12:
        raise ModelQualificationError("unexpected TabPFN encoder stack for MLP compilation")
    if any(type(layer) is not PerFeatureEncoderLayer or type(layer.mlp) is not MLP
           or layer.second_mlp is not None for layer in stack.layers):
        raise ModelQualificationError("unexpected TabPFN MLP layout")
    targets = tuple((name, module) for name, module in model.named_modules() if type(module) is MLP)
    if len(targets) != len(stack.layers) or (
        {id(module) for _, module in targets} != {id(layer.mlp) for layer in stack.layers}
    ):
        raise ModelQualificationError("MLP compilation targets differ from the pinned encoder stack")
    return targets


def _parameter_layout(model: nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple((name, id(parameter)) for name, parameter in model.named_parameters())


@contextmanager
def compiled_training_mlps(
    model: nn.Module, *, enabled: bool = False, backend: str = "inductor",
) -> Iterator[dict]:
    """Compile the MLP calls for an explicit qualification scope.

    Enclose forward and backward to keep the same call path for any layer
    checkpoint recomputation. The scope can span multiple updates. Installation
    is lazy and the first MLP call may compile. Keep setup/startup timing with
    the worker's qualification evidence.

    Inductor is restricted to CUDA. The eager backend exercises full graph
    capture on CPU for contract tests. Full-graph compilation errors propagate.
    Checkpoint flags, stock forward methods, and registered tensors are kept.
    """
    if type(enabled) is not bool:
        raise TypeError("MLP compilation needs an explicit boolean")
    if not enabled:
        yield {"enabled": False, "policy": "eager", "module_names": []}
        return
    if backend not in ("inductor", "eager"):
        raise ValueError("MLP compilation accepts Inductor or the CPU eager test backend")
    if version("torch").split("+", 1)[0] != "2.6.0" or version("tabpfn") != "2.0.9":
        raise ModelQualificationError("MLP compilation requires Torch 2.6.0 and TabPFN 2.0.9")
    from torch._dynamo import config as dynamo_config
    from torch._inductor.compiler_bisector import CompilerBisector

    if dynamo_config.disable or dynamo_config.suppress_errors or os.environ.get("TORCHDYNAMO_DISABLE") == "1":
        raise ModelQualificationError("MLP compilation requires enabled Dynamo and error propagation")
    if CompilerBisector.get_backend() is not None:
        raise ModelQualificationError("MLP compilation cannot replace the selected backend through compiler bisection")
    targets = _qualified_mlps(model)
    if any(getattr(module, _COMPILED_CALL, None) is not None for module in model.modules()):
        raise ModelQualificationError("MLP compilation cannot nest or replace compiled module calls")
    if any("forward" in child.__dict__ for _, module in targets for child in module.modules()):
        raise ModelQualificationError("MLP compilation requires stock MLP forward methods")
    parameters = tuple(model.parameters())
    expected_device = "cuda" if backend == "inductor" else "cpu"
    if (not parameters or len({parameter.device for parameter in parameters}) != 1
            or any(parameter.device.type != expected_device or parameter.dtype != torch.float32
                   for parameter in parameters)):
        raise ModelQualificationError(f"MLP compilation requires FP32 model parameters on one {expected_device} device")
    if backend == "inductor":
        from torch._inductor import config as inductor_config

        if (torch.get_float32_matmul_precision() != "highest" or torch.is_autocast_enabled()
                or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ModelQualificationError("Inductor MLP qualification requires strict FP32 with AMP and TF32 disabled")
        if inductor_config.triton.cudagraphs:
            raise ModelQualificationError("Inductor MLP qualification requires CUDA graphs disabled")

    options = {"backend": backend, "mode": "default", "fullgraph": True, "dynamic": False}
    meta = {
        "enabled": True, "policy": MLP_COMPILE_POLICY, **options,
        "module_names": [name for name, _ in targets],
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_graphs": False,
    }
    layout = _parameter_layout(model)
    state_keys = tuple(model.state_dict())
    installed = []
    try:
        for _, module in targets:
            installed.append((module, module.__dict__.get(_COMPILED_CALL, _ABSENT)))
            module.compile(**options)
            if not callable(module.__dict__.get(_COMPILED_CALL)):
                raise ModelQualificationError("PyTorch did not install an in-place compiled MLP call")
        if _parameter_layout(model) != layout or tuple(model.state_dict()) != state_keys:
            raise ModelQualificationError("MLP compilation changed the model parameter or checkpoint layout")
        yield meta
    finally:
        for module, previous in reversed(installed):
            if previous is _ABSENT:
                module.__dict__.pop(_COMPILED_CALL, None)
            else:
                setattr(module, _COMPILED_CALL, previous)
