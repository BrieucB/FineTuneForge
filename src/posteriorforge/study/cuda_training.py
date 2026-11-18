"""Explicit CUDA engineering qualification; no production campaign activation.

All posterior evaluations remain50,000 complete vectors in the separate sampler.
This module performs probability scoring and bounded optimizer qualification only.
Built with PriorLabs-TabPFN.
"""
from __future__ import annotations
from hashlib import sha256
import os
from pathlib import Path
import time
import torch
from .campaign import StockTrainingBackend, source_identity
from .model import CANONICAL_FULL_SUPPORT, ModelQualificationError
from .training import make_optimizer, prepare_teacher_forced_episode, _cuda_execution_metadata
from .updates import stagewise_cuda_qualification_update


def configure_cuda_determinism() -> None:
    """Set strict FP32 engineering policy before model execution, without TF32/AMP."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ModelQualificationError("set CUBLAS_WORKSPACE_CONFIG before starting CUDA qualification")
    if not torch.cuda.is_available():
        raise ModelQualificationError("CUDA qualification requires an available CUDA device")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


class QualificationTrainingBackend(StockTrainingBackend):
    """One independent CUDA worker with full local model and unchanged scientific loss."""
    backend_kind = "stock_tabpfn_cuda_qualification"

    def __init__(self, spec, checkpoint, *, device="cuda:0", allow_cuda_qualification=False,
                 cache_parametrizations=False):
        if allow_cuda_qualification is not True:
            raise ModelQualificationError("explicit CUDA qualification opt-in required")
        if type(cache_parametrizations) is not bool:
            raise TypeError("cache_parametrizations must be an explicit boolean")
        self.cache_parametrizations = cache_parametrizations
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ModelQualificationError("CUDA qualification requires a CUDA device")
        configure_cuda_determinism()
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if not 0 <= index < torch.cuda.device_count():
            raise ModelQualificationError("requested CUDA qualification device is unavailable")
        self.device = torch.device("cuda", index)
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            super().__init__(spec, checkpoint)
        self.model.to(self.device)
        self.optimizer = make_optimizer(self.model, self.training_config)
        self.identity.update(device=str(self.device), execution=_cuda_execution_metadata(self.model),
            source_files=source_identity(), cuda_training_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification_only=True, scientifically_qualified=False,
            parametrization_cache_policy="per_layer_forward_and_checkpoint_recompute.v1" if cache_parametrizations else "stock_uncached")

    def prepare(self, item):
        return prepare_teacher_forced_episode(item.context, item.query,
            regressor_factory=self.regressor_factory, law=CANONICAL_FULL_SUPPORT, device=self.device)

    def update(self, items, state):
        items = tuple(items)
        if len(items) != self.spec.effective_batch:
            raise ModelQualificationError("CUDA qualification update must preserve the declared effective batch")
        if any(len(item.query.theta) != 1 for item in items):
            raise ModelQualificationError("matched training episodes each require one independent query case")
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        prepared = [self.prepare(item) for item in items]
        torch.cuda.synchronize(self.device)
        preprocessing = time.perf_counter() - started
        report = stagewise_cuda_qualification_update(self.model, self.optimizer, prepared, state,
            law=CANONICAL_FULL_SUPPORT, gradient_clip=1.0,
            allow_cuda_qualification=True, expected_device=self.device,
            cache_parametrizations=self.cache_parametrizations)
        torch.cuda.synchronize(self.device)
        return dict(report, preprocessing_seconds=preprocessing,
            preprocessing_includes_host_to_device=True,
            update_seconds=time.perf_counter()-started-preprocessing)


class CudaLearningPilotBackend(QualificationTrainingBackend):
    """Use the qualified eager rank-4 path and retain capacity measurements."""

    def __init__(self, spec, checkpoint, *, attention_max_score_bytes=None):
        from .pilot import CUDA_TILED_ATTENTION_EXECUTION

        if spec.method != "lora_r4" or spec.phase != "pilot" or spec.diagnostic:
            raise ModelQualificationError("CUDA learning pilot requires a non-diagnostic rank-4 pilot")
        if os.environ.get("TABPFN_ALLOW_CPU_LARGE_DATASET") != "1":
            raise ModelQualificationError("the CUDA pilot requires CPU preprocessing for the 1,024-world contexts")
        if torch.cuda.device_count() != 1:
            raise ModelQualificationError("CUDA learning pilot requires exactly one visible GPU")
        if attention_max_score_bytes is not None and (
                type(attention_max_score_bytes) is not int
                or attention_max_score_bytes != CUDA_TILED_ATTENTION_EXECUTION["attention_max_score_bytes"]):
            raise ModelQualificationError("CUDA learning pilot requires the declared attention score budget")
        self.attention_max_score_bytes = attention_max_score_bytes
        super().__init__(spec, checkpoint, device="cuda:0", allow_cuda_qualification=True,
                         cache_parametrizations=True)
        self.identity.update(fusion_enabled=False, cpu_large_dataset_preprocessing=True)
        if attention_max_score_bytes is not None:
            self.identity.update(attention_policy=CUDA_TILED_ATTENTION_EXECUTION["attention_policy"],
                                 attention_max_score_bytes=attention_max_score_bytes,
                                 attention_source_sha256=sha256(Path(__file__).with_name("cuda_attention.py").read_bytes()).hexdigest())

    def _measure(self, operation, *args):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        if self.attention_max_score_bytes is None:
            report = operation(*args)
        else:
            from .cuda_attention import install_training_tiled_attention

            with install_training_tiled_attention(self.model, max_score_bytes=self.attention_max_score_bytes) as attention:
                report = dict(operation(*args), attention=attention.metadata, attention_statistics=attention.statistics)
        torch.cuda.synchronize(self.device)
        return dict(report, peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))

    def update(self, items, state):
        return self._measure(super().update, items, state)

    def quick_validate(self, items):
        return self._measure(super().quick_validate, items)
