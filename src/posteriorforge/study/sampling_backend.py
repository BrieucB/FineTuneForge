"""Diagnostic CPU/CUDA bridge to the stock TabPFN preprocessing and current weights."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from .config import StudyConfig
from .model import (BASE_CHECKPOINT_SHA256, CANONICAL_FULL_SUPPORT, FEATURE_EMBEDDING_POLICY, ModelQualificationError,
                    TabPFNBackbone, prepare_fitted_regressor_members, weights_fingerprint, resolve_inference_device)


class TabPFNConditionalBackend:
    """Fit stock preprocessing per stage; evaluate only the supplied current backbone."""
    sampling_law = StudyConfig().sampling_law
    source_policy = StudyConfig().source_grid_policy
    backend_kind = 'stock_tabpfn_cpu'

    def __init__(self, checkpoint, network, *, expected_model_sha256, device='cpu'):
        from importlib.metadata import version
        path = Path(checkpoint)
        if version('tabpfn') != '2.0.9' or not path.is_file() or sha256(path.read_bytes()).hexdigest() != BASE_CHECKPOINT_SHA256:
            raise ModelQualificationError('pinned cached TabPFN 2.0.9 checkpoint required')
        if not isinstance(network, TabPFNBackbone) or network.checkpoint_sha256 != BASE_CHECKPOINT_SHA256:
            raise ModelQualificationError('current backbone must identify the pinned checkpoint')
        target = resolve_inference_device(device)
        if (any(p.device != target or (p.is_floating_point() and p.dtype != torch.float32) for p in network.parameters())
                or any(b.device != target for b in network.buffers())
                or any(b.is_floating_point() and b.dtype != torch.float32 for b in network.backbone.buffers())):
            raise ModelQualificationError('FP32 neural weights and buffers must match the requested diagnostic device')
        if weights_fingerprint(network) != expected_model_sha256:
            raise ModelQualificationError('current model weight identity mismatch')
        self.device = target
        self.backend_kind = 'stock_tabpfn_cuda_qualification' if target.type == 'cuda' else 'stock_tabpfn_cpu'
        self.checkpoint = str(path)
        self.network = network
        self.model_sha256 = expected_model_sha256
        self.network.eval()
        self._check_execution()

    def _check_execution(self):
        if (any(p.device != self.device or (p.is_floating_point() and p.dtype != torch.float32) for p in self.network.parameters())
                or any(b.device != self.device for b in self.network.buffers())
                or any(b.is_floating_point() and b.dtype != torch.float32 for b in self.network.backbone.buffers())):
            raise ModelQualificationError("model device or FP32 neural precision changed")
        if self.device.type == "cuda" and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32):
            raise ModelQualificationError("CUDA qualification requires matmul and cuDNN TF32 disabled")

    def identity_metadata(self):
        from importlib.metadata import version
        root = Path(__file__).parent
        execution = {'device': str(self.device), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            uuid = getattr(props, 'uuid', None)
            execution.update(device_name=props.name, device_uuid=None if uuid is None else str(uuid),
                             capability=list(torch.cuda.get_device_capability(self.device)),
                             cuda_runtime=torch.version.cuda, matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                             cudnn_tf32=torch.backends.cudnn.allow_tf32)
        return {'type': 'TabPFNConditionalBackend', 'backend_kind': self.backend_kind,
                'sampling_law': self.sampling_law, 'source_policy': self.source_policy,
                'feature_embedding_policy': FEATURE_EMBEDDING_POLICY, 'execution': execution,
                'base_checkpoint_sha256': BASE_CHECKPOINT_SHA256,
                'model_sha256': self.model_sha256, 'tabpfn_version': version('tabpfn'),
                'torch_version': version('torch'), 'numpy_version': version('numpy'),
                'source_sha256': {name: sha256((root / name).read_bytes()).hexdigest()
                                  for name in ('sampling_backend.py', 'model.py')}}

    def verify_identity(self):
        self._check_execution()
        if weights_fingerprint(self.network) != self.model_sha256:
            raise ModelQualificationError('model weights changed during sampling')

    def prepare(self, context_features, context_targets):
        from tabpfn import TabPFNRegressor
        reg = TabPFNRegressor(model_path=self.checkpoint, device='cpu', n_estimators=8,
            random_state=0, n_jobs=1, inference_precision=torch.float32,
            fit_mode='fit_preprocessors', softmax_temperature=.9, average_before_softmax=False)
        reg.fit(np.array(context_features, copy=True), np.array(context_targets, copy=True))
        # Inference uses only the current network, not the stock wrapper's weights.
        reg.executor_.model = None
        reg.model_ = None
        return reg

    def sample(self, prepared, query_features, uniforms):
        ensemble = prepare_fitted_regressor_members(prepared, query_features, device=self.device, source_policy=self.source_policy)
        with torch.no_grad():
            density = ensemble.density(self.network, law=self.sampling_law)
            values = density.icdf(torch.as_tensor(uniforms, dtype=torch.float64, device=self.device))
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
