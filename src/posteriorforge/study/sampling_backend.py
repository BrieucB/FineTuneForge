"""CPU diagnostic bridge to the stock TabPFN preprocessing and current weights."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import numpy as np
import torch
from .config import StudyConfig
from .model import (BASE_CHECKPOINT_SHA256, CANONICAL_FULL_SUPPORT, ModelQualificationError,
                    TabPFNBackbone, prepare_fitted_regressor_members, weights_fingerprint)


class TabPFNConditionalBackend:
    """Fit stock preprocessing per stage; evaluate only the supplied current backbone."""
    sampling_law = StudyConfig().sampling_law
    source_policy = StudyConfig().source_grid_policy
    backend_kind = 'stock_tabpfn_cpu'

    def __init__(self, checkpoint, network, *, expected_model_sha256):
        from importlib.metadata import version
        path = Path(checkpoint)
        if version('tabpfn') != '2.0.9' or not path.is_file() or sha256(path.read_bytes()).hexdigest() != BASE_CHECKPOINT_SHA256:
            raise ModelQualificationError('pinned cached TabPFN 2.0.9 checkpoint required')
        if not isinstance(network, TabPFNBackbone) or network.checkpoint_sha256 != BASE_CHECKPOINT_SHA256:
            raise ModelQualificationError('current backbone must identify the pinned checkpoint')
        if (any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float32) for p in network.parameters())
                or any(b.device.type != 'cpu' for b in network.buffers())):
            raise ModelQualificationError('only FP32 CPU diagnostic backbone is supported')
        if weights_fingerprint(network) != expected_model_sha256:
            raise ModelQualificationError('current model weight identity mismatch')
        self.checkpoint = str(path)
        self.network = network
        self.model_sha256 = expected_model_sha256
        self.network.eval()

    def identity_metadata(self):
        from importlib.metadata import version
        root = Path(__file__).parent
        return {'type': 'TabPFNConditionalBackend', 'backend_kind': self.backend_kind,
                'sampling_law': self.sampling_law, 'source_policy': self.source_policy,
                'base_checkpoint_sha256': BASE_CHECKPOINT_SHA256,
                'model_sha256': self.model_sha256, 'tabpfn_version': version('tabpfn'),
                'torch_version': version('torch'), 'numpy_version': version('numpy'),
                'source_sha256': {name: sha256((root / name).read_bytes()).hexdigest()
                                  for name in ('sampling_backend.py', 'model.py')}}

    def verify_identity(self):
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
        ensemble = prepare_fitted_regressor_members(prepared, query_features, device='cpu', source_policy=self.source_policy)
        with torch.no_grad():
            density = ensemble.density(self.network, law=self.sampling_law)
            values = density.icdf(torch.as_tensor(uniforms, dtype=torch.float64))
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
