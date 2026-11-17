"""Authenticated frozen inference representations; checkpoint artifacts never change.

A merged deployment is constructed from a fresh authenticated load. Reloading the
original checkpoint restores the unmerged representation. FP32 merge equivalence
remains a qualification question; neither mode asserts scientific qualification.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time

from .model_loading import load_inference_model

MODES = ('unmerged', 'merged_lora')
MERGE_POLICY = 'peft_fp32_linear_and_packed_merge.v1'


def _source_files():
    root = Path(__file__).parent
    return {name: sha256((root / name).read_bytes()).hexdigest()
            for name in ('inference_deployment.py', 'model.py', 'model_loading.py', 'training.py')}


def _inventory(network):
    from torch.nn.utils import parametrize
    if hasattr(network.backbone, 'peft_config'):
        from peft.tuners.lora.layer import LoraLayer
    else:
        LoraLayer = ()
    from .model import _PackedProjectionDelta, _canonical_subspace_pre_hook
    modules = list(network.backbone.modules())
    bases = []
    for module in modules:
        for hook in module._forward_pre_hooks.values():
            if hook is _canonical_subspace_pre_hook:
                bases.append(getattr(module, '_ftf_subspace_seed', None))
    return {
        'lora_linear_modules': sum(isinstance(m, LoraLayer) for m in modules),
        'packed_projection_modules': sum(isinstance(m, _PackedProjectionDelta) for m in modules),
        'parametrized_modules': sum(parametrize.is_parametrized(m) for m in modules),
        'feature_basis_hook_seeds': sorted(bases, key=repr),
    }


def _require_frozen_fp32(network):
    import torch
    from .model import ModelQualificationError, TabPFNBackbone
    if not isinstance(network, TabPFNBackbone):
        raise ModelQualificationError('an authenticated TabPFNBackbone is required')
    if any(m.training for m in network.modules()) or any(p.requires_grad for p in network.parameters()):
        raise ModelQualificationError('deployment preparation requires a frozen eval model; training models cannot be merged')
    parameters = list(network.parameters())
    if not parameters:
        raise ModelQualificationError('neural parameters are missing')
    device = parameters[0].device
    if device.type not in ('cpu', 'cuda'):
        raise ModelQualificationError('only CPU/CUDA frozen deployments are supported')
    if any(p.device != device or p.dtype != torch.float32 for p in parameters):
        raise ModelQualificationError('deployment requires one device and FP32 neural weights')
    if any(b.device != device for b in network.buffers()) or network.borders.dtype != torch.float64:
        raise ModelQualificationError('deployment must retain on-device FP64 density borders')
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in network.backbone.buffers()):
        raise ModelQualificationError('backbone buffers must retain FP32 precision')
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise ModelQualificationError('nonfinite neural weights cannot be deployed')
    return device


def _require_single_supported_adapter(network, metadata):
    from peft.tuners.lora.layer import LoraLayer
    from .model import ModelQualificationError
    rank = metadata.get('rank')
    if metadata.get('method') not in ('lora_r4', 'lora_r16') or rank not in (4, 16):
        raise ModelQualificationError('merged deployment requires a rank4 or rank16 LoRA checkpoint')
    if metadata['method'] != f'lora_r{rank}':
        raise ModelQualificationError('adapter rank and checkpoint method disagree')
    wrapper = network.backbone
    configs = getattr(wrapper, 'peft_config', None)
    if not isinstance(configs, dict) or set(configs) != {'default'}:
        raise ModelQualificationError('exactly one default PEFT adapter is supported')
    if list(getattr(wrapper, 'active_adapters', ())) != ['default']:
        raise ModelQualificationError('the single default adapter must be active')
    config = configs['default']
    if (config.r != rank or config.lora_alpha != 2 * rank or config.lora_dropout != 0
            or config.bias != 'none' or getattr(config, 'use_dora', False)
            or getattr(config, 'use_rslora', False) or getattr(config, 'use_qalora', False)
            or getattr(config, 'rank_pattern', {}) or getattr(config, 'alpha_pattern', {})):
        raise ModelQualificationError('adapter configuration is outside the qualified LoRA recipe')
    layers = [m for m in wrapper.modules() if isinstance(m, LoraLayer)]
    if not layers:
        raise ModelQualificationError('no PEFT linear adapters found')
    for layer in layers:
        if (layer.disable_adapters or layer.merged or list(layer.active_adapters) != ['default']
                or set(layer.lora_A) != {'default'} or set(layer.lora_B) != {'default'}
                or layer.r['default'] != rank or float(layer.scaling['default']) != 2.
                or bool(getattr(layer, 'use_dora', {}).get('default', False))):
            raise ModelQualificationError('disabled, premerged or nonstandard adapter layer cannot be composed')


def load_frozen_deployment(base_checkpoint, *, expected_model_sha256,
                           campaign_checkpoint=None, expected_checkpoint_manifest_sha256=None,
                           mode='unmerged', device='cpu'):
    """Fresh-load an original or merged frozen model and return honest identities.

    ``model_sha256`` identifies the actual returned tensor representation. The
    original trained identity remains in ``origin_model_sha256`` and the nested
    ``origin`` receipt. No inverse floating-point unmerge is used: the unchanged
    source checkpoint is the reversible origin. GPU parity/distribution checks
    must qualify merged execution separately before a production comparison.
    """
    if mode not in MODES:
        raise ValueError('deployment mode must be unmerged or merged_lora')
    if mode == 'merged_lora' and campaign_checkpoint is None:
        raise ValueError('merged LoRA deployment requires an authenticated campaign checkpoint')
    start = time.perf_counter()
    network, origin = load_inference_model(base_checkpoint,
        expected_model_sha256=expected_model_sha256, campaign_checkpoint=campaign_checkpoint,
        expected_checkpoint_manifest_sha256=expected_checkpoint_manifest_sha256, device=device)
    target = _require_frozen_fp32(network)
    from .model import weights_fingerprint, merge_lora, ModelQualificationError, FEATURE_EMBEDDING_POLICY
    actual_origin = weights_fingerprint(network)
    if actual_origin != expected_model_sha256 or origin.get('model_sha256') != expected_model_sha256:
        raise ModelQualificationError('fresh deployment origin fingerprint mismatch')
    before = _inventory(network)
    merge_seconds = 0.
    if mode == 'merged_lora':
        _require_single_supported_adapter(network, origin)
        import torch
        if target.type == 'cuda':
            torch.cuda.synchronize(target)
        merge_start = time.perf_counter()
        network.backbone = merge_lora(network.backbone)
        network.requires_grad_(False)
        network.eval()
        if target.type == 'cuda':
            torch.cuda.synchronize(target)
        merge_seconds = time.perf_counter() - merge_start
    _require_frozen_fp32(network)
    after = _inventory(network)
    if before['feature_basis_hook_seeds'] != after['feature_basis_hook_seeds']:
        raise ModelQualificationError('merge changed the canonical seeded feature-basis hook')
    if mode == 'merged_lora' and any(after[k] for k in ('lora_linear_modules', 'packed_projection_modules', 'parametrized_modules')):
        raise ModelQualificationError('merged deployment retains adapter or packed-parametrization execution')
    actual = weights_fingerprint(network)
    if mode == 'unmerged' and actual != actual_origin:
        raise ModelQualificationError('unmerged deployment altered the loaded checkpoint')
    metadata = dict(origin)
    metadata.update(
        schema='finetuneforge.frozen_inference_deployment.v1', origin=dict(origin),
        origin_model_sha256=actual_origin, model_sha256=actual,
        deployment_model_sha256=actual, deployment_mode=mode,
        merge_policy='no_merge' if mode == 'unmerged' else MERGE_POLICY,
        merge_seconds=merge_seconds, deployment_preparation_seconds=time.perf_counter()-start,
        inventory_before=before, inventory_after=after,
        feature_embedding_policy=FEATURE_EMBEDDING_POLICY,
        trainable_parameter_count=sum(p.numel() for p in network.parameters() if p.requires_grad),
        trainable_tensor_count=sum(p.requires_grad for p in network.parameters()),
        source_files=_source_files(),
        scientifically_qualified=False, equivalence_status='unmodified' if mode == 'unmerged' else 'requires_execution_qualification',
        reversible_by='fresh authenticated load of unchanged origin checkpoint')
    return network, metadata
