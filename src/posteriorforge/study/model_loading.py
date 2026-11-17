"""Authenticated CPU-only loading of pinned or campaign inference weights.

Imports are intentionally lazy: request inspection does not require Torch.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import time

BASE_SHA256 = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'
LAW = 'canonical_half_normal_full_support.v1'
SOURCE_POLICY = 'cancelled_zero_width'


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _file_hash(path):
    h = sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _deployment(checkpoint, metadata, expected_model_sha256):
    value = metadata.get('deployment')
    fields = {'backend_kind','format','path','campaign_method','training_method','rank',
              'base_checkpoint_sha256','weights_sha256','sampling_law','source_policy','scientifically_qualified'}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError('invalid deployment metadata fields')
    allowed = {'lora_r4': ('lora', 4, 'peft_adapter', 'adapter'),
               'lora_r16': ('lora', 16, 'peft_adapter', 'adapter'),
               'full_finetune': ('full_finetune', None, 'full_tabpfn_safetensors', 'full_model.safetensors')}
    wanted = allowed.get(value['campaign_method'])
    if (wanted is None or (value['training_method'], value['rank'], value['format'], value['path']) != wanted
            or value['backend_kind'] != 'stock_tabpfn_cpu' or value['base_checkpoint_sha256'] != BASE_SHA256
            or value['weights_sha256'] != expected_model_sha256 or metadata.get('weights_sha256') != expected_model_sha256
            or value['sampling_law'] != LAW or value['source_policy'] != SOURCE_POLICY
            or value['scientifically_qualified'] is not False):
        raise ValueError('deployment method, weights, base, or sampling policy mismatch')
    identity = metadata.get('identity')
    if not isinstance(identity, dict) or not isinstance(identity.get('spec'), dict) or identity['spec'].get('method') != value['campaign_method']:
        raise ValueError('campaign and deployment method mismatch')
    root = Path(checkpoint)
    directory = root / 'deployment'
    artifact = directory / value['path']
    # No symlink traversal, even when its ultimate target happens to be inside root.
    if directory.is_symlink() or artifact.is_symlink() or not artifact.resolve().is_relative_to(directory.resolve()):
        raise ValueError('unsafe deployment artifact path')
    if not directory.is_dir() or (value['format'] == 'peft_adapter' and not artifact.is_dir()) or (value['format'] == 'full_tabpfn_safetensors' and not artifact.is_file()):
        raise ValueError('missing deployment artifact')
    return value, artifact


def load_inference_model(base_checkpoint, *, expected_model_sha256, campaign_checkpoint=None,
                         expected_checkpoint_manifest_sha256=None):
    """Load authenticated current weights on CPU; no downloads or optimizer state."""
    start = time.perf_counter()
    if not _sha(expected_model_sha256):
        raise ValueError('explicit model SHA256 required')
    base = Path(base_checkpoint)
    if not base.is_file() or _file_hash(base) != BASE_SHA256:
        raise ValueError('pinned cached base checkpoint required')
    deployment = artifact = None
    if campaign_checkpoint is None:
        if expected_checkpoint_manifest_sha256 is not None:
            raise ValueError('frozen base has no campaign manifest')
        checkpoint_identity = BASE_SHA256
    else:
        if not _sha(expected_checkpoint_manifest_sha256):
            raise ValueError('expected campaign manifest SHA256 required')
        root = Path(campaign_checkpoint)
        manifest = root / 'manifest.json'
        if manifest.is_symlink() or not manifest.is_file() or _file_hash(manifest) != expected_checkpoint_manifest_sha256:
            raise ValueError('campaign manifest checksum mismatch')
        from .campaign import read_campaign_checkpoint
        metadata = read_campaign_checkpoint(root)
        deployment, artifact = _deployment(root, metadata, expected_model_sha256)
        checkpoint_identity = expected_checkpoint_manifest_sha256
    from .model import TabPFNBackbone, weights_fingerprint
    import torch
    network = TabPFNBackbone.from_checkpoint(base)
    if deployment is not None:
        if deployment['format'] == 'peft_adapter':
            from .training import load_adapter
            network.backbone = load_adapter(network.backbone, artifact, expected_checkpoint_sha256=BASE_SHA256, trainable=False)
        else:
            from safetensors.torch import load_file
            network.load_state_dict(load_file(str(artifact), device='cpu'), strict=True)
    network.requires_grad_(False)
    network.eval()
    # The density border vector is intentionally FP64; neural state is FP32.
    if (any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float32) for p in network.parameters())
            or any(b.device.type != 'cpu' for b in network.buffers())
            or any(b.is_floating_point() and b.dtype != torch.float32 for b in network.backbone.buffers())):
        raise ValueError('inference network must use FP32 CPU neural weights and buffers')
    if weights_fingerprint(network) != expected_model_sha256:
        raise ValueError('loaded model weight fingerprint mismatch')
    return network, {'model_sha256': expected_model_sha256,
        'checkpoint_manifest_sha256': checkpoint_identity,
        'checkpoint_identity_format': 'base_checkpoint_file_sha256' if deployment is None else 'campaign_manifest_sha256',
        'base_checkpoint_sha256': BASE_SHA256, 'kind': 'frozen' if deployment is None else 'campaign_checkpoint',
        'method': 'frozen' if deployment is None else deployment['campaign_method'],
        'rank': None if deployment is None else deployment['rank'],
        'training_replica': None if deployment is None else metadata.get('identity', {}).get('spec', {}).get('replica'),
        'training_phase': None if deployment is None else metadata.get('identity', {}).get('spec', {}).get('phase'),
        'training_diagnostic': None if deployment is None else metadata.get('identity', {}).get('spec', {}).get('diagnostic'),
        'training_protocol_sha256': None if deployment is None else metadata.get('identity', {}).get('spec', {}).get('protocol_sha256'),
        'completed_updates': 0 if deployment is None else metadata.get('state', {}).get('update'),
        'loader_source_sha256': _file_hash(__file__),
        'load_seconds': time.perf_counter() - start, 'scientifically_qualified': False}
