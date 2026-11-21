"""Opt-in bounded exact attention for TabPFN 2.0.9 qualification.

Queries and independent batch rows are tiled; each query still sees every key
and one global softmax. This does not qualify GPU numerical parity or memory.
"""
from __future__ import annotations
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
import types
import torch
import torch.nn.functional as F

POLICY = 'tabpfn_fp32_math_sdpa_query_batch_tiled.v1'
TRAINING_POLICY = 'tabpfn_fp32_math_sdpa_training_tiled_recompute.v1'


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _validate_tiles(q, k, v, max_score_bytes, dropout_p, is_causal, attn_mask):
    _positive(max_score_bytes, 'max_score_bytes')
    if is_causal is not False or attn_mask is not None or dropout_p not in (0, 0.0, None):
        raise ValueError('tiled qualification supports only noncausal unmasked zero-dropout attention')
    if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (q,k,v)):
        raise ValueError('four-dimensional q/k/v required')
    if any(x.dtype != torch.float32 or x.device != q.device for x in (q,k,v)):
        raise ValueError('q/k/v must share a device and FP32 dtype')
    b, nq, h, d = q.shape
    if not b or not nq or not h or not d or k.shape[0] != b or v.shape[0] != b or k.shape[1] != v.shape[1] or not k.shape[1] or k.shape[2] != v.shape[2] or not k.shape[2] or h % k.shape[2] or k.shape[3] != d or v.shape[3] < 1:
        raise ValueError('incompatible q/k/v or grouped-query head dimensions')


def _tile_shape(q, k, max_score_bytes, statistics):
    batch_size, query_count, heads, _ = q.shape
    bytes_per_query = heads * k.shape[1] * 4
    if bytes_per_query > max_score_bytes:
        raise ValueError('score budget cannot hold one complete-key query')
    query_tile = min(query_count, max_score_bytes // bytes_per_query)
    batch_tile = min(batch_size, max_score_bytes // (bytes_per_query * query_tile))
    if statistics is not None:
        statistics['max_batch_tile'] = max(statistics['max_batch_tile'], batch_tile)
        statistics['max_query_tile'] = max(statistics['max_query_tile'], query_tile)
        statistics['max_tile_score_bytes'] = max(
            statistics['max_tile_score_bytes'], batch_tile * query_tile * bytes_per_query)
    return batch_tile, query_tile


def tiled_attention(q, k, v, *, max_score_bytes=536870912, scale=None,
                    dropout_p=0.0, is_causal=False, attn_mask=None, _statistics=None):
    """Exact independent query tiling with all keys per SDPA invocation.

    Inputs follow TabPFN's (batch, sequence, heads, head-dimension) layout.
    Masks/causality/dropout are absent from the pinned TabPFN attention path;
    reject them rather than silently alter their semantics. GQA is implemented
    with the same ordered repeat of K/V heads as upstream's broadcast helper.
    """
    _validate_tiles(q, k, v, max_score_bytes, dropout_p, is_causal, attn_mask)
    if q.device.type == 'cuda' and (torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or not torch.backends.cuda.math_sdp_enabled() or torch.backends.cuda.flash_sdp_enabled() or torch.backends.cuda.mem_efficient_sdp_enabled() or (hasattr(torch.backends.cuda, 'cudnn_sdp_enabled') and torch.backends.cuda.cudnn_sdp_enabled())):
        raise ValueError('CUDA qualification requires FP32 math SDPA with TF32 and fused SDPA disabled')
    batch_tile, query_tile = _tile_shape(q, k, max_score_bytes, _statistics)
    b, nq, h, _ = q.shape
    result = torch.empty((b,nq,h,v.shape[3]),dtype=q.dtype,device=q.device)
    for b0 in range(0,b,batch_tile):
        kk=k[b0:b0+batch_tile].transpose(1,2)
        vv=v[b0:b0+batch_tile].transpose(1,2)
        if h != k.shape[2]:
            repeat=h//k.shape[2]
            kk=kk.repeat_interleave(repeat,dim=1); vv=vv.repeat_interleave(repeat,dim=1)
        for i in range(0,nq,query_tile):
            qq=q[b0:b0+batch_tile,i:i+query_tile].transpose(1,2)
            result[b0:b0+batch_tile,i:i+query_tile] = F.scaled_dot_product_attention(
                qq,kk,vv,dropout_p=0.0,is_causal=False,scale=scale).transpose(1,2)
    return result


@contextmanager
def _fp32_math_attention(device):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if device.type == 'cuda' and (
        torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
    ):
        raise ValueError('CUDA qualification requires TF32 disabled')
    with sdpa_kernel(SDPBackend.MATH), torch.autocast(device.type, enabled=False):
        yield


def tiled_training_attention(q, k, v, *, max_score_bytes=536870912, scale=None,
                             dropout_p=0.0, is_causal=False, attn_mask=None, _statistics=None):
    """Run FP32 math SDPA over TabPFN batch and query tiles, retaining all keys.

    Checkpoint only SDPA. The checkpoint inputs are views of projected Q/K/V,
    so tile recomputation has no dependency on the layer's parameter cache.
    The byte limit applies to a single score tensor, not total model memory.
    """
    from torch.utils.checkpoint import checkpoint

    _validate_tiles(q, k, v, max_score_bytes, dropout_p, is_causal, attn_mask)
    batch_tile, query_tile = _tile_shape(q, k, max_score_bytes, _statistics)
    checkpoint_tiles = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))
    share_kv = q.shape[2] // k.shape[2]

    def attend(queries, keys, values):
        with _fp32_math_attention(queries.device):
            if _statistics is not None:
                _statistics['sdpa_tile_calls'] += 1
            return F.scaled_dot_product_attention(
                queries, keys, values, dropout_p=0.0, is_causal=False,
                scale=scale, enable_gqa=share_kv > 1)

    def checkpoint_tile(queries, keys, values):
        called = False

        def forward_and_recompute(*inputs):
            nonlocal called
            if _statistics is not None and called:
                _statistics['recomputed_tiles'] += 1
            called = True
            return attend(*inputs)

        if _statistics is not None:
            _statistics['checkpointed_tiles'] += 1
        return checkpoint(
            forward_and_recompute, queries, keys, values,
            use_reentrant=False, preserve_rng_state=False)

    batches = []
    for query_batch, key_batch, value_batch in zip(
        q.split(batch_tile), k.split(batch_tile), v.split(batch_tile), strict=True
    ):
        keys = key_batch.transpose(1, 2)
        values = value_batch.transpose(1, 2)
        query_results = []
        for query_chunk in query_batch.split(query_tile, dim=1):
            queries = query_chunk.transpose(1, 2)
            if checkpoint_tiles:
                out = checkpoint_tile(queries, keys, values)
            else:
                out = attend(queries, keys, values)
            query_results.append(out.transpose(1, 2))
        batches.append(torch.cat(query_results, dim=1) if len(query_results) > 1 else query_results[0])
    return torch.cat(batches, dim=0) if len(batches) > 1 else batches[0]


def _unpack(q,k,v,kv,qkv):
    if qkv is not None:
        if any(x is not None for x in (q,k,v,kv)) or qkv.shape[-3] != 3:
            raise ValueError('unexpected packed qkv')
        return qkv.unbind(-3)
    if kv is not None:
        if q is None or k is not None or v is not None or kv.shape[-3] != 2:
            raise ValueError('unexpected packed kv')
        k,v=kv.unbind(-3)
    if q is None or k is None or v is None:
        raise ValueError('complete q/k/v required')
    return q,k,v


class TiledAttentionHandle:
    """Own scoped per-instance patches; close restores exact original methods."""
    def __init__(self, modules, max_score_bytes, counters):
        self.modules=modules; self.max_score_bytes=max_score_bytes; self.counters=counters
        self.closed=False

    @property
    def metadata(self):
        return {'policy':POLICY,'max_score_bytes':self.max_score_bytes,
                'attention_instances':len(self.modules),'source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
                'scientifically_qualified':False,'training_path':'stock_unchanged'}

    @property
    def statistics(self):
        return dict(self.counters)

    def close(self):
        if not self.closed:
            for module, original, replacement in self.modules:
                if getattr(module._compute, '__func__', None) is not replacement:
                    raise RuntimeError('attention patch ownership changed before restore')
            for module, original, replacement in self.modules:
                delattr(module, '_compute')
            self.closed=True

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def install_tiled_attention(network, *, max_score_bytes=536870912):
    """Install scoped inference-only patches on this network's attention instances.

    Gradient-enabled forwards delegate to stock unchanged. No class/global
    monkeypatch occurs. Pinned support decorator preserves stock residual/shards.
    """
    _positive(max_score_bytes,'max_score_bytes')
    from importlib.metadata import version
    from tabpfn.model.multi_head_attention import MultiHeadAttention
    from tabpfn.model.memory import support_save_peak_mem_factor
    if version('tabpfn') != '2.0.9': raise ValueError('pinned TabPFN 2.0.9 required')
    targets=[m for m in network.modules() if isinstance(m,MultiHeadAttention)]
    if not targets: raise ValueError('network contains no stock TabPFN attention')
    if any('_compute' in m.__dict__ for m in targets):
        raise ValueError('attention instance already has a custom _compute method')
    patched=[]
    counters={'stock_fast_path': 0, 'tiled_inference': 0, 'stock_gradient_path': 0,
              'max_estimated_score_bytes': 0, 'max_batch_tile': 0, 'max_query_tile': 0, 'max_tile_score_bytes': 0}
    try:
        for module in targets:
            original=module._compute
            def core(self, x, x_kv, k_cache, v_cache, kv_cache, *, cache_kv, use_cached_kv,
                     reuse_first_head_kv, use_second_set_of_queries):
                if cache_kv or use_cached_kv or any(v is not None for v in (k_cache,v_cache,kv_cache)):
                    raise ValueError('tiled inference does not support attention caches')
                q,k,v,kv,qkv=self.compute_qkv(x,x_kv,k_cache,v_cache,kv_cache,cache_kv=cache_kv,
                    use_cached_kv=use_cached_kv,reuse_first_head_kv=reuse_first_head_kv,
                    use_second_set_of_queries=use_second_set_of_queries)
                q,k,v=_unpack(q,k,v,kv,qkv)
                out=tiled_attention(q,k,v,max_score_bytes=max_score_bytes,
                                    scale=self.softmax_scale,dropout_p=self.dropout_p,_statistics=counters)
                return torch.einsum('... h d, h d s -> ... s',out,self._w_out)
            tiled=support_save_peak_mem_factor(core)
            def replacement(self,*args,_original=original,_tiled=tiled,**kwargs):
                if torch.is_grad_enabled():
                    counters['stock_gradient_path'] += 1
                    return _original(*args,**kwargs)
                x, x_kv = args[0], args[1]
                key_count = x.shape[1] if x_kv is None else x_kv.shape[1]
                score_bytes = x.shape[0] * self._nhead * x.shape[1] * key_count * 4
                counters['max_estimated_score_bytes'] = max(counters['max_estimated_score_bytes'], score_bytes)
                if score_bytes <= max_score_bytes:
                    counters['stock_fast_path'] += 1
                    return _original(*args,**kwargs)
                counters['tiled_inference'] += 1
                return _tiled(self,*args,**kwargs)
            module._compute=types.MethodType(replacement,module)
            patched.append((module,original,replacement))
    except BaseException:
        for module,_,_ in patched: delattr(module,'_compute')
        raise
    return TiledAttentionHandle(patched,max_score_bytes,counters)


class TrainingTiledAttentionHandle(TiledAttentionHandle):
    @property
    def metadata(self):
        return {
            **super().metadata,
            'policy': TRAINING_POLICY,
            'training_path': 'fp32_math_sdpa_tiled_recompute',
        }


def install_training_tiled_attention(network, *, max_score_bytes=536870912):
    """Temporarily bound over-cap FP32 attention on a pinned TabPFN network.

    The scope must include backward when using layer checkpoints. In-cap
    invocations call the original implementation. Only gradient-requiring
    tiles run under an inner checkpoint.
    """
    from importlib.metadata import version
    from tabpfn.model.memory import support_save_peak_mem_factor
    from tabpfn.model.multi_head_attention import MultiHeadAttention

    _positive(max_score_bytes, 'max_score_bytes')
    if version('tabpfn') != '2.0.9':
        raise ValueError('pinned TabPFN 2.0.9 required')
    targets = [module for module in network.modules() if isinstance(module, MultiHeadAttention)]
    if not targets:
        raise ValueError('network contains no stock TabPFN attention')
    if any('_compute' in module.__dict__ for module in targets):
        raise ValueError('attention instance already has a custom _compute method')

    counters = dict.fromkeys((
        'stock_fast_path', 'tiled_training', 'tiled_inference',
        'max_estimated_score_bytes', 'max_batch_tile', 'max_query_tile', 'max_tile_score_bytes',
        'sdpa_tile_calls', 'checkpointed_tiles', 'recomputed_tiles',
    ), 0)

    @support_save_peak_mem_factor
    def tiled(self, x, x_kv, k_cache, v_cache, kv_cache, *, cache_kv, use_cached_kv,
              reuse_first_head_kv, use_second_set_of_queries):
        if cache_kv or use_cached_kv or any(cache is not None for cache in (k_cache, v_cache, kv_cache)):
            raise ValueError('tiled training attention does not support attention caches')
        packed = self.compute_qkv(
            x, x_kv, k_cache, v_cache, kv_cache,
            cache_kv=cache_kv, use_cached_kv=use_cached_kv,
            reuse_first_head_kv=reuse_first_head_kv,
            use_second_set_of_queries=use_second_set_of_queries)
        q, k, v = _unpack(*packed)
        out = tiled_training_attention(
            q, k, v, max_score_bytes=max_score_bytes, scale=self.softmax_scale,
            dropout_p=self.dropout_p, _statistics=counters)
        return torch.einsum('... h d, h d s -> ... s', out, self._w_out)

    patched = []
    try:
        for module in targets:
            original = module._compute

            def replacement(self, x, x_kv, k_cache, v_cache, kv_cache, *,
                            _original=original, **kwargs):
                if x.dtype != torch.float32 or (x_kv is not None and x_kv.dtype != torch.float32):
                    raise ValueError('tiled training attention requires FP32 inputs')
                key_source = x if x_kv is None else x_kv
                if kwargs['use_cached_kv']:
                    key_source = kv_cache if kv_cache is not None else k_cache
                    if key_source is None:
                        raise ValueError('cached attention keys are unavailable')
                key_count = key_source.shape[1]
                score_bytes = x.shape[0] * self._nhead * x.shape[1] * key_count * 4
                counters['max_estimated_score_bytes'] = max(
                    counters['max_estimated_score_bytes'], score_bytes)
                if score_bytes <= max_score_bytes:
                    counters['stock_fast_path'] += 1
                    with _fp32_math_attention(x.device):
                        return _original(x, x_kv, k_cache, v_cache, kv_cache, **kwargs)

                counter = 'tiled_training' if torch.is_grad_enabled() else 'tiled_inference'
                counters[counter] += 1
                return tiled(self, x, x_kv, k_cache, v_cache, kv_cache, **kwargs)

            module._compute = types.MethodType(replacement, module)
            patched.append((module, original, replacement))
    except BaseException:
        for module, _, _ in patched:
            delattr(module, '_compute')
        raise
    return TrainingTiledAttentionHandle(patched, max_score_bytes, counters)
