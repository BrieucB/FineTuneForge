"""CPU parity and recomputation checks for scoped training attention."""
from importlib.metadata import version
import os

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from posteriorforge.study.cuda_attention import (
    install_training_tiled_attention,
    tiled_training_attention,
)
from posteriorforge.study.training_numerics import (
    compare_named_tensors,
    trainable_gradient_snapshot,
)

ATOL = 2e-7
RTOL = 2e-5
# Match the existing real-backbone training-engineering forward criteria.
LOGIT_ATOL = 1e-4
LOGIT_RTOL = 1e-3
LOSS_ATOL = 1e-3


@pytest.mark.parametrize(
    "kv_heads,rows_per_tile,q_per_tile",
    [(4, 1, 2), (2, 1, 2), (1, 2, 5), (4, 3, 5)],
)
def test_all_keys_and_gqa_gradients_are_recomputed_within_score_cap(
    monkeypatch, kv_heads, rows_per_tile, q_per_tile
):
    batch, queries, heads, keys = 3, 5, 4, 11
    budget = 4 * heads * keys * rows_per_tile * q_per_tile
    generator = torch.Generator().manual_seed(71)
    query = torch.randn(batch, queries, heads, 7, generator=generator, requires_grad=True)
    key = torch.randn(batch, keys, kv_heads, 7, generator=generator, requires_grad=True)
    value = torch.randn(batch, keys, kv_heads, 3, generator=generator, requires_grad=True)
    gradient = torch.randn(batch, queries, heads, 3, generator=generator)
    repeat = heads // kv_heads

    original = F.scaled_dot_product_attention
    with sdpa_kernel(SDPBackend.MATH):
        expected = original(
            query.transpose(1, 2),
            key.repeat_interleave(repeat, dim=2).transpose(1, 2),
            value.repeat_interleave(repeat, dim=2).transpose(1, 2),
            scale=.27,
        ).transpose(1, 2)
    expected_grads = torch.autograd.grad(expected, (query, key, value), gradient)

    tiles = []

    def observe_tile(q, k, v, *args, **kwargs):
        tile_rows, tile_heads, tile_queries, _ = q.shape
        assert k.shape == (tile_rows, kv_heads, keys, 7)
        assert v.shape == (tile_rows, kv_heads, keys, 3)
        assert tile_heads == heads
        assert 4 * tile_rows * tile_heads * tile_queries * keys <= budget
        tiles.append((tile_rows, tile_queries))
        return original(q, k, v, *args, **kwargs)

    saved_shapes = []

    def observe_saved(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(F, "scaled_dot_product_attention", observe_tile)
    with torch.autograd.graph.saved_tensors_hooks(observe_saved, lambda tensor: tensor):
        actual = tiled_training_attention(query, key, value, max_score_bytes=budget, scale=.27)
    forward_tiles = tiles.copy()
    assert max(rows for rows, _ in forward_tiles) == rows_per_tile
    assert max(count for _, count in forward_tiles) == q_per_tile
    assert sum(rows * count for rows, count in forward_tiles) == batch * queries
    if len(forward_tiles) > 1:
        assert not any(shape[-1:] == (keys,) for shape in saved_shapes)
    actual_grads = torch.autograd.grad(actual, (query, key, value), gradient)
    if len(forward_tiles) > 1:
        assert sorted(tiles[len(forward_tiles):]) == sorted(forward_tiles)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=ATOL, rtol=RTOL)


def test_patched_attention_restores_original_after_exception():
    pytest.importorskip("tabpfn")
    from tabpfn.model.multi_head_attention import MultiHeadAttention

    layer = MultiHeadAttention(
        input_size=16, output_size=16, d_k=4, d_v=4, nhead=4,
        device=torch.device("cpu"), dtype=torch.float32, share_kv_across_n_heads=2,
    )
    original = layer._compute
    generator = torch.Generator().manual_seed(83)
    queries = torch.randn(3, 5, 16, generator=generator)
    context = torch.randn(3, 11, 16, generator=generator)
    with torch.no_grad():
        expected = layer(queries, context)
        full_score_bytes = 4 * 3 * 4 * 5 * 11
        with install_training_tiled_attention(layer, max_score_bytes=full_score_bytes) as handle:
            torch.testing.assert_close(layer(queries, context), expected, atol=0, rtol=0)
            assert handle.statistics["stock_fast_path"] == 1
            assert handle.statistics["sdpa_tile_calls"] == 0
    with pytest.raises(RuntimeError, match="scope interruption"):
        with install_training_tiled_attention(layer, max_score_bytes=1024) as handle:
            assert layer._compute != original
            with pytest.raises(ValueError, match="custom"):
                install_training_tiled_attention(layer)
            with torch.no_grad():
                torch.testing.assert_close(layer(queries, context), expected, atol=ATOL, rtol=RTOL)
            assert handle.statistics["tiled_inference"] == 1
            assert handle.statistics["sdpa_tile_calls"] > 1
            assert handle.statistics["checkpointed_tiles"] == 0
            raise RuntimeError("scope interruption")
    assert layer._compute == original
    assert "_compute" not in layer.__dict__


@pytest.mark.parametrize("rank", [4, 16])
def test_real_tabpfn_checkpoint_cache_and_tiling_preserve_lora_gradients(rank, monkeypatch):
    checkpoint = os.environ.get("FTF_CPU_CHECKPOINT")
    if not checkpoint:
        pytest.skip("real cached checkpoint opt-in; no downloads")
    assert version("tabpfn") == "2.0.9"
    from tabpfn.model.multi_head_attention import MultiHeadAttention
    from tabpfn.model.transformer import LayerStack
    from posteriorforge.study.model import (
        PreparedMember,
        TabPFNBackbone,
        attach_lora,
        canonical_full_support_mixture,
    )
    from posteriorforge.study.training_execution import conditional_forward_cache

    network = TabPFNBackbone.from_checkpoint(checkpoint)
    generator = torch.Generator().manual_seed(2026111705)
    context = torch.randn(16, 3, generator=generator)
    targets = torch.randn(16, generator=generator)
    queries = torch.randn(3, 3, generator=generator)
    ensemble = []
    for member in range(8):
        order = torch.roll(torch.arange(3), member % 3)
        ensemble.append(PreparedMember(context[:, order], targets, queries[:, order]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13)
        network.backbone = attach_lora(network.backbone, rank=rank)
    # Use nonzero B so the comparison also exercises A derivatives.
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            if "lora_B" in name:
                parameter.normal_(mean=0., std=.001, generator=generator)
    stacks = [module for module in network.modules() if isinstance(module, LayerStack)]
    assert stacks and all(stack.recompute_each_layer for stack in stacks)
    complete_group_score_bytes = []

    def untiled_math_attention(q, k, v, kv, qkv, dropout_p, softmax_scale):
        # TabPFN's CPU fallback uses einsum/softmax. Use the stock CUDA math
        # SDPA semantics on CPU to isolate tiling from the kernel change.
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None:
            k, v = kv.unbind(dim=-3)
        assert dropout_p in (None, 0)
        complete_group_score_bytes.append(q.element_size() * q.shape[1] * q.shape[2] * k.shape[1])
        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                scale=softmax_scale, enable_gqa=q.shape[2] != k.shape[2],
            ).transpose(1, 2)

    monkeypatch.setattr(
        MultiHeadAttention, "compute_attention_heads", staticmethod(untiled_math_attention)
    )

    def run(attention=None):
        network.zero_grad(set_to_none=True)
        with conditional_forward_cache(network, enabled=True):
            outputs = network(ensemble)
            density = canonical_full_support_mixture(outputs, [network.borders] * 8, network.borders)
            losses = -density.log_prob(
                torch.tensor([-4., .25, 4.], dtype=torch.float64), log_abs_det_decode=.17
            )
            # Stagewise training weights a single-conditional role group by
            # 1/(total query cases * 3 equiprobable role groups).
            contribution = losses.sum() / (len(queries) * 3)
            if attention is not None:
                forward_calls = attention.statistics["tiled_training"]
                assert forward_calls > 0
            contribution.backward()
        if attention is not None:
            stats = attention.statistics
            assert stats["tiled_training"] > forward_calls
            assert stats["recomputed_tiles"] > 0
            assert stats["max_estimated_score_bytes"] > attention.metadata["max_score_bytes"]
            assert 0 < stats["max_tile_score_bytes"] <= attention.metadata["max_score_bytes"]
        gradients = trainable_gradient_snapshot(network)
        for adapter in ("lora_A", "lora_B"):
            assert any(
                gradient is not None and torch.count_nonzero(gradient).item() > 0
                for name, gradient in gradients.items() if adapter in name
            )
        return (
            {f"member_{i}": output.detach() for i, output in enumerate(outputs)},
            {"density_weights": density.weights.detach()},
            {"losses": losses.detach()},
            gradients,
        )

    reference = run()
    # Mirror the planned 1024-context route: force a split across feature
    # groups and retain all queries and keys within each group.
    budget = max(complete_group_score_bytes)
    with install_training_tiled_attention(network, max_score_bytes=budget) as attention:
        actual = run(attention)
        assert attention.statistics["max_query_tile"] == len(context)
    tolerances = [(LOGIT_ATOL, LOGIT_RTOL), (ATOL, RTOL), (LOSS_ATOL, 0), (ATOL, RTOL)]
    for expected, observed, (atol, rtol) in zip(reference, actual, tolerances, strict=True):
        report = compare_named_tensors(expected, observed, atol=atol, rtol=rtol)
        assert report["passed"], {
            "missing": report["missing"],
            "unexpected": report["unexpected"],
            "failed": {name: report["tensors"][name] for name in report["failed"]},
        }
