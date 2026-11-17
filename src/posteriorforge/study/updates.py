"""Stage-wise CPU diagnostics and explicitly gated CUDA qualification updates.

A realized query/measurement case is a training problem. Setup frequency follows
catalogue sampling: an episode with Q queries has Q/total_Q of the batch weight,
exactly as the existing full-graph wrapper. Production remains blocked by I07.
"""
from __future__ import annotations
import math
from contextlib import nullcontext
from typing import Sequence
import torch
from torch import nn
from .model import CANONICAL_FULL_SUPPORT, LEGACY_FINITE, ModelQualificationError, configure_differentiable_forward
from .training import PreparedTeacherForcedEpisode, TargetSupportMismatch, TrainingState

_ROLE_GROUP = {"population": 0, "batch": 0, "physical": 1, "noise": 2}


def _preflight(model, optimizer, episodes, *, law, gradient_clip, allow_unqualified_cpu_diagnostic, device_type="cpu"):
    if not allow_unqualified_cpu_diagnostic:
        raise ModelQualificationError("explicit unqualified CPU-diagnostic opt-in required; I07 does not authorize a campaign" if device_type == "cpu" else "explicit CUDA qualification opt-in required; production is still I07-gated")
    if law not in (LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
        raise ModelQualificationError("explicit implemented density law required; no finite fallback")
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    parameters = list(model.parameters())
    if not parameters or any(value.device.type != device_type for value in parameters + list(model.buffers())):
        raise ModelQualificationError("this unqualified diagnostic update is CPU-only" if device_type == "cpu" else "this qualification update is CUDA-only")
    devices = {value.device for value in parameters + list(model.buffers())}
    if len(devices) != 1:
        raise ModelQualificationError("one local model device is required per qualification worker")
    target_device = parameters[0].device
    if device_type == "cuda" and any(p.is_floating_point() and p.dtype != torch.float32 for p in parameters):
        raise ModelQualificationError("CUDA qualification uses FP32 neural parameters")
    trainable = [p for p in parameters if p.requires_grad]
    if not trainable:
        raise ModelQualificationError("no trainable transformer parameters")
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_parameters) != len({id(p) for p in optimizer_parameters}) or {id(p) for p in optimizer_parameters} != {id(p) for p in trainable}:
        raise ModelQualificationError("optimizer must cover exactly the trainable model parameters")
    if not episodes:
        raise ValueError("a nonempty effective batch of prepared episodes is required")
    counts, total_cases = [], 0
    for episode in episodes:
        if not isinstance(episode, PreparedTeacherForcedEpisode):
            raise TypeError("prepared teacher-forced episodes required")
        if episode.split not in ("train", "diagnostic"):
            raise ModelQualificationError("validation/test episodes cannot update weights")
        if episode.density_law != law:
            raise ModelQualificationError("prepared/update density-law mismatch")
        if type(episode.query_count) is not int or episode.query_count < 1:
            raise ValueError("positive query-case count required")
        dimension = len(episode.parameter_order)
        if not dimension or len(set(episode.parameter_order)) != dimension or len(episode.roles) != dimension or len(episode.stages) != dimension:
            raise ModelQualificationError("incomplete ordered teacher-forced episode")
        try:
            groups = [_ROLE_GROUP[role] for role in episode.roles]
        except KeyError as exc:
            raise ModelQualificationError("unrecognized parameter role") from exc
        group_counts = tuple(groups.count(group) for group in range(3))
        if not all(group_counts):
            raise ModelQualificationError("population/batch, physical and inferred-noise groups required")
        for j, stage in enumerate(episode.stages):
            if stage.index != j or stage.parameter_name != episode.parameter_order[j] or stage.role != episode.roles[j]:
                raise ModelQualificationError("teacher-forced stage order/role mismatch")
            for value in (stage.targets, stage.decoder_log_jac):
                if not isinstance(value, torch.Tensor) or value.device != target_device or value.shape != (episode.query_count,) or not bool(torch.isfinite(value).all()):
                    raise ModelQualificationError("finite CPU target/Jacobian vectors required" if device_type == "cpu" else "finite target/Jacobian vectors on the model CUDA device required")
                if device_type == "cuda" and value.dtype != torch.float64:
                    raise ModelQualificationError("CUDA probability targets and decoder Jacobians must remain FP64")
            for member in getattr(stage.ensemble, "members", ()):
                if any(value.device != target_device for value in (member.context_x, member.context_y, member.query_x)):
                    raise ModelQualificationError("prepared numerical inputs must remain on CPU" if device_type == "cpu" else "prepared numerical inputs must use the model CUDA device")
        counts.append(group_counts)
        total_cases += episode.query_count
    return trainable, counts, total_cases


def _require_finite_gradients(parameters):
    if not any(p.grad is not None for p in parameters):
        raise ModelQualificationError("no trainable parameter received a gradient")
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ModelQualificationError("nonfinite accumulated gradients; optimizer step refused")


def _stagewise_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                         episodes: Sequence[PreparedTeacherForcedEpisode], state: TrainingState,
                         *, law: str, gradient_clip: float = 1.0,
                         allow_unqualified_cpu_diagnostic: bool = False, device_type: str = "cpu",
                         cache_parametrizations: bool = False) -> dict:
    """Backward each conditional, then take one step for the entire batch.

    Each parameter/query contributes
    NLL/(total_query_cases * 3 * parameter_count_in_its_role_group).
    Its NLL includes its own decoder log-Jacobian. Weights stay unchanged across
    conditional forwards. Each backward releases that stage's graph before the
    next forward; only preprocessing tensors and scalar records are retained.
    This enables later GPU memory qualification without claiming measured gains.

    Preflight/forward/score/backward/gradient failures clear gradients, preserve
    counters and never call optimizer.step(). Failures inside optimizer.step()
    itself propagate; transactional rollback of a partially failed optimizer is
    not claimed. Neither qualification primitive authorizes a production campaign.
    """
    if type(cache_parametrizations) is not bool:
        raise TypeError("cache_parametrizations must be an explicit boolean")
    optimizer.zero_grad(set_to_none=True)
    try:
        episodes = tuple(episodes)
        parameters, group_counts, total_cases = _preflight(model, optimizer, episodes,
            law=law, gradient_clip=gradient_clip,
            allow_unqualified_cpu_diagnostic=allow_unqualified_cpu_diagnostic, device_type=device_type)
        if type(state.update) is not int or type(state.data_position) is not int or state.update < 0 or state.data_position < 0:
            raise ValueError("training counters must be nonnegative integers")
        scalar_losses = []
        backward_calls = 0
        for episode, counts in zip(episodes, group_counts):
            for stage in episode.stages:
                configure_differentiable_forward(model)
                if cache_parametrizations:
                    from .training_execution import conditional_forward_cache
                    context = conditional_forward_cache(model, enabled=True)
                else:
                    context = nullcontext()
                # Method wrappers span backward so checkpoint recomputation uses
                # the same per-layer cache scope as the original forward.
                with context:
                    density = stage.ensemble.density(model, law=law)
                    losses = -density.log_prob(stage.targets, log_abs_det_decode=stage.decoder_log_jac)
                    if losses.shape != (episode.query_count,) or not bool(torch.isfinite(losses).all()):
                        raise TargetSupportMismatch(f"{episode.problem_id}:{stage.parameter_name}: nonfinite/zero generating density; no repair")
                    contribution = losses.sum() / (total_cases * 3 * counts[_ROLE_GROUP[stage.role]])
                    if not contribution.requires_grad:
                        raise ModelQualificationError("conditional score disconnected from transformer weights")
                    scalar_losses.append(float(contribution.detach()))
                    contribution.backward()
                    backward_calls += 1
                    del contribution, losses, density
                _require_finite_gradients(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
        _require_finite_gradients(parameters)
        optimizer.step()  # Sole step, after every conditional used unchanged weights.
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise ModelQualificationError("optimizer produced nonfinite parameters; update failed")
        configure_differentiable_forward(model)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    state.update += 1
    state.data_position += total_cases
    return {"update": state.update, "data_position": state.data_position,
        "loss": math.fsum(scalar_losses), "gradient_norm_before_clip": float(norm.detach()),
        "episode_count": len(episodes), "query_case_count": total_cases,
        "conditional_count": backward_calls, "backward_calls": backward_calls,
        "optimizer_steps": 1, "sampling_law": law,
        "query_weighting": "equal realized query cases; setup frequency follows catalogue sampling",
        "execution_mode": "unqualified_cpu_diagnostic" if device_type == "cpu" else "unqualified_cuda_qualification", "scientifically_qualified": False}


def stagewise_cpu_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                         episodes: Sequence[PreparedTeacherForcedEpisode], state: TrainingState,
                         *, law: str, gradient_clip: float = 1.0,
                         allow_unqualified_cpu_diagnostic: bool = False) -> dict:
    """Preserve the explicit CPU-only diagnostic contract."""
    return _stagewise_update(model, optimizer, episodes, state, law=law,
        gradient_clip=gradient_clip, allow_unqualified_cpu_diagnostic=allow_unqualified_cpu_diagnostic)


def stagewise_cuda_qualification_update(model: nn.Module, optimizer: torch.optim.Optimizer,
                         episodes: Sequence[PreparedTeacherForcedEpisode], state: TrainingState,
                         *, law: str, gradient_clip: float = 1.0,
                         allow_cuda_qualification: bool = False, expected_device=None,
                         cache_parametrizations: bool = False) -> dict:
    """One explicitly authorized engineering update; does not qualify a campaign.

    Neural parameters remain FP32 and probability/Jacobian calculations FP64.
    Every conditional is backpropagated before the next; there is exactly one
    optimizer step for the complete effective batch. No autocast or scaler is used.
    """
    if allow_cuda_qualification is not True:
        raise ModelQualificationError("explicit CUDA qualification opt-in required")
    if law != CANONICAL_FULL_SUPPORT:
        raise ModelQualificationError("CUDA qualification requires the agreed canonical full-support law")
    if torch.is_autocast_enabled():
        raise ModelQualificationError("AMP is not enabled by FP32 CUDA qualification")
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != "cuda" for p in parameters):
        raise ModelQualificationError("this qualification update is CUDA-only")
    target = parameters[0].device
    if expected_device is not None:
        expected = torch.device(expected_device)
        if expected.type != "cuda" or (expected.index is not None and target != expected):
            raise ModelQualificationError("CUDA qualification device mismatch")
    if not torch.cuda.is_available():
        raise ModelQualificationError("CUDA qualification requires an available CUDA device")
    torch.cuda.synchronize(target)
    report = _stagewise_update(model, optimizer, episodes, state, law=law,
        gradient_clip=gradient_clip, allow_unqualified_cpu_diagnostic=True, device_type="cuda",
        cache_parametrizations=cache_parametrizations)
    torch.cuda.synchronize(target)
    return dict(report, device=str(target), neural_precision="float32",
                probability_precision="float64", mixed_precision=False,
                parametrization_cache_policy="per_layer_forward_and_checkpoint_recompute.v1" if cache_parametrizations else "stock_uncached")
