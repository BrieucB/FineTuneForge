"""Differentiable TabPFN-v2 projections and explicitly identified density laws.

Built with PriorLabs-TabPFN. This module does not activate a production sampling
policy. The legacy finite-bar law, native half-normal criterion, and selected
canonical full-tail candidate are explicitly named. The new shared-grid law
uses consistent density and sampling but remains scientifically unqualified.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence
import math

import torch
from torch import nn
from torch.nn.utils import parametrize

LEGACY_FINITE = "legacy_finite_v3"
PENDING_FULL_TAIL = "full_tail.pending_ensemble_choice.v1"
CANONICAL_FULL_SUPPORT = "canonical_half_normal_full_support.v1"
NATIVE_FULL_SUPPORT = "native_half_normal_full_support.v1"
BASE_CHECKPOINT_SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
_HALF_NORMAL_MEDIAN = 0.6744897501960817


class ModelQualificationError(RuntimeError):
    """A model, density, or cache identity is not valid for the requested use."""


def _borders(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or len(value) < 2:
        raise ValueError("one finite increasing border vector is required")
    value = value.to(dtype=torch.float64)
    if not bool(torch.isfinite(value).all()) or not bool((value[1:] > value[:-1]).all()):
        raise ValueError("borders must be finite and strictly increasing")
    return value


def _safe_log(value: torch.Tensor) -> torch.Tensor:
    # Zero mass stays zero; this avoids undefined 0*inf backward through log(0).
    positive = value > 0
    return torch.where(positive, torch.log(torch.where(positive, value, 1.0)), -torch.inf)


def _intervals_from_api(api: torch.Tensor) -> torch.Tensor:
    cdf = api.cumsum(-1)
    cdf = cdf / cdf[..., -1:]
    cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), -1)
    return torch.diff(torch.cat((torch.zeros_like(cdf[..., :1]), cdf), -1), dim=-1)


def normalized_interval_weights(raw_weights: torch.Tensor) -> torch.Tensor:
    """The represented finite-CDF intervals used by the corrected sampler.

    No positive floor or tail deletion is introduced. Rounding may collapse an
    interval; its represented mass is then exactly zero and is retained as such.
    """
    raw = raw_weights.to(dtype=torch.float64)
    if raw.ndim < 1 or not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()):
        raise ValueError("finite nonnegative weights are required")
    if not bool((raw.sum(-1) > 0).all()):
        raise ValueError("each row needs positive total mass")
    return _intervals_from_api(torch.softmax(_safe_log(raw), -1))


class BarDensity:
    """A normalized scalar density with an explicit finite or full-support law."""
    def __init__(self, logits: torch.Tensor, borders: torch.Tensor, *, law: str):
        if law not in (LEGACY_FINITE, NATIVE_FULL_SUPPORT, CANONICAL_FULL_SUPPORT):
            raise ValueError("an explicit recognized density law is required")
        self.borders = _borders(borders).to(logits.device)
        self.logits = logits.to(dtype=torch.float64)
        if self.logits.ndim < 1 or self.logits.shape[-1] + 1 != len(self.borders):
            raise ValueError("logit and border dimensions differ")
        good = torch.isfinite(self.logits) | torch.isneginf(self.logits)
        if not bool(good.all()) or not bool(torch.isfinite(self.logits).any(-1).all()):
            raise ValueError("each logit row needs finite mass and no NaN/+inf")
        if law != LEGACY_FINITE and self.logits.shape[-1] < 2:
            raise ValueError("full-support density needs two outer bins")
        self.law = law
        # Finite-v3 scores its represented CDF intervals. Native full-tail
        # density instead keeps direct FP64 masses: CDF subtraction must not
        # erase a positive extreme-tail mass before density evaluation.
        weights = torch.softmax(self.logits, -1)
        self.weights = _intervals_from_api(weights) if law == LEGACY_FINITE else weights
        self.widths = torch.diff(self.borders)

    @classmethod
    def from_weights(cls, weights: torch.Tensor, borders: torch.Tensor, *, law: str):
        return cls(_safe_log(weights), borders, law=law)

    def _values(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.as_tensor(values, dtype=torch.float64, device=self.logits.device)
        values = torch.broadcast_to(values, torch.broadcast_shapes(values.shape, self.logits.shape[:-1]))
        if not bool(torch.isfinite(values).all()):
            raise ValueError("finite target values are required")
        index = torch.searchsorted(self.borders, values.contiguous(), right=True) - 1
        return values, index.clamp(0, len(self.widths) - 1)

    def log_prob(self, values: torch.Tensor, *, log_abs_det_decode=0.0) -> torch.Tensor:
        """Physical log density is latent log density minus the decoder Jacobian."""
        y, index = self._values(values)
        weights = self.weights.expand(y.shape + (len(self.widths),))
        probs = weights.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        result = _safe_log(probs) - torch.log(self.widths[index])
        if self.law == LEGACY_FINITE:
            result = torch.where((y >= self.borders[0]) & (y <= self.borders[-1]), result, -torch.inf)
        else:
            left_scale = self.widths[0] / _HALF_NORMAL_MEDIAN
            right_scale = self.widths[-1] / _HALF_NORMAL_MEDIAN
            left = _safe_log(self.weights[..., 0]) + .5 * math.log(2 / math.pi) - torch.log(left_scale) - .5 * ((self.borders[1] - y) / left_scale) ** 2
            right = _safe_log(self.weights[..., -1]) + .5 * math.log(2 / math.pi) - torch.log(right_scale) - .5 * ((y - self.borders[-2]) / right_scale) ** 2
            result = torch.where(y < self.borders[1], left, result)
            result = torch.where(y >= self.borders[-2], right, result)
        jac = torch.as_tensor(log_abs_det_decode, dtype=result.dtype, device=result.device)
        if not bool(torch.isfinite(jac).all()):
            raise ValueError("decoder log Jacobian must be finite")
        return result - jac

    def pdf(self, values: torch.Tensor, *, log_abs_det_decode=0.0) -> torch.Tensor:
        return self.log_prob(values, log_abs_det_decode=log_abs_det_decode).exp()

    def cdf(self, values: torch.Tensor) -> torch.Tensor:
        y, index = self._values(values)
        weights = self.weights.expand(y.shape + (len(self.widths),))
        cumulative = weights.cumsum(-1)
        before = cumulative - weights
        base = before.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        mass = weights.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        fraction = ((y - self.borders[index]) / self.widths[index]).clamp(0, 1)
        value = base + mass * fraction
        if self.law == LEGACY_FINITE:
            return torch.where(y <= self.borders[0], 0.0, torch.where(y >= self.borders[-1], 1.0, value))
        left = self.weights[..., 0] * torch.special.erfc((self.borders[1] - y) / (math.sqrt(2) * self.widths[0] / _HALF_NORMAL_MEDIAN))
        right = 1 - self.weights[..., -1] * torch.special.erfc((y - self.borders[-2]) / (math.sqrt(2) * self.widths[-1] / _HALF_NORMAL_MEDIAN))
        value = torch.where(y < self.borders[1], left, value)
        return torch.where(y >= self.borders[-2], right, value)

    def sf(self, values: torch.Tensor) -> torch.Tensor:
        """Survival probabilities without subtracting a tiny upper tail from one."""
        y, index = self._values(values)
        weights = self.weights.expand(y.shape + (len(self.widths),))
        after = weights.flip(-1).cumsum(-1).flip(-1) - weights
        base = after.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        mass = weights.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        fraction = ((self.borders[index + 1] - y) / self.widths[index]).clamp(0, 1)
        value = base + mass * fraction
        if self.law == LEGACY_FINITE:
            return torch.where(y <= self.borders[0], 1.0, torch.where(y >= self.borders[-1], 0.0, value))
        left = 1 - self.weights[..., 0] * torch.special.erfc((self.borders[1] - y) / (math.sqrt(2) * self.widths[0] / _HALF_NORMAL_MEDIAN))
        right = self.weights[..., -1] * torch.special.erfc((y - self.borders[-2]) / (math.sqrt(2) * self.widths[-1] / _HALF_NORMAL_MEDIAN))
        value = torch.where(y < self.borders[1], left, value)
        return torch.where(y >= self.borders[-2], right, value)

    def icdf(self, uniforms: torch.Tensor) -> torch.Tensor:
        u = torch.as_tensor(uniforms, dtype=torch.float64, device=self.logits.device)
        u = torch.broadcast_to(u, torch.broadcast_shapes(u.shape, self.logits.shape[:-1]))
        if not bool(torch.isfinite(u).all()) or bool(((u < 0) | (u >= 1)).any()):
            raise ValueError("uniforms must be finite in [0,1)")
        if self.law != LEGACY_FINITE and bool((u <= 0).any()):
            raise ValueError("full-support inverse needs open uniforms; no clipping")
        weights = self.weights.expand(u.shape + (len(self.widths),))
        cdf = weights.cumsum(-1)
        cdf = torch.cat((cdf[..., :-1], torch.ones_like(cdf[..., -1:])), -1)
        index = torch.searchsorted(cdf.contiguous(), u.unsqueeze(-1).contiguous(), right=False).squeeze(-1)
        # Explicit finite zero-uniform policy: first represented positive interval.
        if self.law == LEGACY_FINITE:
            first_positive = (weights > 0).to(torch.int64).argmax(-1)
            index = torch.where(u == 0, first_positive, index)
        chosen = weights.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        if not bool((chosen > 0).all()):
            raise ModelQualificationError("inverse selected a zero-mass interval")
        before = (cdf - weights).gather(-1, index.unsqueeze(-1)).squeeze(-1)
        value = self.borders[index] + self.widths[index] * (u - before) / chosen
        if self.law != LEGACY_FINITE:
            in_left = index == 0
            in_right = index == len(self.widths) - 1
            # Non-tail branches use benign numerators/denominators before
            # division; zero-mass unused tails must not create NaN gradients.
            left_arg = torch.where(in_left, u, .5) / (2 * torch.where(in_left, self.weights[..., 0], 1.0))
            right_arg = torch.where(in_right, 1 - u, .5) / (2 * torch.where(in_right, self.weights[..., -1], 1.0))
            left = self.borders[1] + self.widths[0] / _HALF_NORMAL_MEDIAN * torch.special.ndtri(left_arg)
            right = self.borders[-2] - self.widths[-1] / _HALF_NORMAL_MEDIAN * torch.special.ndtri(right_arg)
            value = torch.where(index == 0, left, value)
            value = torch.where(index == len(self.widths) - 1, right, value)
        if not bool(torch.isfinite(value).all()):
            raise ModelQualificationError("nonfinite inverse; no repair or clipping")
        return value


    def sample(self, sample_shape=(), *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Explicit local primitive; full tails use open uniforms, never clipping."""
        shape = tuple(sample_shape)
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("sample_shape must contain nonnegative integer sizes")
        shape = shape + tuple(self.logits.shape[:-1])
        if self.law == LEGACY_FINITE:
            uniforms = torch.rand(shape, dtype=torch.float64, device=self.logits.device, generator=generator)
        else:
            integers = torch.randint(1, 2**53, shape, dtype=torch.int64, device=self.logits.device, generator=generator)
            uniforms = integers.to(torch.float64) / 2**53
        return self.icdf(uniforms)


def translate_finite_logits(logits: torch.Tensor, source_borders: torch.Tensor, canonical_borders: torch.Tensor) -> torch.Tensor:
    """Corrected differentiable finite CDF translation, before mass normalization."""
    source = _borders(source_borders).to(logits.device)
    canonical = _borders(canonical_borders).to(logits.device)
    if logits.shape[-1] + 1 != len(source):
        raise ValueError("member logit/border dimensions differ")
    probabilities = torch.softmax(logits.to(torch.float64), -1)
    before = probabilities.cumsum(-1) - probabilities
    indices = (torch.searchsorted(source, canonical, right=True) - 1).clamp(0, len(source) - 2)
    shape = logits.shape[:-1] + (len(canonical),)
    index = indices.expand(shape)
    fraction = ((canonical - source[indices]) / (source[indices + 1] - source[indices])).clamp(0, 1)
    cdf = before.gather(-1, index) + probabilities.gather(-1, index) * fraction
    cdf = torch.where(canonical <= source[0], 0.0, torch.where(canonical >= source[-1], 1.0, cdf))
    # SDK canonicalization fixes CDF endpoints, not sample parameters.
    cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf[..., 1:-1], torch.ones_like(cdf[..., -1:])), -1)
    weights = torch.diff(cdf, dim=-1)
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ModelQualificationError("invalid or non-monotone canonical CDF")
    return weights


def canonical_finite_mixture(member_logits: Sequence[torch.Tensor], member_borders: Sequence[torch.Tensor], canonical_borders: torch.Tensor, *, expected_members: int = 8, sampling_borders: torch.Tensor | None = None) -> BarDensity:
    """Corrected CDF translation then raw-mass mean and sampler normalization.

    Logits already include the declared temperature/cancellation policy. Source
    and canonical borders are in the fitted standardized target coordinates;
    sampling_borders, when supplied, are the actual fitted API border bits in
    decoded target units. Never replace their Float32 fit by a Float64 refit.
    """
    if len(member_logits) != expected_members or len(member_borders) != expected_members:
        raise ValueError(f"all {expected_members} configured members are required")
    raw = [translate_finite_logits(x, b, canonical_borders) for x, b in zip(member_logits, member_borders)]
    api_borders = canonical_borders if sampling_borders is None else sampling_borders
    return BarDensity.from_weights(torch.stack(raw).mean(0), api_borders, law=LEGACY_FINITE)


class _PackedProjectionDelta(nn.Module):
    """PEFT-managed rank-r matrices over whole logical projections, never per head."""
    def __init__(self, original: torch.Tensor, *, output: bool = False):
        super().__init__()
        self.output = output
        self.shape = tuple(original.shape)
        self.stacked = original.ndim == 4 and not output
        count = original.shape[0] if self.stacked else 1
        if output:
            n_in, n_out = original.shape[0] * original.shape[1], original.shape[2]
        else:
            n_in = original.shape[-1]
            n_out = math.prod(original.shape[-3:-1])
        self.branches = nn.ModuleList([nn.Linear(n_in, n_out, bias=False, device=original.device, dtype=original.dtype) for _ in range(count)])
        for branch in self.branches:
            nn.init.zeros_(branch.weight)
            branch.weight.requires_grad_(False)

    @staticmethod
    def _weight(branch: nn.Module) -> torch.Tensor:
        if hasattr(branch, "get_base_layer"):
            weight = branch.get_base_layer().weight
            if not branch.disable_adapters and not branch.merged:
                for name in branch.active_adapters:
                    if name in branch.lora_A:
                        weight = weight + branch.get_delta_weight(name)
            return weight
        return branch.weight

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        matrices = [self._weight(branch) for branch in self.branches]
        if self.output:
            delta = matrices[0].T.reshape(self.shape)
        elif self.stacked:
            delta = torch.stack([x.reshape(self.shape[1:]) for x in matrices])
        else:
            delta = matrices[0].reshape(self.shape)
        return original + delta


def prepare_packed_attention(model: nn.Module) -> list[str]:
    """Expose packed TabPFN Q/K/V/output matrices as PEFT-compatible 2D branches."""
    targets = []
    for module_name, module in list(model.named_modules()):
        if not hasattr(module, "compute_qkv") or not hasattr(module, "_w_out"):
            continue
        if parametrize.is_parametrized(module):
            raise ValueError("attention is already parametrized")
        for name in ("_w_qkv", "_w_q", "_w_kv", "_w_k", "_w_v", "_w_out"):
            original = getattr(module, name, None)
            if original is None:
                continue
            if not isinstance(original, nn.Parameter) or original.ndim not in (3, 4):
                raise ModelQualificationError("unexpected packed attention layout")
            wrapper = _PackedProjectionDelta(original, output=name == "_w_out")
            parametrize.register_parametrization(module, name, wrapper)
            targets.extend(f"{module_name}.parametrizations.{name}.0.branches.{i}".lstrip(".") for i in range(len(wrapper.branches)))
    if not targets:
        raise ModelQualificationError("no supported packed TabPFN attention modules found")
    return targets


def attach_lora(model: nn.Module, *, rank: int, base_model_id: str = "Prior-Labs/TabPFN-v2-reg") -> nn.Module:
    """Attach real HF PEFT adapters to attention, MLP and regression output layers."""
    if rank not in (4, 16):
        raise ValueError("approved LoRA ranks are 4 and 16")
    from peft import LoraConfig, get_peft_model
    packed_targets = prepare_packed_attention(model)
    normal_targets = [name for name, module in model.named_modules() if isinstance(module, nn.Linear) and (".mlp." in name or ".second_mlp." in name or name.startswith("decoder_dict."))]
    targets = packed_targets + normal_targets
    if not normal_targets or not any(name.startswith("decoder_dict.") for name in normal_targets):
        raise ModelQualificationError("MLP/output target coverage is incomplete")
    config = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none", target_modules=targets, inference_mode=False)
    adapted = get_peft_model(model, config)
    adapted.peft_config["default"].base_model_name_or_path = base_model_id
    adapted.eval()  # Eval mode preserves all layers; it does not disable autograd.
    return adapted


def merge_lora(model: nn.Module) -> nn.Module:
    """Merge PEFT linears and remove packed parametrizations into ordinary weights."""
    if not hasattr(model, "merge_and_unload"):
        raise ValueError("a PEFT-wrapped model is required")
    plain = model.merge_and_unload()
    for module in list(plain.modules()):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None:
            for name in list(parametrizations.keys()):
                if isinstance(parametrizations[name][0], _PackedProjectionDelta):
                    parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    return plain


def configure_differentiable_forward(model: nn.Module) -> None:
    """Disable inference-only caches/peak-memory mutation while retaining gradients."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "empty_trainset_representation_cache"):
        base.empty_trainset_representation_cache()
    if hasattr(base, "cache_trainset_representation"):
        base.cache_trainset_representation = False
    for module in base.modules():
        if hasattr(module, "empty_kv_cache"):
            module.empty_kv_cache()
        if hasattr(module, "save_peak_mem_factor"):
            module.save_peak_mem_factor = None
    base.eval()


@dataclass(frozen=True)
class PreparedMember:
    """Only context labels and query features; no unknown query label input."""
    context_x: torch.Tensor
    context_y: torch.Tensor
    query_x: torch.Tensor
    source_borders: torch.Tensor | None = None
    categorical_indices: tuple[int, ...] = ()

    def validate(self) -> None:
        if self.context_x.ndim != 2 or self.query_x.ndim != 2 or self.context_y.ndim != 1:
            raise ValueError("context/query matrices and one context-label vector are required")
        if len(self.context_x) != len(self.context_y) or len(self.context_y) < 1 or len(self.query_x) < 1:
            raise ValueError("nonempty paired context/query rows are required")
        if self.context_x.shape[1] != self.query_x.shape[1] or self.context_x.shape[1] < 1:
            raise ValueError("context/query feature widths differ")
        if not all(bool(torch.isfinite(x).all()) for x in (self.context_x, self.context_y, self.query_x)):
            raise ValueError("prepared neural inputs must be finite")


class TabPFNBackbone(nn.Module):
    """Actual pinned TabPFN network with gradient-enabled prepared-member forwards."""
    def __init__(self, backbone: nn.Module, borders: torch.Tensor, *, checkpoint_sha256: str):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("borders", _borders(borders).clone())
        self.checkpoint_sha256 = checkpoint_sha256
        configure_differentiable_forward(backbone)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, expected_sha256: str = BASE_CHECKPOINT_SHA256):
        from importlib.metadata import version
        if version("tabpfn") != "2.0.9":
            raise ModelQualificationError("the qualified backbone requires TabPFN 2.0.9")
        path = Path(checkpoint)
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ModelQualificationError("missing checkpoint or checkpoint checksum mismatch")
        from tabpfn.model.loading import load_model_criterion_config
        model, criterion, _ = load_model_criterion_config(path, check_bar_distribution_criterion=True, cache_trainset_representation=False, which="regressor", download=False, model_seed=0)
        # The baseline is frozen. CPU biased-Linear kernels can select a
        # different FP32 path when base weights require gradients, even inside
        # no_grad; compare zero adapters with the same frozen-base semantics.
        model.requires_grad_(False)
        return cls(model, criterion.borders, checkpoint_sha256=expected_sha256)

    def forward_member(self, member: PreparedMember) -> torch.Tensor:
        if torch.is_inference_mode_enabled():
            raise ModelQualificationError("gradient-enabled forward cannot run in inference_mode")
        member.validate()
        configure_differentiable_forward(self.backbone)
        x = torch.cat((member.context_x, member.query_x), 0).unsqueeze(1)
        y = member.context_y.unsqueeze(1)
        logits = self.backbone(x, y, single_eval_pos=len(y), only_return_standard_out=True, categorical_inds=list(member.categorical_indices))
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[1] != 1:
            raise ModelQualificationError("unexpected actual TabPFN output shape")
        return logits[:, 0]

    def forward(self, members: Sequence[PreparedMember], *, expected_members: int = 8) -> list[torch.Tensor]:
        if len(members) != expected_members:
            raise ValueError(f"all {expected_members} configured members are required")
        return [self.forward_member(member) for member in members]

    def metadata(self) -> dict:
        return {"checkpoint_sha256": self.checkpoint_sha256, "backbone": "TabPFN-v2-regressor", "source_version": "2.0.9", "gradient_enabled_path": True, "sampling_law_activated": False, "scientifically_qualified": False}


@dataclass(frozen=True)
class PreparedEnsemble:
    """All eight exact stock fitted preprocessors, with no neural inference call."""
    members: tuple[PreparedMember, ...]
    cancel_masks: tuple[torch.Tensor | None, ...]
    canonical_borders: torch.Tensor
    sampling_borders: torch.Tensor
    temperature: float
    source_policy: str = "legacy_strict"

    def adjusted_logits(self, network: TabPFNBackbone) -> list[torch.Tensor]:
        logits = network(self.members)
        adjusted = []
        for output, mask in zip(logits, self.cancel_masks):
            output = output.float() / self.temperature
            if mask is not None:
                output = output.masked_fill(mask.to(output.device), -torch.inf)
            adjusted.append(output)
        return adjusted

    def density(self, network: TabPFNBackbone, *, law: str) -> BarDensity:
        if law not in (LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
            raise ModelQualificationError("a concrete ensemble law is required; no finite fallback")
        mixture = canonical_finite_mixture if law == LEGACY_FINITE else canonical_full_support_mixture
        return mixture(self.adjusted_logits(network), [m.source_borders for m in self.members], self.canonical_borders, sampling_borders=self.sampling_borders)

    def sampling_metadata(self, *, law: str) -> dict:
        return {**sampling_law_metadata(law), "source_policy": self.source_policy}


def prepare_fitted_regressor_members(regressor, query, *, device: str | torch.device = "cpu", source_policy: str = "legacy_strict") -> PreparedEnsemble:
    """Extract numerical stock preprocessing without the inference-mode executor.

    Fit this separate context/preprocessor object, never the trainable backbone:
    refitting the sklearn wrapper may reload original weights. Query labels are
    deliberately absent. The default retains the strict legacy-v3 source guard.
    The explicit cancelled_zero_width source policy additionally accepts only
    upstream-cancelled interior zero-mass bars, preserving all original bits and
    members. This does not activate a new sampling or ensemble law.
    """
    import numpy as np
    from tabpfn.inference import InferenceEngineCachePreprocessing
    from tabpfn.regressor import _transform_borders_one
    if getattr(regressor, "fit_mode", None) != "fit_preprocessors" or type(getattr(regressor, "executor_", None)) is not InferenceEngineCachePreprocessing:
        raise ModelQualificationError("a fitted stock cache-preprocessing regressor is required")
    if source_policy not in ("legacy_strict", "cancelled_zero_width"):
        raise ValueError("recognized explicit source-border policy required")
    executor = regressor.executor_
    fields = (executor.X_trains, executor.y_trains, executor.cat_ixs, executor.preprocessors, executor.ensemble_configs)
    if any(len(x) != 8 for x in fields) or getattr(regressor, "average_before_softmax", True):
        raise ModelQualificationError("all eight probability-averaged members are required")
    query = np.asarray(query)
    if query.ndim != 2 or not np.isfinite(query).all():
        raise ValueError("one finite numeric query matrix is required; no query labels")
    members, masks = [], []
    standardized = regressor.bardist_.borders.detach().cpu()
    for member_index, (x, y, cats, preprocessor, config) in enumerate(zip(*fields)):
        if config.target_transform is None:
            mask, descending, borders = None, False, standardized.numpy().copy()
        else:
            mask, descending, borders = _transform_borders_one(standardized.numpy(), target_transform=config.target_transform, repair_nan_borders_after_transform=regressor.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM)
        if descending:
            raise ModelQualificationError("descending target transform needs independent qualification")
        source = torch.as_tensor(borders, dtype=torch.float32, device=device)
        if source_policy == "legacy_strict":
            try:
                _borders(source)
            except ValueError as exc:
                raise ModelQualificationError(f"member {member_index} has a non-strict source border grid; no head/border repair") from exc
        else:
            widths = torch.diff(source)
            if not bool(torch.isfinite(source).all()) or bool((widths < 0).any()) or not bool((widths[[0,-1]] > 0).all()):
                raise ModelQualificationError("invalid source grid/outer-tail scales; no border repair")
            zero = widths == 0
            if bool(zero.any()) and (mask is None or not bool(torch.as_tensor(mask,device=device)[zero].all())):
                raise ModelQualificationError("zero-width source bin lacks exact upstream logit cancellation")
        member = PreparedMember(torch.as_tensor(x, dtype=torch.float32, device=device), torch.as_tensor(y, dtype=torch.float32, device=device).flatten(), torch.as_tensor(preprocessor.transform(query).X, dtype=torch.float32, device=device), source, tuple(cats))
        member.validate()
        members.append(member)
        masks.append(None if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=device))
    temperature = float(regressor.softmax_temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("positive finite member temperature required")
    return PreparedEnsemble(tuple(members), tuple(masks), standardized.to(device), regressor.renormalized_criterion_.borders.detach().to(device), temperature, source_policy)


def weights_fingerprint(model: nn.Module) -> str:
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CacheIdentity:
    weights_sha256: str
    preprocessing_sha256: str
    context_sha256: str
    protocol_sha256: str

    def require_same(self, other: "CacheIdentity") -> None:
        if self != other:
            raise ModelQualificationError("stale model/adapter/preprocessing/context cache")


def full_support_cdf_grid(logits: torch.Tensor, source_borders: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native-shaped full-tail CDF/SF at a shared grid, without a Q*K*K tensor.

    Nondecreasing inverse-target grids may contain upstream-cancelled interior
    bars. Their logits MUST be -inf, proving exactly zero probability for every
    query. No border moves, positive mass is discarded, or member is omitted.
    Positive-mass collapsed bars and nonpositive outer-tail scales are rejected.
    This primitive does not choose/activate an ensemble canonicalization policy.
    """
    source = torch.as_tensor(source_borders, dtype=torch.float64, device=logits.device)
    grid = torch.as_tensor(values, dtype=torch.float64, device=logits.device)
    if source.ndim != 1 or len(source) < 3 or grid.ndim != 1 or not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(grid).all()):
        raise ValueError("finite source borders and one evaluation grid are required")
    widths = torch.diff(source)
    if bool((widths < 0).any()) or not bool((widths[[0,-1]] > 0).all()):
        raise ModelQualificationError("nondecreasing source borders and positive outer-tail scales required")
    if logits.ndim < 1 or logits.shape[-1] != len(widths):
        raise ValueError("source logit/border dimensions differ")
    good = torch.isfinite(logits) | torch.isneginf(logits)
    if not bool(good.all()) or not bool(torch.isfinite(logits).any(-1).all()):
        raise ValueError("source logit rows must contain finite mass without NaN/+inf")
    collapsed = widths == 0
    if bool(collapsed.any()) and not bool(torch.isneginf(logits[..., collapsed]).all()):
        raise ModelQualificationError("positive-mass zero-width source bar; no mass deletion")
    probability = torch.softmax(logits.to(torch.float64), -1)
    cumulative = probability.cumsum(-1)
    before = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), -1)
    reverse = probability.flip(-1).cumsum(-1).flip(-1)
    after = torch.cat((reverse[..., 1:], torch.zeros_like(reverse[..., :1])), -1)
    index = (torch.searchsorted(source, grid.contiguous(), right=True) - 1).clamp(0, len(widths)-1)
    selected_widths = widths[index]
    interior = (grid >= source[1]) & (grid < source[-2])
    if bool(((selected_widths == 0) & interior).any()):
        raise ModelQualificationError("invalid collapsed interior CDF interval")
    # The benign denominator is used only for branches replaced by outer tails.
    denominator = torch.where(selected_widths > 0, selected_widths, 1.0)
    fraction = ((grid-source[index])/denominator).clamp(0,1)
    shape = logits.shape[:-1] + (len(grid),)
    expanded_index = index.expand(shape)
    chosen = probability.gather(-1, expanded_index)
    cdf = before.gather(-1, expanded_index) + chosen*fraction
    survival = after.gather(-1, expanded_index) + chosen*(1-fraction)
    left = probability[..., :1] * torch.special.erfc((source[1]-grid)/(math.sqrt(2)*widths[0]/_HALF_NORMAL_MEDIAN))
    right = probability[..., -1:] * torch.special.erfc((grid-source[-2])/(math.sqrt(2)*widths[-1]/_HALF_NORMAL_MEDIAN))
    cdf = torch.where(grid < source[1], left, torch.where(grid >= source[-2],1-right,cdf))
    survival = torch.where(grid < source[1],1-left,torch.where(grid >= source[-2],right,survival))
    return cdf, survival


def translate_full_support_logits(logits: torch.Tensor, source_borders: torch.Tensor, canonical_borders: torch.Tensor) -> torch.Tensor:
    """Project native-shaped source CDF masses onto canonical full-tail bins.

    Outer bins cover (-inf,canonical[1]) and [canonical[-2],inf). Their
    finite outer border positions set half-normal scales, never truncation.
    This preserves the exact upstream cancellation of collapsed source bars.
    """
    canonical = _borders(canonical_borders).to(logits.device)
    if len(canonical) < 3:
        raise ValueError("full-tail canonical grid needs at least two bins")
    cdf, survival = full_support_cdf_grid(logits, source_borders, canonical[1:-1])
    source = torch.as_tensor(source_borders, dtype=torch.float64, device=logits.device)
    if torch.equal(source, canonical):
        # Identity projection preserves direct masses even below CDF resolution.
        raw = torch.softmax(logits.to(torch.float64), -1)
    else:
        raw = torch.cat((cdf[..., :1], torch.diff(cdf, dim=-1), survival[..., -1:]), -1)
    if not bool(torch.isfinite(raw).all()) or bool((raw < 0).any()) or not bool((raw.sum(-1)>0).all()):
        raise ModelQualificationError("invalid full-tail projected masses; no clipping or mass floor")
    return raw


def canonical_full_support_mixture(member_logits: Sequence[torch.Tensor], member_borders: Sequence[torch.Tensor], canonical_borders: torch.Tensor, *, expected_members: int = 8, sampling_borders: torch.Tensor | None = None) -> BarDensity:
    """Selected shared-grid full-tail candidate, identically used by all models.

    Each member is interpreted as an inverse-border histogram with half-normal
    outer tails. Its CDF at canonical inner borders defines projected masses;
    averaging all member masses precedes one FP64 normalization. The resulting
    canonical histogram uses uniform interior bins and half-normal outer tails.
    This is a coherent shared-grid projection, not an exact nonlinear target-
    transform mixture. Native identity/affine cases retain native density.
    Production use still requires I07 qualification of the complete workflow.
    """
    if len(member_logits) != expected_members or len(member_borders) != expected_members:
        raise ValueError(f"all {expected_members} configured members are required")
    raw = [translate_full_support_logits(x,b,canonical_borders) for x,b in zip(member_logits,member_borders)]
    borders = canonical_borders if sampling_borders is None else sampling_borders
    density = BarDensity.from_weights(torch.stack(raw).mean(0),borders,law=CANONICAL_FULL_SUPPORT)
    if not bool((density.weights[..., [0,-1]]>0).all()):
        raise ModelQualificationError("represented full-tail mass exhausted; no floor or tail repair")
    return density


def sampling_law_metadata(law: str) -> dict:
    if law not in (LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
        raise ModelQualificationError("an explicit concrete ensemble law is required")
    metadata = {"sampling_law":law,"members":8,"probability_dtype":"torch.float64","neural_dtype":"torch.float32","temperature_applications":1,"canonical_fitted_api_border_bits_preserved":True,"physical_parameter_clipping":False,"positive_mass_floor":False,"scientifically_qualified":False,"production_gate":"I07"}
    if law == LEGACY_FINITE:
        metadata.update(source_policy="legacy_strict",outer_bins="finite uniform bars",weights="raw finite-CDF translations averaged across all8 before represented-CDF normalization")
    else:
        metadata.update(source_policy="cancelled_zero_width",outer_bins="half-normal tails outside first/last inner border",tail_scale="corresponding outer finite border width / Phi^-1(0.75)",member_interpretation="native-shaped histogram in inverse-border coordinates; not exact nonlinear inverse-transform mixture",weights="member CDF at canonical inner borders; stable SF for last mass; average all8 raw masses then direct FP64 normalization",uniforms="open FP64 k/2^53 with integer k in1..2^53-1; no clipping")
    return metadata
