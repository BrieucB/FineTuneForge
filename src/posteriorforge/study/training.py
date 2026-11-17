"""CPU-testable fine-tuning primitives; no campaign or job is launched here.

The legacy finite sampler has bounded conditional support despite unbounded
Gaussian prior coordinates. Its normalized score can be tested, but a positive
out-of-support target probability makes its population expected log loss
infinite. This is a qualification question for I07, not permission to drop
examples, clip values, or silently activate a different tail law. The selected
new-study candidate uses consistent canonical full tails for baseline/adapted
models; retained scientific/GPU I07 qualification is still pending.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence
import json
import math
import os
import random
import shutil
import uuid

import numpy as np
import torch
from torch import nn

from .model import LEGACY_FINITE, PENDING_FULL_TAIL, CANONICAL_FULL_SUPPORT, BarDensity, ModelQualificationError, PreparedEnsemble, configure_differentiable_forward


class TargetSupportMismatch(ModelQualificationError):
    """A known generating target has zero density under the selected sampler law."""


@dataclass(frozen=True)
class TrainingConfig:
    """Selected full-tail candidate; production qualification remains mandatory.

    Optimizer construction may use these hyperparameters for CPU fixtures.
    Every density-scoring operation still requires a concrete law explicitly;
    pending never dispatches to finite scoring. Campaigns remain blocked.
    """
    method: str = "lora"
    rank: int = 4
    learning_rate: float = 1e-4
    max_updates: int = 1000
    effective_batch: int = 16
    gradient_clip: float = 1.0
    seed: int = 0
    density_law: str = CANONICAL_FULL_SUPPORT
    sampler_law_qualified: bool = False

    def validate(self) -> None:
        if self.method not in ("lora", "full_finetune"):
            raise ValueError("training method must be lora or full_finetune")
        if self.method == "lora" and self.rank not in (4, 16):
            raise ValueError("approved LoRA ranks are 4 and 16")
        for name in ("max_updates", "effective_batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01:
            raise ValueError("learning rate must be finite in (0,.01]")
        if not math.isfinite(self.gradient_clip) or self.gradient_clip <= 0:
            raise ValueError("gradient clip must be positive")
        if self.sampler_law_qualified:
            raise ModelQualificationError("qualification cannot be asserted by a configuration boolean")
        if self.density_law not in (PENDING_FULL_TAIL, LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
            raise ModelQualificationError("a concrete new training/sampling law is not activated by this CPU implementation")

    def require_production_qualification(self) -> None:
        self.validate()
        raise ModelQualificationError("I07 retained sampler-law qualification is pending; CPU unit tests/configuration flags do not authorize a campaign")


@dataclass
class TrainingState:
    update: int = 0
    data_position: int = 0
    stream_id: str = ""


def balanced_problem_loss(parameter_losses: Sequence[torch.Tensor], parameter_roles: Sequence[Sequence[str]]) -> torch.Tensor:
    """Equal problem influence and equal means of the three scientific role groups."""
    if not parameter_losses or len(parameter_losses) != len(parameter_roles):
        raise ValueError("paired nonempty problem losses and roles are required")
    mapping = {"population": 0, "batch": 0, "physical": 1, "noise": 2}
    problem_losses = []
    for losses, roles in zip(parameter_losses, parameter_roles):
        if losses.ndim != 1 or len(losses) != len(roles) or not len(roles):
            raise ValueError("each problem needs one scalar loss per declared parameter")
        if not bool(torch.isfinite(losses).all()):
            raise TargetSupportMismatch("nonfinite proper density score; no target dropping or clipping")
        try:
            groups = [mapping[role] for role in roles]
        except KeyError as exc:
            raise ValueError("unrecognized parameter role") from exc
        if set(groups) != {0, 1, 2}:
            raise ValueError("population/batch, physical and inferred noise groups are all required")
        problem_losses.append(torch.stack([losses[torch.tensor([g == group for g in groups], device=losses.device)].mean() for group in range(3)]).mean())
    return torch.stack(problem_losses).mean()


def conditional_nll(logits: torch.Tensor, borders: torch.Tensor, generating_targets: torch.Tensor, *, law: str, log_abs_det_decode=0.0) -> torch.Tensor:
    """Score the named sampler law, including bin widths and decoder Jacobians."""
    if law == PENDING_FULL_TAIL:
        raise ModelQualificationError("pending full-tail metadata is not an executable law; no finite fallback")
    density = BarDensity(logits, borders, law=law)
    losses = -density.log_prob(generating_targets, log_abs_det_decode=log_abs_det_decode)
    if not bool(torch.isfinite(losses).all()):
        raise TargetSupportMismatch("generating target is outside represented sampling support or has nonfinite density")
    return losses


def make_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    config.validate()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(parameters, lr=config.learning_rate, betas=(.9, .999), eps=1e-8, weight_decay=0)


def configure_full_finetune(model: nn.Module) -> nn.Module:
    model.requires_grad_(True)
    configure_differentiable_forward(model)
    return model


def train_update(model: nn.Module, optimizer: torch.optim.Optimizer, parameter_losses: Sequence[torch.Tensor], parameter_roles: Sequence[Sequence[str]], state: TrainingState, *, gradient_clip: float = 1.0) -> dict:
    """One explicit optimizer update; callers own episodes and execution authorization."""
    if not math.isfinite(gradient_clip) or gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    loss = balanced_problem_loss(parameter_losses, parameter_roles)
    optimizer.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        raise ModelQualificationError("the loss is disconnected from trainable transformer weights")
    loss.backward()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not any(p.grad is not None for p in parameters):
        raise ModelQualificationError("no trainable parameter received a gradient")
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
        optimizer.zero_grad(set_to_none=True)
        raise ModelQualificationError("nonfinite gradients; optimizer step refused")
    norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True)
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in parameters):
        raise ModelQualificationError("optimizer produced nonfinite parameters; run failed")
    state.update += 1
    state.data_position += len(parameter_losses)
    return {"update": state.update, "data_position": state.data_position, "loss": float(loss.detach()), "gradient_norm_before_clip": float(norm.detach()), "scientifically_qualified": False}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(x) for x in value]
    return value


def _tuple_tree(value):
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def save_training_checkpoint(output: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, state: TrainingState, *, config: TrainingConfig, episode_rng: np.random.Generator | None = None) -> dict:
    """Atomically save local optimizer/RNG continuation; public weights use safetensors."""
    config.validate()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        numpy_state = np.random.get_state()
        rng = {"python": _jsonable(random.getstate()), "numpy_global": _jsonable(numpy_state), "episode_generator": None if episode_rng is None else _jsonable(episode_rng.bit_generator.state)}
        _write_json(temporary / "rng.json", rng)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state()}, temporary / "training.pt")
        manifest = {"schema": "finetuneforge.training_checkpoint.v1", "state": asdict(state), "config": asdict(config), "trainable_parameter_names": [name for name, value in model.named_parameters() if value.requires_grad], "files": {"rng.json": _hash(temporary / "rng.json"), "training.pt": _hash(temporary / "training.pt")}, "scientifically_qualified": False}
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_training_checkpoint(directory: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, *, config: TrainingConfig, expected_stream_id: str, episode_rng: np.random.Generator | None = None) -> TrainingState:
    """Restore only a matching, checksummed local checkpoint with safe torch loading."""
    config.validate()
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != "finetuneforge.training_checkpoint.v1" or manifest.get("config") != asdict(config):
        raise ModelQualificationError("checkpoint configuration mismatch")
    if manifest["state"]["stream_id"] != expected_stream_id:
        raise ModelQualificationError("episode stream mismatch")
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if names != manifest["trainable_parameter_names"]:
        raise ModelQualificationError("trainable parameter inventory mismatch")
    if set(manifest.get("files", {})) != {"rng.json", "training.pt"}:
        raise ModelQualificationError("training checkpoint file inventory mismatch")
    for name, checksum in manifest["files"].items():
        if name not in ("rng.json", "training.pt") or _hash(directory / name) != checksum:
            raise ModelQualificationError("training checkpoint checksum mismatch")
    rng = json.loads((directory / "rng.json").read_text())
    if rng["episode_generator"] is not None and episode_rng is None:
        raise ModelQualificationError("an explicit episode generator is required for this checkpoint")
    value = torch.load(directory / "training.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    torch.set_rng_state(value["torch_rng"])
    random.setstate(_tuple_tree(rng["python"]))
    global_state = rng["numpy_global"]
    np.random.set_state((global_state[0], np.asarray(global_state[1], dtype=np.uint32), *global_state[2:]))
    if episode_rng is not None and rng["episode_generator"] is not None:
        episode_rng.bit_generator.state = rng["episode_generator"]
    return TrainingState(**manifest["state"])


def save_adapter(output: str | Path, model: nn.Module, *, checkpoint_sha256: str) -> dict:
    if not hasattr(model, "save_pretrained"):
        raise ValueError("a PEFT-wrapped model is required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    files = {p.name: _hash(p) for p in output.iterdir() if p.is_file()}
    if "adapter_model.safetensors" not in files or "adapter_config.json" not in files:
        raise ModelQualificationError("PEFT did not emit the expected safe adapter artifacts")
    manifest = {"schema": "finetuneforge.adapter.v1", "base_checkpoint_sha256": checkpoint_sha256, "files": files, "scientifically_qualified": False}
    _write_json(output / "adapter_manifest.json", manifest)
    return manifest


def load_adapter(base_model: nn.Module, directory: str | Path, *, expected_checkpoint_sha256: str, trainable: bool = True) -> nn.Module:
    from peft import PeftModel
    from .model import prepare_packed_attention
    directory = Path(directory)
    manifest = json.loads((directory / "adapter_manifest.json").read_text())
    if manifest.get("schema") != "finetuneforge.adapter.v1" or manifest.get("base_checkpoint_sha256") != expected_checkpoint_sha256:
        raise ModelQualificationError("adapter base checkpoint mismatch")
    if not {"adapter_model.safetensors", "adapter_config.json"}.issubset(manifest.get("files", {})):
        raise ModelQualificationError("adapter file inventory mismatch")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or _hash(directory / name) != checksum:
            raise ModelQualificationError("adapter checksum mismatch")
    prepare_packed_attention(base_model)
    model = PeftModel.from_pretrained(base_model, directory, is_trainable=trainable)
    configure_differentiable_forward(model)
    return model


def export_full_weights(output: str | Path, model: nn.Module) -> dict:
    from safetensors.torch import save_file
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}, str(output))
    return {"file": output.name, "sha256": _hash(output), "scientifically_qualified": False}


@dataclass(frozen=True)
class TeacherForcedArrays:
    """One conditional's public evidence and separately held generating label."""
    index: int
    parameter_name: str
    role: str
    context_features: np.ndarray
    context_targets: np.ndarray
    query_features: np.ndarray
    query_targets: np.ndarray
    decoder_log_jac: np.ndarray


def teacher_forced_inputs(context_bank, query_bank) -> tuple[TeacherForcedArrays, ...]:
    """Construct arbitrary-D conditionals; current/future labels never enter X.

    These are training/probability-scoring inputs, not posterior-generation
    inputs: only known generating *predecessors* are teacher-forced. Independent
    context and query worlds are mandatory. Final-test cases never enter this
    preparation path. The complete-prior transport is causal in parameter order.
    """
    from .banks import PairedBank
    from .science import encode_prior, log_abs_det_decode
    if not isinstance(context_bank, PairedBank) or not isinstance(query_bank, PairedBank):
        raise TypeError("paired context and query banks are required")
    spec = context_bank.spec
    if spec.fingerprint != query_bank.spec.fingerprint:
        raise ValueError("context/query problem fingerprints differ")
    if spec.split == "test":
        raise ModelQualificationError("test cases cannot enter teacher-forced preparation")
    context_rows = {row.tobytes() for row in context_bank.theta}
    if any(row.tobytes() in context_rows for row in query_bank.theta):
        raise ModelQualificationError("query generating parameters occur in the context bank")
    context_z = encode_prior(spec, context_bank.theta)
    query_z = encode_prior(spec, query_bank.theta)
    jac = log_abs_det_decode(spec, query_z, per_parameter=True)
    if jac.shape != query_z.shape or not np.isfinite(jac).all():
        raise ModelQualificationError("one finite decoder log-Jacobian per parameter is required")

    def readonly(value):
        result = np.array(value, dtype=np.float64, order="C", copy=True)
        result.setflags(write=False)
        return result

    return tuple(TeacherForcedArrays(j, parameter.name, parameter.role,
        readonly(np.concatenate((context_bank.observations, context_z[:, :j]), axis=1)),
        readonly(context_z[:, j]),
        readonly(np.concatenate((query_bank.observations, query_z[:, :j]), axis=1)),
        readonly(query_z[:, j]), readonly(jac[:, j]))
        for j, parameter in enumerate(spec.parameters))


@dataclass(frozen=True)
class PreparedTeacherForcedStage:
    index: int
    parameter_name: str
    role: str
    ensemble: PreparedEnsemble
    targets: torch.Tensor
    decoder_log_jac: torch.Tensor


@dataclass(frozen=True)
class PreparedTeacherForcedEpisode:
    problem_id: str
    problem_fingerprint: str
    split: str
    parameter_order: tuple[str, ...]
    roles: tuple[str, ...]
    context_count: int
    query_count: int
    density_law: str
    source_policy: str
    stages: tuple[PreparedTeacherForcedStage, ...]


def prepare_teacher_forced_episode(context_bank, query_bank, *, regressor_factory, law: str, device: str | torch.device = "cpu", source_policy: str | None = None) -> PreparedTeacherForcedEpisode:
    """Fit stock eight-member preprocessing independently of trainable weights.

    The explicitly supplied factory must return a stock TabPFN-v2 regressor with
    fit_preprocessors, eight members and probability averaging, using the pinned
    cached checkpoint. Fitting this separate wrapper never fits neural weights.
    Only preprocessed numerical tensors, border transforms and cancellation
    masks survive; no weight-dependent neural context or fitted model is kept.
    No model downloads, posterior draws, optimizer step or campaign occurs here.
    """
    from .model import prepare_fitted_regressor_members
    if law not in (LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
        raise ModelQualificationError("a concrete full-tail ensemble law is required; no finite fallback")
    if source_policy is None:
        source_policy = "legacy_strict" if law == LEGACY_FINITE else "cancelled_zero_width"
    if law == LEGACY_FINITE and source_policy != "legacy_strict":
        raise ModelQualificationError("legacy finite scoring retains its original strict source-grid guard")
    if not callable(regressor_factory):
        raise TypeError("an explicit offline stock-regressor factory is required")
    arrays = teacher_forced_inputs(context_bank, query_bank)
    stages = []
    for stage in arrays:
        regressor = regressor_factory()
        # Fit owned copies: scientific bank arrays are immutable and unchanged.
        regressor.fit(stage.context_features.copy(), stage.context_targets.copy())
        if hasattr(regressor.executor_.model, "peft_config"):
            raise ModelQualificationError("preprocessing must use a separate pristine stock backbone")
        prepared = prepare_fitted_regressor_members(regressor, stage.query_features.copy(), device=device, source_policy=source_policy)
        stages.append(PreparedTeacherForcedStage(stage.index, stage.parameter_name, stage.role, prepared,
            torch.tensor(stage.query_targets, dtype=torch.float64, device=device),
            torch.tensor(stage.decoder_log_jac, dtype=torch.float64, device=device)))
    spec = context_bank.spec
    return PreparedTeacherForcedEpisode(spec.id, spec.fingerprint, spec.split,
        tuple(spec.parameter_order), tuple(spec.roles), len(context_bank.theta), len(query_bank.theta), law, source_policy, tuple(stages))


@dataclass(frozen=True)
class TeacherForcedScore:
    problem_id: str
    problem_fingerprint: str
    split: str
    roles: tuple[str, ...]
    parameter_losses: torch.Tensor
    density_law: str

    @property
    def physical_log_prob(self) -> torch.Tensor:
        """Unweighted full joint score; training role balance is applied separately."""
        return -self.parameter_losses.sum(-1)


def score_teacher_forced_episode(network, episode: PreparedTeacherForcedEpisode, *, law: str) -> TeacherForcedScore:
    """Score every known generating coordinate through the real eight-member law.

    Stock fitted API borders already undo target normalization. Subtracting each
    coordinate's decoder log-Jacobian then produces physical conditional density;
    the full joint Jacobian is never added separately to every scalar/group.
    """
    if law not in (LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
        raise ModelQualificationError("a concrete ensemble law is required; no finite fallback")
    if episode.density_law != law:
        raise ModelQualificationError("prepared/scored density-law mismatch")
    if episode.split == "test":
        raise ModelQualificationError("test cases cannot enter training/quick-validation scoring")
    if len(episode.stages) != len(episode.parameter_order) or len(episode.roles) != len(episode.stages):
        raise ModelQualificationError("incomplete ordered teacher-forced episode")
    losses = []
    for j, stage in enumerate(episode.stages):
        if stage.index != j or stage.parameter_name != episode.parameter_order[j] or stage.role != episode.roles[j]:
            raise ModelQualificationError("teacher-forced parameter order/role mismatch")
        density = stage.ensemble.density(network, law=law)
        loss = -density.log_prob(stage.targets, log_abs_det_decode=stage.decoder_log_jac)
        if loss.shape != (episode.query_count,) or not bool(torch.isfinite(loss).all()):
            raise TargetSupportMismatch(f"{episode.problem_id}:{stage.parameter_name}: generating label has nonfinite/zero sampling density; no repair")
        losses.append(loss)
    return TeacherForcedScore(episode.problem_id, episode.problem_fingerprint, episode.split,
        episode.roles, torch.stack(losses, dim=-1), law)


def train_teacher_forced_update(model: nn.Module, optimizer: torch.optim.Optimizer, scores: Sequence[TeacherForcedScore], state: TrainingState, *, gradient_clip: float = 1.0, allow_diagnostic_cpu: bool = False) -> dict:
    """Apply one explicit update with split provenance checked before backward.

    Validation scores may select checkpoints but never update weights. Diagnostic
    fixtures require an explicit CPU-only unit-step opt-in. This primitive does
    not authorize a production campaign or override pending I07 qualification.
    """
    if not scores:
        raise ValueError("nonempty training-episode scores are required")
    losses, roles = [], []
    for score in scores:
        diagnostic = score.split == "diagnostic" and allow_diagnostic_cpu
        if score.split != "train" and not diagnostic:
            raise ModelQualificationError("validation/test episodes cannot update transformer weights")
        if diagnostic and (score.parameter_losses.device.type != "cpu" or any(p.device.type != "cpu" for p in model.parameters())):
            raise ModelQualificationError("diagnostic optimizer fixtures are CPU-only")
        if score.density_law not in (LEGACY_FINITE, CANONICAL_FULL_SUPPORT):
            raise ModelQualificationError("an explicit implemented training density law is required")
        if score.parameter_losses.ndim != 2 or score.parameter_losses.shape[1] != len(score.roles):
            raise ValueError("one loss per query and declared parameter required")
        losses.extend(score.parameter_losses.unbind(0))
        roles.extend([score.roles] * len(score.parameter_losses))
    return train_update(model, optimizer, losses, roles, state, gradient_clip=gradient_clip)
