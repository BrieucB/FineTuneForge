"""Reproducible synthetic setup definitions, never production simulation banks.

Control grids depend on declared prior reference scales, never query truths.
The structural signature is a label-invariant color-refinement digest: equal
signatures are conservatively kept out of structural-holdout training, even
when a rare non-isomorphic collision would merely exclude extra configurations.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from functools import lru_cache
import json
import math

import numpy as np

from .spec import DatasetSpec, FamilySpec, ParameterSpec, PriorSpec, ProblemSpec, TRAIN_PHYSICS, TEST_PHYSICS

MASTER_SEED = 20261117
CONTEXT_SIZES = (16, 64, 256, 1024, 4096)
SIZE_BANDS = {"small": (1, 32), "medium": (33, 64), "large": (65, 96)}
TEST_GROUPS = ("familiar", "upper_sizes", "sharing", "heldout_physics", "combined")
SHARING = ("distinct", "repeated", "cross", "batch", "overlap")
Z90 = 1.6448536269514722


def _rng(seed: int, split: str, index: int, role: int = 0):
    if type(seed) is not int or seed < 0 or type(index) is not int or index < 0:
        raise ValueError("catalogue seeds/indices must be nonnegative integers")
    split_code = {"train": 1, "validation": 2, "test": 3, "diagnostic": 4}[split]
    return np.random.default_rng(np.random.SeedSequence([seed, split_code, index, role]))


def _positive_prior(rng, centre):
    factor = float(rng.uniform(1.5, 3.0))
    log_sd = math.log(factor) / Z90
    mean = float(centre) * math.exp(0.5 * log_sd * log_sd)
    sd = mean * math.sqrt(math.expm1(log_sd * log_sd))
    return PriorSpec("lognormal", mean, sd)


def _root_priors(rng, name, positive, unit, batch):
    if positive:
        mean_prior = _positive_prior(rng, float(rng.uniform(0.5, 2.0)) * unit)
    else:
        mean_prior = PriorSpec("normal", float(rng.uniform(-1.0, 1.0)) * unit,
                               float(rng.uniform(0.5, 1.5)) * unit / Z90)
    roots = [ParameterSpec(name + ".mean", "population", mean_prior),
             ParameterSpec(name + ".within_sd", "population", _positive_prior(rng, float(rng.uniform(.05, .30)) * unit))]
    if batch:
        roots.append(ParameterSpec(name + ".between_sd", "population", _positive_prior(rng, float(rng.uniform(.05, .30)) * unit)))
    return roots


def _recipe(physics, rng, p=None):
    if physics == "sensor":
        size = int(rng.integers(1, 7)) if p is None else p
        slots = [(f"p{k}", False, 1.0) for k in range(size)]
        known = {}
    elif physics == "quadratic":
        slots = [("a", False, 1.), ("b", False, 1.), ("d", False, 1.)]
        known = {"a": 1., "b": .5, "d": 0.}
    elif physics == "scalar":
        slots = [("a", False, 1.)]; known = {"a": 1.}
    elif physics == "saturation":
        slots = [("d", False, 1.), ("A", True, 1.), ("K", True, 1.)]
        known = {"d": 0., "A": 1., "K": 1., "h": int(rng.integers(1, 3))}
    elif physics == "decay":
        slots = [("d", False, 1.), ("A_fast", True, 1.), ("tau_fast", True, 1.)]
        known = {"d": 0., "A_fast": 1., "tau_fast": 1.}
        if rng.random() < .5:
            slots += [("A_slow", True, 1.), ("delta_tau", True, 1.)]
            known.update(A_slow=.5, delta_tau=2.)
    elif physics == "oscillation":
        slots = [("d", False, 1.), ("A", False, 1.), ("B", False, 1.), ("gamma", True, .1), ("omega", True, 1.)]
        known = {"d": 0., "A": 1., "B": .5, "gamma": .1, "omega": 1., "v": 0.}
        if rng.random() < .5:
            slots += [("v", False, 1.)]
            known["v"] = .05
    elif physics == "spring":
        slots = [("k", True, 1.), ("c", True, 1.)]; known = {"k": 1., "c": .5}
    elif physics == "contact":
        slots = [("d", False, 1.), ("k", True, 1.), ("u0", True, 1.)]
        known = {"d": 0., "k": 1., "u0": 1.}
    else:
        raise ValueError("unknown physics recipe")
    if p is None:
        p = int(rng.integers(1, len(slots) + 1))
    if not 1 <= p <= len(slots):
        raise ValueError("unknown coefficient count is unsupported by the recipe")
    selected = sorted(rng.choice(len(slots), p, replace=False).tolist())
    unknown = [slots[j] for j in selected]
    if physics == "sensor":
        # Known sensor coefficients are retained rather than padding unknowns.
        known = {name: 1.0 for name, _, _ in slots}
    for name, _, _ in unknown:
        known.pop(name, None)
    return unknown, known, slots


def _settings(physics, coverage, rng, known, slots):
    broad = coverage == "broad"
    if physics == "scalar":
        return (), ()
    if physics == "sensor":
        # Ten channels, full-rank broad design; restricted settings suppress
        # all but one direction (all directions when P=1).
        h, r = np.linalg.qr(rng.normal(size=(10, len(slots))))
        h = h * np.where(np.diag(r) < 0, -1.0, 1.0)[None, :] * math.sqrt(10.0)
        if not broad:
            scales = np.full(len(slots), .05)
            if len(slots) > 1:
                scales[0] = 1.
            h *= scales
        return (), tuple(tuple(float(x) for x in row) for row in h)
    if physics == "quadratic":
        controls = np.linspace(-1., 1., 10) if broad else np.linspace(-.1, .1, 10)
    elif physics == "saturation":
        controls = np.r_[0., np.geomspace(.05, 5., 9)] if broad else np.linspace(0., .1, 10)
    elif physics == "decay":
        horizon = 15. if "A_slow" in known or any(s[0] == "A_slow" for s in slots) else 5.
        controls = np.r_[0., np.geomspace(.02, horizon, 9)] if broad else np.linspace(0., .1, 10)
    elif physics == "oscillation":
        fractions = np.array([0., .013, .041, .097, .173, .283, .431, .619, .811, 1.])
        controls = fractions * (4. * math.pi if broad else .1)
    elif physics == "spring":
        controls = np.linspace(0., 6., 10) if broad else np.linspace(0., .1, 10)
    elif physics == "contact":
        controls = np.linspace(0., 3., 10) if broad else np.linspace(0., .25, 10)
    else:
        raise ValueError("unknown measurement protocol")
    return tuple(float(x) for x in controls), ()


def structural_signature(spec: ProblemSpec) -> str:
    labels, edges = {}, {}
    def node(key, label):
        labels[key] = label; edges[key] = []
    def link(a, b, label):
        edges[a].append((label + ":out", b)); edges[b].append((label + ":in", a))
    for p in spec.parameters:
        node("p:" + p.name, "parameter:" + p.role + ":" + p.prior.distribution)
    for p in spec.parameters:
        for field in ("mean", "sd"):
            parent = getattr(p.prior, field)
            if isinstance(parent, str):
                link("p:" + parent, "p:" + p.name, field)
    for f in spec.families:
        node("f:" + f.id, "family:" + f.physics)
        link("p:" + f.noise_parameter, "f:" + f.id, "noise")
    for d in spec.datasets:
        node("d:" + d.id, "dataset")
        link("f:" + d.family_id, "d:" + d.id, "family")
        skey = "s:" + d.specimen_id
        if skey not in labels:
            node(skey, "specimen")
        link(skey, "d:" + d.id, "specimen")
        for coefficient, parameter in d.parameters.items():
            link("p:" + parameter, "d:" + d.id, "coefficient:" + coefficient)
    color = {k: sha256(v.encode()).hexdigest() for k, v in labels.items()}
    for _ in range(6):
        color = {k: sha256(json.dumps([color[k], sorted((label, color[n]) for label, n in edges[k])], separators=(",", ":")).encode()).hexdigest() for k in labels}
    return sha256(json.dumps(sorted(color.values()), separators=(",", ":")).encode()).hexdigest()


def _assemble(identifier, split, rng, physics, nis, sharing, coverage, recipes=None, matched_variant=None, batches=None):
    batch = sharing in ("batch", "overlap")
    if batches is None:
        batches = int(rng.integers(1, 4)) if batch else 0
    families, datasets, roots, batch_nodes, physical_nodes = [], [], [], [], []
    descriptors = []
    root_specs = {}
    bindings = {}
    groups = {}
    shared_pool = max(1, max(nis) if matched_variant is not None else math.ceil(max(nis) / 2))
    for fi, (kind, ni) in enumerate(zip(physics, nis)):
        fid = f"family-{fi}"
        unknown, known, slots = _recipe(kind, rng) if recipes is None else recipes[fi]
        families.append(FamilySpec(fid, kind, fid + ".noise_sd"))
        rank_by_support = {}
        for coefficient, positive, unit in unknown:
            support = (positive, unit)
            ordinal = rank_by_support.get(support, 0); rank_by_support[support] = ordinal + 1
            group = ("shared", positive, unit, ordinal) if sharing in ("cross", "overlap") else (fid, coefficient, positive, unit)
            if matched_variant is not None:
                group = ("shared", positive, unit, ordinal)
            if group not in groups:
                groups[group] = f"population-{len(groups)}"
                root_specs[group] = (positive, unit)
        for j in range(ni):
            if matched_variant is not None:
                specimen_index = j if fi == 0 or matched_variant == 0 else j % max(1, ni // 2)
                sid = f"shared-specimen-{specimen_index}"
            elif sharing in ("cross", "overlap"):
                specimen_index = j % shared_pool; sid = f"shared-specimen-{specimen_index}"
            elif sharing == "repeated":
                specimen_index = j % max(1, math.ceil(ni / 2)); sid = f"{fid}-specimen-{specimen_index}"
            else:
                specimen_index = j; sid = f"{fid}-specimen-{specimen_index}"
            did = f"{fid}-dataset-{j}"
            cov = coverage if coverage != "mixed" else ("broad" if (fi + j) % 2 == 0 else "restricted")
            if kind == "scalar":
                cov = "not_applicable"
            controls, matrix = _settings(kind, cov, rng, known, slots)
            parameters = {}
            ranks = {}
            for coefficient, positive, unit in unknown:
                support = (positive, unit); ordinal = ranks.get(support, 0); ranks[support] = ordinal + 1
                group = ("shared", positive, unit, ordinal) if sharing in ("cross", "overlap") or matched_variant is not None else (fid, coefficient, positive, unit)
                pname = groups[group] + "." + sid
                b = specimen_index % max(1, batches)
                bindings.setdefault(group, {})[pname] = b
                parameters[coefficient] = pname
            descriptors.append((did, fid, sid, parameters, controls, dict(known), matrix, cov))
    for group, name in groups.items():
        positive, unit = root_specs[group]
        roots.extend(_root_priors(rng, name, positive, unit, batch))
        active_batches = sorted(set(bindings[group].values())) if batch else []
        for b in active_batches:
            batch_nodes.append(ParameterSpec(f"{name}.batch-{b}.mean", "batch", PriorSpec("lognormal" if positive else "normal", name + ".mean", name + ".between_sd")))
        for pname, b in sorted(bindings[group].items()):
            parent = f"{name}.batch-{b}.mean" if batch else name + ".mean"
            physical_nodes.append(ParameterSpec(pname, "physical", PriorSpec("lognormal" if positive else "normal", parent, name + ".within_sd")))
    noises = [ParameterSpec(f.noise_parameter, "noise", _positive_prior(rng, float(rng.uniform(.02, .20)))) for f in families]
    for did, fid, sid, pars, controls, known, matrix, cov in descriptors:
        datasets.append(DatasetSpec(did, fid, sid, pars, controls, known, matrix))
    metadata = {"catalogue_version": "hierarchical-ft-v1", "sharing": sharing,
                "coverage": "not_applicable" if all(kind == "scalar" for kind in physics) else coverage, "dataset_coverage": {d[0]: d[7] for d in descriptors},
                "batch_count": len({b for members in bindings.values() for b in members.values()}) if batch else 0, "query_truth_used_for_controls": False,
                "property_reference_units": {groups[g]: root_specs[g][1] for g in groups},
                "response_reference_unit": 1., "population_correlation": "independent conditional populations; posterior dependence retained"}
    return ProblemSpec(identifier, split, tuple(roots + batch_nodes + physical_nodes + noises), tuple(families), tuple(datasets), metadata)


def matched_sharing_pair(pair_index: int, *, seed: int = MASTER_SEED, split="test"):
    """Same prior and dimensions; different actual cross-family specimen links."""
    if type(pair_index) is not int or not 0 <= pair_index < 4:
        raise ValueError("four matched sharing pairs are defined")
    ni, p, batches = [(4, 2, 0), (6, 3, 0), (8, 2, 2), (10, 3, 3)][pair_index]
    a = [(f"p{k}", False, 1.) for k in range(p)]
    b = [(name, False, 1.) for name in (("a", "d") if p == 2 else ("a", "d", "b"))]
    recipes = [(a, {}, a), (b, {"b": .5} if p == 2 else {}, b)]
    outputs = []
    for variant in (0, 1):
        rng = _rng(seed, split, pair_index, 970)
        spec = _assemble(f"{split}-sharing-pair-{pair_index}-{variant}", split, rng,
                         ["sensor", "quadratic"], [ni, ni], "overlap" if batches else "cross", "mixed",
                         recipes=recipes, matched_variant=variant, batches=batches)
        meta = dict(spec.metadata, test_group="sharing", matched_pair=pair_index, matched_variant=variant)
        outputs.append(replace(spec, metadata=meta))
    return tuple(outputs)


@lru_cache(maxsize=32)
def heldout_sharing_signatures(seed: int = MASTER_SEED):
    return frozenset(structural_signature(s) for i in range(4) for s in matched_sharing_pair(i, seed=seed))


def generate_setup(index: int, *, split: str = "train", size_band: str | None = None,
                   seed: int = MASTER_SEED, physics=None, sharing=None, test_group=None, required_physics=None,
                   max_attempts: int = 10000) -> ProblemSpec:
    """Generate a valid setup definition; rejection never inspects query truth."""
    if split not in ("train", "validation", "test", "diagnostic"):
        raise ValueError("invalid split")
    if size_band is not None and size_band not in SIZE_BANDS:
        raise ValueError("invalid size stratum")
    if sharing is not None and sharing not in SHARING:
        raise ValueError("unknown sharing pattern")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if split != "test" and physics and set(physics) & set(TEST_PHYSICS):
        raise ValueError("reserved physics cannot enter training or validation")
    if required_physics is not None and (required_physics not in TRAIN_PHYSICS or split == "test"):
        raise ValueError("required training-family anchor is invalid")
    if physics is not None and size_band is not None:
        maxima = {"sensor": 6, "quadratic": 3, "scalar": 1, "saturation": 3, "decay": 5, "oscillation": 6, "spring": 2, "contact": 3}
        if SIZE_BANDS[size_band][0] > 18 * sum(maxima[k] for k in physics) + len(physics):
            raise ValueError("requested physics cannot reach the unknown-count stratum")
    if physics is not None and len(physics) == 1 and sharing in ("cross", "overlap"):
        raise ValueError("cross-family sharing needs more than one family")
    rng = _rng(seed, split, index)
    reserved = heldout_sharing_signatures(seed) if split in ("train", "validation") else frozenset()
    for attempt in range(max_attempts):
        if physics is None:
            allowed = TEST_PHYSICS if test_group == "heldout_physics" else TRAIN_PHYSICS
            n = int(rng.integers(1, len(allowed) + 1))
            if required_physics is None:
                kinds = [str(x) for x in rng.choice(allowed, n, replace=False)]
            else:
                extras = [x for x in allowed if x != required_physics]
                kinds = [required_physics] + [str(x) for x in rng.choice(extras, n - 1, replace=False)]
            if test_group == "combined":
                nknown = int(rng.integers(1, 6))
                kinds = [str(x) for x in rng.choice(TRAIN_PHYSICS, nknown, replace=False)]
                kinds += [str(rng.choice(TEST_PHYSICS))]
        else:
            kinds = list(physics)
        if len(set(kinds)) != len(kinds):
            raise ValueError("families must have distinct physics types")
        nis = [int(rng.integers(1, 13)) for _ in kinds]
        pattern = str(rng.choice(SHARING)) if sharing is None else sharing
        if len(kinds) < 2 and pattern in ("cross", "overlap"):
            continue
        coverage = str(rng.choice(("broad", "restricted", "mixed")))
        try:
            spec = _assemble(f"{split}-{index:05d}", split, rng, kinds, nis, pattern, coverage)
        except ValueError as exc:
            if "bounds" in str(exc) or "450" in str(exc):
                continue
            raise
        if pattern in ("cross", "overlap"):
            sharing_uses = {}
            for d in spec.datasets:
                for parameter in d.parameters.values():
                    sharing_uses.setdefault(parameter, set()).add(d.family_id)
            if not any(len(users) > 1 for users in sharing_uses.values()):
                continue
        if size_band is not None:
            lo, hi = SIZE_BANDS[size_band]
            if not lo <= spec.dimension <= hi:
                continue
        if test_group == "upper_sizes" and not (spec.dimension >= 65 and (len(kinds) >= 4 or max(nis) >= 9 or max(len(d.parameters) for d in spec.datasets) >= 5)):
            continue
        if structural_signature(spec) in reserved:
            continue
        meta = dict(spec.metadata, size_band=next(k for k, (lo, hi) in SIZE_BANDS.items() if lo <= spec.dimension <= hi),
                    master_seed=seed, setup_index=index, definition_attempt=attempt,
                    test_group=test_group, generation_is_definition_only=True)
        return replace(spec, metadata=meta)
    raise RuntimeError("exhausted setup-definition attempts; no query data was generated")


def generate_catalogue(split: str = "train", count: int | None = None, seed: int = MASTER_SEED) -> tuple[ProblemSpec, ...]:
    defaults = {"train": 10000, "validation": 20, "test": 40}
    if split not in defaults:
        raise ValueError("catalogue split must be train, validation or test")
    if count is None:
        count = defaults[split]
    if type(count) is not int or count < 1 or (split == "test" and count != 40):
        raise ValueError("positive count required; sealed test catalogue has exactly 40 setup definitions")
    result = []
    appearance = {p: 0 for p in TRAIN_PHYSICS}
    for index in range(count):
        if split == "test":
            group = TEST_GROUPS[index // 8]
            if group == "sharing":
                pair = matched_sharing_pair((index % 8) // 2, seed=seed)
                result.append(pair[index % 2]); continue
            band = "large" if group == "upper_sizes" else ("small", "medium", "large")[index % 3]
            spec = generate_setup(index, split=split, size_band=band, seed=seed, test_group=group)
        else:
            # Uniform random family subsets are conditioned on the agreed size
            # band. A least-represented-family anchor balances family exposure.
            band = ("small", "medium", "large")[index % 3]
            anchor = min(appearance, key=lambda p: (appearance[p], TRAIN_PHYSICS.index(p)))
            spec = generate_setup(index, split=split, size_band=band, seed=seed, required_physics=anchor)
            for f in spec.families:
                appearance[f.physics] += 1
        result.append(spec)
    return tuple(result)
