"""Deterministic role-stratified selection for the bounded CUDA pilot."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from random import Random
import re
from typing import Sequence


ROLE_GROUP = {"population": 0, "batch": 0, "physical": 1, "noise": 2}
ROLE_NAMES = ("population/batch", "physical", "noise")
HALF_CONDITIONAL_POLICY = "role_stratified_half_episode_sha256_shuffle.v1"
HALF_CONDITIONAL_PERCENT = 50


@dataclass(frozen=True)
class ConditionalSelection:
    request_id: str
    stage_indices: tuple[int, ...]
    role_counts: tuple[int, ...]
    selected_role_counts: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "stage_indices": list(self.stage_indices),
            "role_counts": dict(zip(ROLE_NAMES, self.role_counts, strict=True)),
            "selected_role_counts": dict(zip(ROLE_NAMES, self.selected_role_counts, strict=True)),
        }


def select_half_conditionals(roles: Sequence[str], request_id: str) -> ConditionalSelection:
    """Replay the validated benchmark's independent shuffle for each ordered role group."""
    if not isinstance(request_id, str) or re.fullmatch(r"[0-9a-f]{64}", request_id) is None:
        raise ValueError("conditional sampling requires an authenticated episode request identity")
    groups = [[] for _ in ROLE_NAMES]
    for index, role in enumerate(roles):
        try:
            groups[ROLE_GROUP[role]].append(index)
        except KeyError as exc:
            raise ValueError("unrecognized parameter role") from exc
    counts = tuple(map(len, groups))
    if not all(counts):
        raise ValueError("conditional sampling requires all three parameter role groups")

    kept = tuple((count + 1) // 2 for count in counts)
    selected = []
    for group, (indices, count) in enumerate(zip(groups, kept, strict=True)):
        seed = sha256(f"conditional-pilot-v1/{request_id}/{group}".encode("ascii")).digest()
        Random(seed).shuffle(indices)
        selected.extend(indices[:count])
    return ConditionalSelection(request_id, tuple(sorted(selected)), counts, kept)
