"""Deterministically assign known conditional costs to data-parallel workers."""
from __future__ import annotations

from bisect import bisect_left
from itertools import accumulate
import math
from typing import Sequence


def plan_conditional_shards(stage_seconds: Sequence[Sequence[float]], workers: int, *,
                            split_threshold: float = 1.10, case_setup_seconds: float = 1.0,
                            task_setup_seconds: float = 0.1) -> tuple[dict[int, tuple[int, ...]], ...]:
    """Schedule longest work first and prefer reusing a case on the same worker.

    The input is one cost vector per case in original conditional order. The
    output maps case positions to their original stage indices on each worker.
    The global objective can therefore use the full case and role counts.
    """
    if type(workers) is not int or not 1 <= workers <= len(stage_seconds):
        raise ValueError("the worker count must be between one and the number of cases")
    if (not math.isfinite(split_threshold) or split_threshold < 1
            or any(not math.isfinite(v) or v < 0 for v in (case_setup_seconds, task_setup_seconds))):
        raise ValueError("invalid scheduling overhead or split threshold")
    if any(len(row) == 0 or any(not math.isfinite(v) or v <= 0 for v in row) for row in stage_seconds):
        raise ValueError("every case needs finite positive conditional cost estimates")
    totals = [math.fsum(row) for row in stage_seconds]
    target = split_threshold * math.fsum(totals) / workers

    tasks = []
    for position, (stages, total) in enumerate(zip(stage_seconds, totals, strict=True)):
        parts = min(workers, len(stages), math.ceil(total / target))
        cumulative = list(accumulate(stages))
        cuts = [0]
        for piece in range(1, parts):
            next_cut = bisect_left(cumulative, total * piece / parts) + 1
            cuts.append(min(len(stages) - (parts - piece), max(cuts[-1] + 1, next_cut)))
        cuts.append(len(stages))
        for start, stop in zip(cuts, cuts[1:]):
            tasks.append((math.fsum(stages[start:stop]), position, start, stop))

    assignments: list[dict[int, list[int]]] = [{} for _ in range(workers)]
    loads = [0.0] * workers
    for cost, position, start, stop in sorted(tasks, key=lambda task: (-task[0], *task[1:])):
        finish = [load + cost + task_setup_seconds + case_setup_seconds * (position not in shard)
                  for load, shard in zip(loads, assignments, strict=True)]
        worker = min(range(workers), key=finish.__getitem__)
        assignments[worker].setdefault(position, []).extend(range(start, stop))
        loads[worker] = finish[worker]
    return tuple({position: tuple(sorted(indices)) for position, indices in sorted(shard.items())}
                 for shard in assignments)
