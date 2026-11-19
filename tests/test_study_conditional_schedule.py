import math

import pytest

from posteriorforge.study.conditional_schedule import plan_conditional_shards


def test_selectively_splits_the_slow_case_and_covers_every_original_stage():
    costs = ((3.0,) * 8, (2.0, 2.0), (4.0,), (1.0, 3.0))
    ranks = plan_conditional_shards(costs, 4)
    assert ranks == plan_conditional_shards(costs, 4)
    assert all(ranks)
    expected = [(case, index) for case, stages in enumerate(costs) for index in range(len(stages))]
    actual = sorted((case, index) for shard in ranks for case, stages in shard.items() for index in stages)
    assert actual == expected
    assert sum(0 in shard for shard in ranks) == 3
    assert all(sum(case in shard for shard in ranks) == 1 for case in range(1, 4))
    workload = [sum(costs[case][stage] for case, stages in shard.items() for stage in stages) for shard in ranks]
    assert max(workload) < sum(costs[0])


@pytest.mark.parametrize('costs', [((),), ((0.0,),), ((math.nan,),)])
def test_refuses_missing_or_invalid_conditional_costs(costs):
    with pytest.raises(ValueError, match='cost estimates'):
        plan_conditional_shards(costs, 1)


def test_distributes_subdivided_cases_to_more_workers_than_cases():
    cases = ((2.0,) * 8, (1.0,) * 6)
    ranks = plan_conditional_shards(cases, 8)
    assert len(ranks) == 8 and all(ranks)
    seen = sorted((case, index) for rank in ranks for case, stages in rank.items() for index in stages)
    assert seen == [(case, index) for case, stages in enumerate(cases) for index in range(len(stages))]
