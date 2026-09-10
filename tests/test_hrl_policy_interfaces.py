# Tests Manager and Worker policy interfaces.
import numpy as np
import pytest

from hrl.agents.high_level_policy import (
    ExplicitFixedPartitionManager,
    NearFieldFirstSequentialManager,
    RandomCompositeActionManager,
    service_mask_for_mode,
    select_partition_from_scores,
)
from hrl.agents.low_level_policy import validate_worker_action
from hrl.envs.task_sampler import RSMAScenario
from hrl.grouping.candidate_groups import canonicalize_partition, enumerate_candidate_partitions


def test_manager_selection_is_deterministic_on_ties() -> None:
    candidates = enumerate_candidate_partitions(num_users=3)
    decision = select_partition_from_scores(np.ones(len(candidates)), candidates)
    assert decision.candidate_index == 0
    assert decision.partition == candidates[0]


def test_worker_action_validation_enforces_fixed_bounds() -> None:
    np.testing.assert_allclose(validate_worker_action(np.zeros(18), num_users=6, logit_bound=20.0), np.zeros(18))
    with pytest.raises(ValueError):
        validate_worker_action(np.zeros(17), num_users=6, logit_bound=20.0)
    with pytest.raises(ValueError):
        validate_worker_action(np.full(18, 21.0), num_users=6, logit_bound=20.0)


def _scenario_with_near_mask(near_mask: np.ndarray) -> RSMAScenario:
    users = near_mask.size
    return RSMAScenario(
        channels=np.ones((users, 2), dtype=np.complex128),
        ranges_m=np.ones(users, dtype=np.float64),
        angles_rad=np.zeros(users, dtype=np.float64),
        near_field_mask=near_mask,
        path_loss_linear=np.ones(users, dtype=np.float64),
        k_factors_linear=np.ones(users, dtype=np.float64),
        qos_rate_targets=np.zeros(users, dtype=np.float64),
    )


def test_near_field_first_manager_returns_the_expected_legal_partition() -> None:
    candidates = enumerate_candidate_partitions(num_users=6)
    scenario = _scenario_with_near_mask(
        np.array([True, False, True, False, False, True], dtype=np.bool_)
    )

    manager = NearFieldFirstSequentialManager()
    index = manager.select(scenario, candidates)

    assert index == manager.select(scenario, candidates)
    assert candidates[index] == canonicalize_partition(((0,), (2,), (5,), (1, 3, 4)), num_users=6)


@pytest.mark.parametrize(
    ("near_mask", "expected_partition"),
    [
        (np.ones(6, dtype=np.bool_), ((0,), (1,), (2,), (3,), (4,), (5,))),
        (np.zeros(6, dtype=np.bool_), ((0, 1, 2, 3, 4, 5),)),
    ],
)
def test_near_field_first_manager_handles_single_field_scenarios(
    near_mask: np.ndarray, expected_partition: tuple[tuple[int, ...], ...]
) -> None:
    candidates = enumerate_candidate_partitions(num_users=6)
    index = NearFieldFirstSequentialManager().select(_scenario_with_near_mask(near_mask), candidates)
    assert candidates[index] == expected_partition


def test_explicit_fixed_partition_manager_keeps_a_multiuser_grouping() -> None:
    candidates = enumerate_candidate_partitions(num_users=6)
    partition = ((0, 1), (2, 3), (4, 5))
    decision = ExplicitFixedPartitionManager(partition=partition).select(
        _scenario_with_near_mask(np.array([True, True, True, False, False, False], dtype=np.bool_)),
        candidates,
    )

    assert decision.partition == partition
    assert decision.service_mode == "all"
    assert candidates[decision.candidate_index] == partition


def test_near_first_manager_alternates_service_modes_after_reset() -> None:
    candidates = enumerate_candidate_partitions(num_users=6)
    scenario = _scenario_with_near_mask(
        np.array([True, False, True, False, False, True], dtype=np.bool_)
    )
    manager = NearFieldFirstSequentialManager()
    manager.reset(seed=7)
    near_decision = manager.select(scenario, candidates)
    far_decision = manager.select(scenario, candidates)

    assert near_decision.service_mode == "near"
    assert far_decision.service_mode == "far"
    np.testing.assert_array_equal(
        service_mask_for_mode(scenario, near_decision.service_mode), scenario.near_field_mask
    )
    np.testing.assert_array_equal(
        service_mask_for_mode(scenario, far_decision.service_mode), ~scenario.near_field_mask
    )


def test_random_composite_manager_is_seed_reproducible_and_legal() -> None:
    candidates = enumerate_candidate_partitions(num_users=6)
    scenario = _scenario_with_near_mask(
        np.array([True, False, True, False, False, True], dtype=np.bool_)
    )
    first = RandomCompositeActionManager(seed=23)
    second = RandomCompositeActionManager(seed=23)
    first.reset(seed=23)
    second.reset(seed=23)
    sequence_one = [first.select(scenario, candidates) for _ in range(5)]
    sequence_two = [second.select(scenario, candidates) for _ in range(5)]

    assert sequence_one == sequence_two
    for decision in sequence_one:
        assert candidates[decision.candidate_index] == decision.partition
        assert np.any(service_mask_for_mode(scenario, decision.service_mode))
