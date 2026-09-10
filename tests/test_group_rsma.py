# Tests group-common resource slots, bottleneck rates, and hierarchical timing.
import numpy as np
import pytest

from hrl.envs.hierarchical_env import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from hrl.envs.task_sampler import ScenarioSamplerConfig, TaskSampler
from hrl.grouping.candidate_groups import canonicalize_partition
from hrl.rsma.group_rsma import (
    active_group_slots,
    group_rsma_precoders,
    group_rsma_rates,
    project_group_action,
    service_masked_partition,
)


def test_group_slots_use_minimum_member_and_ignore_singletons() -> None:
    partition = canonicalize_partition(((0, 3), (1,), (2, 4, 5)), num_users=6)
    assert active_group_slots(partition, num_users=6) == {0: (0, 3), 2: (2, 4, 5)}


def test_group_common_rate_allocation_matches_each_bottleneck() -> None:
    sampler = TaskSampler()
    sampler.reset(seed=5)
    scenario = sampler.sample()
    partition = canonicalize_partition(((0, 1), (2,), (3,), (4,), (5,)), num_users=6)
    action = np.linspace(-1.0, 1.0, 18)
    initial = project_group_action(action, partition=partition, num_users=6, total_power=1.0)
    beams = group_rsma_precoders(scenario.channels, initial, partition)
    received = np.abs(scenario.channels @ beams) ** 2
    signal = received[[0, 1], 6]
    interference = np.sum(received[[0, 1], :6], axis=1) + np.sum(received[[0, 1], 6:], axis=1) - signal
    budgets = np.zeros(6)
    budgets[0] = np.min(np.log2(1.0 + signal / (interference + 1.0)))
    allocation = project_group_action(action, partition=partition, num_users=6, total_power=1.0, group_rate_budgets=budgets)
    rates = group_rsma_rates(scenario.channels, beams, allocation, partition, noise_variance=1.0)
    assert allocation.common_rate_allocations[[0, 1]].sum() == pytest.approx(rates.group_common_rates[0])


def test_hierarchical_environment_enforces_manager_clock() -> None:
    environment = HierarchicalRSMAEnv(HierarchicalRSMAEnvConfig(high_level_interval=2, episode_length=3))
    observation, info = environment.reset(seed=8)
    assert info["manager_action_required"]
    environment.set_manager_action(0)
    _, _, terminated, _, first = environment.step(np.zeros(18, dtype=np.float32))
    assert not terminated
    assert not first["manager_action_required"]
    _, _, _, _, second = environment.step(np.zeros(18, dtype=np.float32))
    assert second["manager_action_required"]
    assert observation.shape == environment.observation_space.shape


def test_service_mask_projects_unscheduled_resources_and_groups() -> None:
    partition = canonicalize_partition(((0, 1, 2), (3, 4), (5,)), num_users=6)
    mask = np.array([False, True, True, False, True, False], dtype=np.bool_)
    effective = service_masked_partition(partition, service_mask=mask, num_users=6)
    assert effective == canonicalize_partition(((0,), (1, 2), (3,), (4,), (5,)), num_users=6)
    assert active_group_slots(partition, num_users=6, service_mask=mask) == {1: (1, 2)}

    allocation = project_group_action(
        np.linspace(-1.0, 1.0, 18),
        partition=partition,
        num_users=6,
        total_power=1.0,
        group_rate_budgets=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        service_mask=mask,
    )
    assert allocation.private_powers[~mask].sum() == 0.0
    assert allocation.group_common_powers[[0, 2, 3, 4, 5]].sum() == 0.0
    assert allocation.common_rate_allocations[~mask].sum() == 0.0
    assert allocation.common_rate_allocations[[1, 2]].sum() == pytest.approx(1.0)


def test_hierarchical_environment_enforces_service_mode_and_episode_diagnostics() -> None:
    sampler = ScenarioSamplerConfig(near_field_user_count_range=(3, 3))
    environment = HierarchicalRSMAEnv(
        HierarchicalRSMAEnvConfig(sampler=sampler, high_level_interval=1, episode_length=2)
    )
    observation, _ = environment.reset(seed=31)
    encoded = environment.encode_manager_action(2, "near")
    assert environment.decode_manager_action(encoded) == (2, "near")
    observation, manager_info = environment.set_manager_action(encoded)
    mask = environment.scenario.near_field_mask
    np.testing.assert_array_equal(manager_info["service_mask"], mask)
    np.testing.assert_array_equal(observation[-6:], mask.astype(np.float32))

    next_observation, _, terminated, _, first = environment.step(
        np.zeros(18, dtype=np.float32)
    )
    assert not terminated
    assert first["service_mode"] == "near"
    assert first["allocation"].private_powers[~mask].sum() == 0.0
    assert first["allocation"].group_common_powers[~mask].sum() == 0.0
    assert first["allocation"].common_rate_allocations[~mask].sum() == 0.0
    assert first["user_rates"][~mask].sum() == 0.0
    np.testing.assert_allclose(
        first["cumulative_qos_gaps"],
        np.maximum(environment.scenario.qos_rate_targets - first["cumulative_user_rates"], 0.0),
    )
    # The base state ends in [last rates, cumulative rates, cumulative gaps,
    # service mask], making the history-dependent reward observable to PPO.
    np.testing.assert_allclose(next_observation[-24:-18], first["user_rates"])
    np.testing.assert_allclose(
        next_observation[-18:-12], first["cumulative_user_rates"]
    )
    np.testing.assert_allclose(next_observation[-12:-6], first["cumulative_qos_gaps"])
    np.testing.assert_allclose(next_observation[-6:], mask.astype(np.float32))

    environment.set_manager_action((2, "far"))
    _, _, terminated, _, second = environment.step(np.zeros(18, dtype=np.float32))
    assert terminated
    np.testing.assert_array_equal(second["service_counts"], np.ones(6, dtype=np.int64))
    np.testing.assert_allclose(
        second["service_fractions"], np.full(6, 0.5))
    np.testing.assert_allclose(
        second["cumulative_qos_gaps"],
        np.maximum(
            2.0 * environment.scenario.qos_rate_targets - second["cumulative_user_rates"], 0.0
        ),
    )
    assert np.isfinite(second["episode_reward"])
    assert second["partition_switch_count"] == 0
    assert second["service_mode_switch_count"] == 1
