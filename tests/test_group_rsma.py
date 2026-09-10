# Tests group-common resource slots, bottleneck rates, and hierarchical timing.
import numpy as np
import pytest

from hrl.envs.hierarchical_env import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from hrl.envs.task_sampler import TaskSampler
from hrl.grouping.candidate_groups import canonicalize_partition
from hrl.rsma.group_rsma import active_group_slots, group_rsma_precoders, group_rsma_rates, project_group_action


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
