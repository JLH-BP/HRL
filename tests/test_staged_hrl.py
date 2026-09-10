# Tests staged Manager selection and the standard Worker training wrapper.
import numpy as np

from hrl.agents.high_level_policy import (
    FixedPartitionManager,
    HeuristicPartitionManager,
    NearFieldFirstSequentialManager,
)
from hrl.envs.worker_training_env import WorkerTrainingEnv, partition_membership_features
from hrl.grouping.candidate_groups import canonicalize_partition


def test_partition_membership_features_are_symmetric() -> None:
    partition = canonicalize_partition(((0, 2), (1,), (3, 4, 5)), num_users=6)
    matrix = partition_membership_features(partition, num_users=6).reshape(6, 6)
    np.testing.assert_array_equal(matrix, matrix.T)
    assert matrix[0, 2] == 1.0
    assert matrix[0, 1] == 0.0


def test_worker_wrapper_runs_standard_episode_with_fixed_manager() -> None:
    environment = WorkerTrainingEnv(manager=FixedPartitionManager(candidate_index=0))
    observation, info = environment.reset(seed=17)
    assert environment.observation_space.contains(observation)
    assert info["manager_candidate_index"] == 0
    next_observation, reward, _, _, _ = environment.step(np.zeros(18, dtype=np.float32))
    assert environment.observation_space.contains(next_observation)
    assert np.isfinite(reward)


def test_heuristic_manager_selects_configured_candidate() -> None:
    environment = WorkerTrainingEnv(manager=HeuristicPartitionManager())
    _, info = environment.reset(seed=29)
    assert environment.hierarchical_env.manager_action_space.contains(info["manager_candidate_index"])


def test_near_field_first_manager_runs_a_legal_worker_step() -> None:
    environment = WorkerTrainingEnv(manager=NearFieldFirstSequentialManager())
    observation, info = environment.reset(seed=41)
    assert environment.observation_space.contains(observation)
    manager_index = info["manager_candidate_index"]
    assert environment.hierarchical_env.manager_action_space.contains(manager_index)

    next_observation, reward, _, _, info = environment.step(np.zeros(18, dtype=np.float32))
    assert environment.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert info["partition"] == environment.hierarchical_env.candidates[manager_index]
