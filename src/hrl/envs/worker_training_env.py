"""在固定或启发式 Manager 下训练 Worker 的包装环境"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hrl.agents.high_level_policy import FixedPartitionManager
from hrl.grouping.candidate_groups import UserPartition

from .hierarchical_env import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from .task_sampler import RSMAScenario

__all__ = ["PartitionManager", "WorkerTrainingEnv", "partition_membership_features"]


class PartitionManager(Protocol):
    """选择当前抽样场景的候选分区重置元。"""

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> int: ...


def partition_membership_features(partition: UserPartition, *, num_users: int) -> NDArray[np.float32]:
    """返回一个布尔矩阵，表示每个用户是否属于同一组。"""
    features = np.zeros((num_users, num_users), dtype=np.float32)
    for group in partition:
        indices = np.asarray(group, dtype=np.int64)
        features[np.ix_(indices, indices)] = 1.0
    return features.reshape(-1)


class WorkerTrainingEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
        manager: PartitionManager | None = None,
    ) -> None:
        self.hierarchical_env = HierarchicalRSMAEnv(config)
        self.manager = FixedPartitionManager() if manager is None else manager
        if not hasattr(self.manager, "select"):
            raise TypeError("manager must provide a select(scenario, candidates) method.")
        self.action_space = self.hierarchical_env.worker_action_space
        base_size = self.hierarchical_env.observation_space.shape[0]
        users = config.sampler.num_users
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_size + users * users,), dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        observation, info = self.hierarchical_env.reset(seed=seed, options=options)
        manager_index = self.manager.select(
            self.hierarchical_env.scenario, self.hierarchical_env.candidates
        )
        observation, manager_info = self.hierarchical_env.set_manager_action(manager_index)
        info.update(manager_info)
        info["manager_candidate_index"] = manager_index
        return self._augment(observation), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.hierarchical_env.step(action)
        if info["manager_action_required"]:
            manager_index = self.manager.select(
                self.hierarchical_env.scenario, self.hierarchical_env.candidates
            )
            observation, manager_info = self.hierarchical_env.set_manager_action(manager_index)
            info.update(manager_info)
            info["manager_candidate_index"] = manager_index
        return self._augment(observation), reward, terminated, truncated, info

    def _augment(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.concatenate(
            (
                observation,
                partition_membership_features(
                    self.hierarchical_env.partition,
                    num_users=self.hierarchical_env.config.sampler.num_users,
                ),
            )
        ).astype(np.float32, copy=False)
