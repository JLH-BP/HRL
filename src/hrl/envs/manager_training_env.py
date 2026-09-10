"""冻结 Worker 后训练离散分区 Manager 的环境"""

from __future__ import annotations

from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .hierarchical_env import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from .worker_training_env import partition_membership_features

__all__ = ["ManagerTrainingEnv", "PredictiveWorker"]


class PredictiveWorker(Protocol):
    """最小确定性 Worker 预测接口，用于 Manager 环境。"""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


class ManagerTrainingEnv(gym.Env[NDArray[np.float32], int]):


    metadata = {"render_modes": []}

    def __init__(
        self,
        worker: PredictiveWorker,
        config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
    ) -> None:
        if not hasattr(worker, "predict"):
            raise TypeError("worker must provide predict(observation, deterministic=True).")
        self.worker = worker
        self.hierarchical_env = HierarchicalRSMAEnv(config)
        self.action_space = self.hierarchical_env.manager_action_space
        self._base_size = self.hierarchical_env.observation_space.shape[0]
        self._users = config.sampler.num_users
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._base_size + self._users * self._users,),
            dtype=np.float32,
        )
        self._last_partition_index: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Sample a task; the first Manager action selects its initial partition."""
        observation, info = self.hierarchical_env.reset(seed=seed, options=options)
        self._last_partition_index = None
        info["manager_action_required"] = True
        return self._augment(observation, None), info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply one partition and roll the frozen Worker for one control interval."""
        if not self.action_space.contains(action):
            raise ValueError("action must be a valid Manager candidate partition index.")
        observation, manager_info = self.hierarchical_env.set_manager_action(int(action))
        interval_reward = 0.0
        interval_steps = 0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = dict(manager_info)
        while interval_steps < self.hierarchical_env.config.high_level_interval and not terminated:
            worker_observation = self._augment(observation, int(action))
            worker_action, _ = self.worker.predict(worker_observation, deterministic=True)
            observation, reward, terminated, truncated, step_info = self.hierarchical_env.step(worker_action)
            interval_reward += float(reward)
            interval_steps += 1
            last_info.update(step_info)
            if truncated:
                break
        self._last_partition_index = int(action)
        last_info.update(
            {
                "manager_candidate_index": int(action),
                "manager_interval_steps": interval_steps,
                "manager_interval_reward": interval_reward,
            }
        )
        return self._augment(observation, int(action)), interval_reward, terminated, truncated, last_info

    def _augment(
        self, observation: NDArray[np.float32], partition_index: int | None
    ) -> NDArray[np.float32]:
        if partition_index is None:
            partition_features = np.zeros(self._users * self._users, dtype=np.float32)
        else:
            partition_features = partition_membership_features(
                self.hierarchical_env.candidates[partition_index], num_users=self._users
            )
        return np.concatenate((observation, partition_features)).astype(np.float32, copy=False)
