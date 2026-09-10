"""Freeze a Worker and train a composite partition/service-mode PPO Manager."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .hierarchical_env import HierarchicalRSMAEnv, HierarchicalRSMAEnvConfig
from .worker_training_env import partition_membership_features

__all__ = ["ManagerTrainingEnv", "PredictiveWorker"]


class PredictiveWorker(Protocol):
    """Minimal deterministic prediction interface implemented by PPO Workers."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


class ManagerTrainingEnv(gym.Env[NDArray[np.float32], int]):
    """One Manager action rolls the frozen Worker for a control interval.

    ``action_space`` is ``Discrete(3 * num_candidate_partitions)``. The codec
    is exactly ``partition_index * 3 + mode_index`` with mode order
    ``all, near, far``. The tuple passed to the physical environment makes the
    interpretation unambiguous even for low-valued encoded actions.
    """

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
        self._last_service_mode: str | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Sample a task; the first Manager action selects partition and mode."""
        observation, info = self.hierarchical_env.reset(seed=seed, options=options)
        self._last_partition_index = None
        self._last_service_mode = None
        info["manager_action_required"] = True
        return self._augment(observation, None), info

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply a composite action and roll the frozen Worker for one interval."""
        candidate_index, service_mode = self._decode_manager_action(action)
        observation, manager_info = self.hierarchical_env.set_manager_action(
            (candidate_index, service_mode)
        )
        interval_reward = 0.0
        interval_steps = 0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = dict(manager_info)
        while (
            interval_steps < self.hierarchical_env.config.high_level_interval
            and not terminated
            and not truncated
        ):
            worker_observation = self._augment(observation, candidate_index)
            worker_action, _ = self.worker.predict(worker_observation, deterministic=True)
            observation, reward, terminated, truncated, step_info = self.hierarchical_env.step(worker_action)
            interval_reward += float(reward)
            interval_steps += 1
            last_info.update(step_info)
        self._last_partition_index = candidate_index
        self._last_service_mode = service_mode
        last_info.update(
            {
                "manager_action": int(action),
                "manager_candidate_index": candidate_index,
                "manager_service_mode": service_mode,
                "manager_service_mode_index": ("all", "near", "far").index(service_mode),
                "manager_interval_steps": interval_steps,
                "manager_interval_reward": interval_reward,
            }
        )
        return self._augment(observation, candidate_index), interval_reward, terminated, truncated, last_info

    def _decode_manager_action(self, action: int) -> tuple[int, str]:
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
            raise ValueError("action must be a valid encoded Manager action.")
        encoded = int(action)
        if not self.action_space.contains(encoded):
            raise ValueError("action must be a valid encoded Manager action.")
        decoder = getattr(self.hierarchical_env, "decode_manager_action", None)
        if callable(decoder):
            decoded = decoder(encoded)
            if isinstance(decoded, tuple) and len(decoded) == 2:
                return int(decoded[0]), str(decoded[1])
            if hasattr(decoded, "partition_index") and hasattr(decoded, "service_mode"):
                return int(decoded.partition_index), str(decoded.service_mode)
            raise TypeError("decode_manager_action() must return (partition_index, service_mode).")
        candidate_index, mode_index = divmod(encoded, 3)
        return candidate_index, ("all", "near", "far")[mode_index]

    def _active_partition(self, partition_index: int) -> tuple[tuple[int, ...], ...]:
        effective = getattr(self.hierarchical_env, "effective_partition", None)
        if effective is not None:
            return effective
        return self.hierarchical_env.candidates[partition_index]

    def _augment(
        self, observation: NDArray[np.float32], partition_index: int | None
    ) -> NDArray[np.float32]:
        if partition_index is None:
            partition_features = np.zeros(self._users * self._users, dtype=np.float32)
        else:
            partition_features = partition_membership_features(
                self._active_partition(partition_index), num_users=self._users
            )
        return np.concatenate((observation, partition_features)).astype(np.float32, copy=False)
