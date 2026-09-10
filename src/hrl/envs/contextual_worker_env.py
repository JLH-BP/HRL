# 添加任务条件上下文转移到预置群RSMA Worker观测。
"""将按 task ID 隔离的 transition context 追加给 Worker 观察"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from meta_hrl.agents.meta_context import MetaContextEncoder
from meta_hrl.agents.replay_buffer import TaskContextBuffer
from meta_hrl.envs.hierarchical_env import HierarchicalRSMAEnvConfig
from meta_hrl.envs.meta_task_sampler import MetaTaskSampler

from .worker_training_env import PartitionManager, WorkerTrainingEnv

__all__ = ["ContextualWorkerTrainingEnv"]


class ContextualWorkerTrainingEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """采样Meta任务并追加任务特殊最近转移上下文。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
        manager: PartitionManager | None = None,
        meta_task_sampler: MetaTaskSampler | None = None,
        context_buffer: TaskContextBuffer | None = None,
        context_encoder: MetaContextEncoder | None = None,
        split: str = "train",
        ood_shift_scale: float = 1.0,
    ) -> None:
        if split not in {"train", "validation", "ood"}:
            raise ValueError("split must be 'train', 'validation', or 'ood'.")
        self.config = config
        self.manager = manager
        self.meta_task_sampler = (
            MetaTaskSampler(config.sampler, ood_shift_scale) if meta_task_sampler is None else meta_task_sampler
        )
        self.context_buffer = TaskContextBuffer() if context_buffer is None else context_buffer
        self.context_encoder = MetaContextEncoder() if context_encoder is None else context_encoder
        self.split = split
        self._environment: WorkerTrainingEnv | None = None
        self._task_id: int | None = None
        self._task_seed: int | None = None
        self._last_worker_observation: NDArray[np.float32] | None = None
        prototype = WorkerTrainingEnv(config, manager)
        base_size = prototype.observation_space.shape[0]
        self.action_space = prototype.action_space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_size + self.context_encoder.context_size,), dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """重置环境并返回初始观察值和信息。"""
        options = {} if options is None else options
        if not isinstance(options, dict):
            raise TypeError("options must be a dictionary or None.")
        task_id = int(options.get("task_id", 0))
        task_seed = int(options.get("task_seed", seed if seed is not None else 0))
        episode_seed = int(seed if seed is not None else task_seed)
        task = self.meta_task_sampler.sample(split=self.split, task_id=task_id, seed=task_seed)
        task_config = replace(
            self.config, sampler=task.sampler_config, noise_variance=task.noise_variance
        )
        self._environment = WorkerTrainingEnv(task_config, self.manager)
        observation, info = self._environment.reset(seed=episode_seed)
        self._task_id = task.task_id
        self._task_seed = task_seed
        self._last_worker_observation = observation.copy()
        info.update({"meta_task_id": task.task_id, "meta_split": task.split})
        return self._augment(observation), info

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """执行一个动作并返回增强的观察值、奖励、终止标志、截断标志和信息字典。"""
        if self._environment is None or self._task_id is None or self._last_worker_observation is None:
            raise RuntimeError("reset() must be called before step().")
        augmented_before = self._augment(self._last_worker_observation)
        next_observation, reward, terminated, truncated, info = self._environment.step(action)
        self._last_worker_observation = next_observation.copy()
        self.context_buffer.add(
            task_id=self._task_id, observation=augmented_before, action=action, reward=reward,
            next_observation=self._augment(next_observation), terminated=terminated or truncated,
        )
        augmented_after = self._augment(next_observation)
        info.update({"meta_task_id": self._task_id, "meta_context_size": len(self.context_buffer.recent(self._task_id))})
        return self._augment(next_observation), reward, terminated, truncated, info

    def _augment(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._task_id is None:
            context = np.zeros(self.context_encoder.context_size, dtype=np.float32)
        else:
            context = self.context_encoder.encode(self.context_buffer.recent(self._task_id))
        return np.concatenate((observation, context)).astype(np.float32, copy=False)
