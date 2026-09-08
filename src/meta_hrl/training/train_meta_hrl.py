# 为Meta强化学习实验训练任务循环、上下文条件的PPO工作者。
"""
`MetaPPOConfig`、`train_meta_ppo()` :在 task ID 之间按 episode 轮换训练 context-conditioned PPO Worker 
`train_meta_hrl.py` `TaskCyclingContextualWorkerEnv`: 为 SB3 提供任务轮换的 Gymnasium 环境 
Stable-Baselines3 PPO训练涵盖可重现Meta强化学习任务分布。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from meta_hrl.agents.learned_context import ContextConditionedFeaturesExtractor
from meta_hrl.agents.meta_context import MetaContextEncoder
from meta_hrl.agents.replay_buffer import TaskContextBuffer
from meta_hrl.envs import ContextualWorkerTrainingEnv, HierarchicalRSMAEnvConfig
from meta_hrl.envs.worker_training_env import PartitionManager

__all__ = [
    "MetaPPOConfig",
    "MetaPPOTrainingResult",
    "TaskCyclingContextualWorkerEnv",
    "train_meta_ppo",
]


@dataclass(frozen=True, slots=True)
class MetaPPOConfig:
    """在有限可重现任务集上配置上下文条件PPO。"""

    env: HierarchicalRSMAEnvConfig = field(default_factory=HierarchicalRSMAEnvConfig)
    manager: PartitionManager | None = None
    split: str = "train"
    task_ids: tuple[int, ...] = tuple(range(16))
    task_seed: int = 42
    seed: int = 42
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    policy_net_arch: tuple[int, ...] = (256, 256)
    use_learned_context_encoder: bool = True
    state_feature_dim: int = 128
    context_feature_dim: int = 32
    context_capacity_per_task: int = 64
    checkpoint_directory: Path = Path("outputs/checkpoints")
    log_directory: Path = Path("outputs/logs")

    def __post_init__(self) -> None:
        if not isinstance(self.env, HierarchicalRSMAEnvConfig):
            raise TypeError("env must be a HierarchicalRSMAEnvConfig instance.")
        if self.split not in {"train", "validation", "ood"}:
            raise ValueError("split must be 'train', 'validation', or 'ood'.")
        if not self.task_ids:
            raise ValueError("task_ids must contain at least one task ID.")
        if any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in self.task_ids):
            raise TypeError("task_ids must contain integers.")
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("task_ids must be unique.")
        for name in (
            "task_seed", "seed", "total_timesteps", "n_steps", "batch_size", "n_epochs",
            "context_capacity_per_task", "state_feature_dim", "context_feature_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or (name not in {"task_seed", "seed"} and value < 1):
                raise ValueError(f"{name} must be an integer with positive training values.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps and not exceed it.")
        for name in ("learning_rate", "gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be within (0, 1].")
        if not isinstance(self.use_learned_context_encoder, bool):
            raise TypeError("use_learned_context_encoder must be a boolean.")
        if not self.policy_net_arch or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1
            for width in self.policy_net_arch
        ):
            raise ValueError("policy_net_arch must contain positive layer widths.")


class TaskCyclingContextualWorkerEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """在情节边界处循环任务ID，同时保持任务特殊上下文。"""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
        manager: PartitionManager | None = None,
        split: str = "train",
        task_ids: tuple[int, ...] = (0,),
        task_seed: int = 42,
        context_capacity_per_task: int = 64,
    ) -> None:
        if not task_ids:
            raise ValueError("task_ids must contain at least one task ID.")
        if any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in task_ids):
            raise TypeError("task_ids must contain integers.")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be unique.")
        if isinstance(task_seed, bool) or not isinstance(task_seed, int):
            raise TypeError("task_seed must be an integer.")
        self.task_ids = task_ids
        self.task_seed = task_seed
        self._episode_index = 0
        self.context_buffer = TaskContextBuffer(context_capacity_per_task)
        self.environment = ContextualWorkerTrainingEnv(
            config=config,
            manager=manager,
            split=split,
            context_buffer=self.context_buffer,
            context_encoder=MetaContextEncoder(),
        )
        self.action_space = self.environment.action_space
        self.observation_space = self.environment.observation_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """以固定任务分布和新的信道样本开始下一个任务情节。"""
        if options is not None:
            raise ValueError("TaskCyclingContextualWorkerEnv does not accept reset options.")
        task_id = self.task_ids[self._episode_index % len(self.task_ids)]
        episode_seed = (
            self.task_seed + self._episode_index
            if seed is None
            else int(seed) + self._episode_index
        )
        observation, info = self.environment.reset(
            seed=episode_seed,
            options={"task_id": task_id, "task_seed": self.task_seed},
        )
        self._episode_index += 1
        info["meta_episode_index"] = self._episode_index - 1
        return observation, info

    def step(
        self, action: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return self.environment.step(action)

    def close(self) -> None:
        self.environment.close()


@dataclass(frozen=True, slots=True)
class MetaPPOTrainingResult:
    """返回已训练的上下文条件策略和可重现工件。"""

    model: Any
    model_path: Path
    total_timesteps: int
    seed: int
    task_ids: tuple[int, ...]
    split: str


def _load_ppo() -> Any:
    """仅当计划训练时导入可选PPO依赖。"""
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError("需要Stable-Baselines3验Meta-PPO训练。") from error
    return PPO


def train_meta_ppo(config: MetaPPOConfig = MetaPPOConfig()) -> MetaPPOTrainingResult:
    """训练一个MLP PPO工作者，以最近的任务本地转移为context条件。"""
    if not isinstance(config, MetaPPOConfig):
        raise TypeError("config must be a MetaPPOConfig instance.")
    PPO = _load_ppo()
    np.random.seed(config.seed)
    environment = TaskCyclingContextualWorkerEnv(
        config=config.env,
        manager=config.manager,
        split=config.split,
        task_ids=config.task_ids,
        task_seed=config.task_seed,
        context_capacity_per_task=config.context_capacity_per_task,
    )
    environment.reset(seed=config.seed)
    checkpoint_directory = Path(config.checkpoint_directory)
    log_directory = Path(config.log_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    policy_kwargs: dict[str, Any] = {"net_arch": list(config.policy_net_arch)}
    if config.use_learned_context_encoder:
        policy_kwargs.update(
            {
                "features_extractor_class": ContextConditionedFeaturesExtractor,
                "features_extractor_kwargs": {
                    "state_feature_dim": config.state_feature_dim,
                    "context_feature_dim": config.context_feature_dim,
                },
            }
        )
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        policy_kwargs=policy_kwargs,
        seed=config.seed,
        tensorboard_log=str(log_directory),
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    model_path = checkpoint_directory / "meta_context_worker_ppo"
    model.save(str(model_path))
    environment.close()
    return MetaPPOTrainingResult(
        model=model,
        model_path=model_path.with_suffix(".zip"),
        total_timesteps=config.total_timesteps,
        seed=config.seed,
        task_ids=config.task_ids,
        split=config.split,
    )
