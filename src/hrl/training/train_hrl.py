"""
`StagedHRLConfig`、`train_staged_hrl_worker()` :固定/启发式 Manager 下训练 PPO Worker     
`ManagerPPOConfig`、`train_manager_ppo()` : 冻结 Worker 后训练离散 PPO Manager 
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from hrl.envs import HierarchicalRSMAEnvConfig, ManagerTrainingEnv, WorkerTrainingEnv
from hrl.envs.worker_training_env import PartitionManager

__all__ = [
    "ManagerPPOConfig",
    "ManagerPPOTrainingResult",
    "StagedHRLConfig",
    "StagedHRLTrainingResult",
    "train_manager_ppo",
    "train_staged_hrl_worker",
]


@dataclass(frozen=True, slots=True)
class StagedHRLConfig:

    env: HierarchicalRSMAEnvConfig = field(default_factory=HierarchicalRSMAEnvConfig)
    manager: PartitionManager | None = None
    seed: int = 42
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    policy_net_arch: tuple[int, ...] = (256, 256)
    checkpoint_directory: Path = Path("outputs/checkpoints")
    log_directory: Path = Path("outputs/logs")

    def __post_init__(self) -> None:
        if not isinstance(self.env, HierarchicalRSMAEnvConfig):
            raise TypeError("env must be a HierarchicalRSMAEnvConfig instance.")
        for name in ("seed", "total_timesteps", "n_steps", "batch_size", "n_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or (name != "seed" and value < 1):
                raise ValueError(f"{name} must be an integer with positive training values.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps and not exceed it.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not self.policy_net_arch or any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in self.policy_net_arch):
            raise ValueError("policy_net_arch must contain positive layer widths.")


@dataclass(frozen=True, slots=True)
class ManagerPPOConfig:

    worker: Any
    env: HierarchicalRSMAEnvConfig = field(default_factory=HierarchicalRSMAEnvConfig)
    seed: int = 42
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    policy_net_arch: tuple[int, ...] = (256, 256)
    checkpoint_directory: Path = Path("outputs/checkpoints")
    log_directory: Path = Path("outputs/logs")

    def __post_init__(self) -> None:
        if not hasattr(self.worker, "predict"):
            raise TypeError("worker must provide predict(observation, deterministic=True).")
        staged = StagedHRLConfig(
            env=self.env,
            seed=self.seed,
            total_timesteps=self.total_timesteps,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            policy_net_arch=self.policy_net_arch,
            checkpoint_directory=self.checkpoint_directory,
            log_directory=self.log_directory,
        )
        del staged


@dataclass(frozen=True, slots=True)
class ManagerPPOTrainingResult:
    """返回已训练的Manager PPO模式和上存棂欅路徑。"""

    model: Any
    model_path: Path
    total_timesteps: int
    seed: int


def train_manager_ppo(config: ManagerPPOConfig) -> ManagerPPOTrainingResult:
    """在Worker策略加上鎖时训练Manager PPO策略。"""
    if not isinstance(config, ManagerPPOConfig):
        raise TypeError("config must be a ManagerPPOConfig instance.")
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError("需要Stable-Baselines3便Manager PPO训练。") from error
    np.random.seed(config.seed)
    environment = ManagerTrainingEnv(config.worker, config.env)
    environment.reset(seed=config.seed)
    checkpoint_directory = Path(config.checkpoint_directory)
    log_directory = Path(config.log_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    model = PPO(
        "MlpPolicy", environment, learning_rate=config.learning_rate, n_steps=config.n_steps,
        batch_size=config.batch_size, n_epochs=config.n_epochs,
        policy_kwargs={"net_arch": list(config.policy_net_arch)}, seed=config.seed,
        tensorboard_log=str(log_directory), device="cpu", verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    model_path = checkpoint_directory / "staged_hrl_manager_ppo"
    model.save(str(model_path))
    environment.close()
    return ManagerPPOTrainingResult(
        model, model_path.with_suffix(".zip"), config.total_timesteps, config.seed
    )


@dataclass(frozen=True, slots=True)
class StagedHRLTrainingResult:
    """返回已训练的Worker PPO模式和上存模式位置。"""

    model: Any
    model_path: Path
    total_timesteps: int
    seed: int


def train_staged_hrl_worker(config: StagedHRLConfig = StagedHRLConfig()) -> StagedHRLTrainingResult:
    """完善Worker PPO预误或启发式Manager提供分区。"""
    if not isinstance(config, StagedHRLConfig):
        raise TypeError("config must be a StagedHRLConfig instance.")
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError("需要Stable-Baselines3便戆阶段HRL训练。") from error
    np.random.seed(config.seed)
    environment = WorkerTrainingEnv(config.env, config.manager)
    environment.reset(seed=config.seed)
    checkpoint_directory = Path(config.checkpoint_directory)
    log_directory = Path(config.log_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)
    model = PPO(
        "MlpPolicy", environment, learning_rate=config.learning_rate, n_steps=config.n_steps,
        batch_size=config.batch_size, n_epochs=config.n_epochs,
        policy_kwargs={"net_arch": list(config.policy_net_arch)}, seed=config.seed,
        tensorboard_log=str(log_directory), device="cpu", verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    model_path = checkpoint_directory / "staged_hrl_worker_ppo"
    model.save(str(model_path))
    environment.close()
    return StagedHRLTrainingResult(model, model_path.with_suffix(".zip"), config.total_timesteps, config.seed)
