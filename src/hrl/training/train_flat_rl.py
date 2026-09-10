# 训练可重现Stable-Baselines3 PPO扁平资源分配基线。
"""训练单步环境上的连续动作 PPO"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from meta_hrl.envs import OneStepRSMAEnv, RSMAEnvConfig
from meta_hrl.training.evaluate import EvaluationMetrics, evaluate_flat_policy

__all__ = ["FlatPPOConfig", "FlatPPOTrainingResult", "train_flat_ppo"]


@dataclass(frozen=True, slots=True)
class FlatPPOConfig:
    """超参数和Flat PPO基线的输出位置。"""

    env: RSMAEnvConfig = field(default_factory=RSMAEnvConfig)
    seed: int = 42
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 1.0
    gae_lambda: float = 1.0
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    clip_range: float = 0.2
    policy_net_arch: tuple[int, ...] = (256, 256)
    checkpoint_directory: Path = Path("outputs/checkpoints")
    log_directory: Path = Path("outputs/logs")
    evaluation_directory: Path = Path("outputs/evaluations")
    evaluation_interval: int = 5_000
    evaluation_episodes: int = 100
    evaluation_seed: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.env, RSMAEnvConfig):
            raise TypeError("env must be an RSMAEnvConfig instance.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        for name in ("total_timesteps", "n_steps", "batch_size", "n_epochs", "evaluation_interval", "evaluation_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps and not exceed it.")
        for name in ("learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "clip_range"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be within [0, 1].")
        if not self.policy_net_arch or any(isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in self.policy_net_arch):
            raise ValueError("policy_net_arch must contain positive layer widths.")


@dataclass(frozen=True, slots=True)
class FlatPPOTrainingResult:
    """返回PPO运行的拟合模型及可重现工件。"""

    model: Any
    model_path: Path
    total_timesteps: int
    seed: int
    evaluation_csv_path: Path
    evaluation_svg_path: Path


def _write_evaluation_csv(records: list[dict[str, float]], path: Path) -> None:
    """Write deterministic policy-evaluation metrics in a portable format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _write_reward_curve_svg(records: list[dict[str, float]], path: Path) -> None:
    """Render the mean-reward curve without a plotting-library dependency."""
    width, height, margin = 800, 480, 70
    x_values = [record["timesteps"] for record in records]
    y_values = [record["mean_reward"] for record in records]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    plot_width, plot_height = width - 2 * margin, height - 2 * margin
    points = " ".join(
        f"{margin + (value_x - x_min) / (x_max - x_min) * plot_width:.2f},"
        f"{height - margin - (value_y - y_min) / (y_max - y_min) * plot_height:.2f}"
        for value_x, value_y in zip(x_values, y_values)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="18">Flat PPO evaluation reward</text>
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="black"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="black"/>
<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/>
<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif">Training timesteps</text>
<text x="20" y="{height / 2}" text-anchor="middle" font-family="sans-serif" transform="rotate(-90 20 {height / 2})">Mean reward</text>
<text x="{margin}" y="{height - margin + 22}" font-family="sans-serif" font-size="12">{x_min:.0f}</text>
<text x="{width - margin}" y="{height - margin + 22}" text-anchor="end" font-family="sans-serif" font-size="12">{x_max:.0f}</text>
<text x="{margin - 8}" y="{height - margin}" text-anchor="end" font-family="sans-serif" font-size="12">{y_min:.3f}</text>
<text x="{margin - 8}" y="{margin + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_max:.3f}</text>
</svg>''',
        encoding="utf-8",
    )


def _evaluation_record(timesteps: int, metrics: EvaluationMetrics) -> dict[str, float]:
    return {"timesteps": float(timesteps), "mean_reward": metrics.mean_reward, "mean_sum_rate": metrics.mean_sum_rate,
            "mean_min_user_rate": metrics.mean_min_user_rate, "mean_jain_fairness": metrics.mean_jain_fairness,
            "mean_total_qos_gap": metrics.mean_total_qos_gap, "qos_satisfaction_rate": metrics.qos_satisfaction_rate}


def _load_ppo() -> Any:
    """仅当PPO被调采时引入可选训练依赖。"""
    try:
        from stable_baselines3 import PPO
    except ImportError as error:
        raise ImportError(
            "需要Stable-Baselines3便PPO训练。请首先安装项目依赖。"
        ) from error
    return PPO


def _load_base_callback() -> Any:
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as error:
        raise ImportError("需要Stable-Baselines3便PPO训练。请首先安装项目依赖。") from error
    return BaseCallback


def train_flat_ppo(config: FlatPPOConfig = FlatPPOConfig()) -> FlatPPOTrainingResult:
    """训练并保存一个MLP PPO策略用于自主采样RSMA任务。"""
    if not isinstance(config, FlatPPOConfig):
        raise TypeError("config must be a FlatPPOConfig instance.")
    PPO = _load_ppo()
    BaseCallback = _load_base_callback()
    np.random.seed(config.seed)
    environment = OneStepRSMAEnv(config.env)
    environment.reset(seed=config.seed)
    checkpoint_directory = Path(config.checkpoint_directory)  # 模型保存目录
    log_directory = Path(config.log_directory)                # TensorBoard日志目录
    evaluation_directory = Path(config.evaluation_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    log_directory.mkdir(parents=True, exist_ok=True)  
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    evaluation_records: list[dict[str, float]] = []

    def evaluate(timesteps: int) -> None:
        metrics = evaluate_flat_policy(
            model,
            env_config=config.env,
            episodes=config.evaluation_episodes,
            seed=config.evaluation_seed,
        )
        evaluation_records.append(_evaluation_record(timesteps, metrics))

    class EvaluationCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_evaluation = config.evaluation_interval

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_evaluation:
                evaluate(self.num_timesteps)
                self.next_evaluation += config.evaluation_interval
            return True

    # ?使用 MLP 策略，网络架构为两个隐藏层
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,       # 每次更新前收集的步数（rollout 长度）
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,     # 每次更新时在缓冲区上训练的轮数
        gamma=config.gamma,           # 折扣因子
        gae_lambda=config.gae_lambda, # GAE 平滑参数
        ent_coef=config.ent_coef,     # 熵正则化系数，鼓励探索
        vf_coef=config.vf_coef,       # 值函数损失的系数
        clip_range=config.clip_range, 
        policy_kwargs={"net_arch": list(config.policy_net_arch)}, # 策略网络隐藏层架构
        seed=config.seed,
        tensorboard_log=str(log_directory),
        device="cpu",
        verbose=0,
    )
    evaluate(0)
    model.learn(total_timesteps=config.total_timesteps, callback=EvaluationCallback(), progress_bar=True)
    if evaluation_records[-1]["timesteps"] != float(model.num_timesteps):
        evaluate(model.num_timesteps)
    model_path = checkpoint_directory / "flat_ppo_rsma"  # training/outputs/checkpoints/flat_ppo_rsma.zip
    model.save(str(model_path))   
    environment.close()
    evaluation_csv_path = evaluation_directory / "flat_ppo_evaluation.csv"
    evaluation_svg_path = evaluation_directory / "flat_ppo_reward_curve.svg"
    _write_evaluation_csv(evaluation_records, evaluation_csv_path)
    _write_reward_curve_svg(evaluation_records, evaluation_svg_path)
    return FlatPPOTrainingResult(
        model=model,
        model_path=model_path.with_suffix(".zip"),
        total_timesteps=config.total_timesteps,
        seed=config.seed,
        evaluation_csv_path=evaluation_csv_path,
        evaluation_svg_path=evaluation_svg_path,
    )


def _main() -> int:
    """Command-line entry point for a reproducible Flat PPO run."""
    project_directory = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Train and save the Flat PPO RSMA baseline")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-directory", type=Path, default=project_directory / "outputs/checkpoints")
    parser.add_argument("--log-directory", type=Path, default=project_directory / "outputs/logs")
    parser.add_argument("--evaluation-directory", type=Path, default=project_directory / "outputs/evaluations")
    parser.add_argument("--evaluation-interval", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=10_000)
    arguments = parser.parse_args()
    result = train_flat_ppo(FlatPPOConfig(
        total_timesteps=arguments.timesteps,
        seed=arguments.seed,
        checkpoint_directory=arguments.checkpoint_directory,
        log_directory=arguments.log_directory,
        evaluation_directory=arguments.evaluation_directory,
        evaluation_interval=arguments.evaluation_interval,
        evaluation_episodes=arguments.evaluation_episodes,
        evaluation_seed=arguments.evaluation_seed,
    ))
    print(f"Saved model: {result.model_path.resolve()}")
    print(f"Saved evaluation data: {result.evaluation_csv_path.resolve()}")
    print(f"Saved reward curve: {result.evaluation_svg_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
