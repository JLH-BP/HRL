# Runs reproducible learned-context versus raw-context Meta-PPO benchmark experiments.
"""`learned/raw context 的配对多 seed 基准"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json

import numpy as np

from .evaluate_meta_hrl import MetaAdaptationCurve, evaluate_meta_adaptation
from .train_meta_hrl import MetaPPOConfig, train_meta_ppo

__all__ = [
    "MetaPPOBenchmarkConfig",
    "MetaPPOBenchmarkResult",
    "run_meta_ppo_benchmark",
]


@dataclass(frozen=True, slots=True)
class MetaPPOBenchmarkConfig:
    """Configure a paired learned-context ablation across seeds and task splits."""

    seeds: tuple[int, ...] = (11, 29, 47)
    training_task_ids: tuple[int, ...] = tuple(range(16))
    evaluation_task_ids: tuple[int, ...] = tuple(range(16))
    total_timesteps: int = 100_000
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    adaptation_episodes: tuple[int, ...] = (0, 1, 5, 10)
    evaluation_episodes: int = 5
    task_seed: int = 42
    output_directory: Path = Path("outputs/meta_ppo_benchmark")

    def __post_init__(self) -> None:
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in self.seeds):
            raise ValueError("seeds must contain at least one integer.")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique.")
        for name in ("training_task_ids", "evaluation_task_ids"):
            task_ids = getattr(self, name)
            if not task_ids or any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in task_ids):
                raise ValueError(f"{name} must contain at least one integer.")
            if len(set(task_ids)) != len(task_ids):
                raise ValueError(f"{name} must be unique.")
        for name in ("total_timesteps", "n_steps", "batch_size", "n_epochs", "evaluation_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps and not exceed it.")
        if not self.adaptation_episodes or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.adaptation_episodes
        ):
            raise ValueError("adaptation_episodes must contain nonnegative integers.")
        if tuple(sorted(self.adaptation_episodes)) != self.adaptation_episodes or len(set(self.adaptation_episodes)) != len(self.adaptation_episodes):
            raise ValueError("adaptation_episodes must be strictly increasing.")
        if isinstance(self.task_seed, bool) or not isinstance(self.task_seed, int):
            raise TypeError("task_seed must be an integer.")


@dataclass(frozen=True, slots=True)
class MetaPPOBenchmarkResult:
    """Serializable aggregate results for paired Meta-PPO policy variants."""

    config: MetaPPOBenchmarkConfig
    variants: dict[str, dict[str, dict[str, object]]]
    result_path: Path


def _curve_records(curves: list[MetaAdaptationCurve]) -> dict[str, object]:
    """Aggregate matching adaptation points across seeds into mean and standard deviation."""
    records: list[dict[str, float]] = []
    seed_records: list[list[dict[str, float]]] = []
    for point_index, budget in enumerate(curves[0].points):
        points = [curve.points[point_index] for curve in curves]
        record: dict[str, float] = {"adaptation_episodes": float(budget.adaptation_episodes)}
        per_seed: list[dict[str, float]] = []
        for point in points:
            per_seed.append(
                {
                    "mean_reward": point.mean_reward,
                    "mean_sum_rate": point.mean_sum_rate,
                    "mean_min_user_rate": point.mean_min_user_rate,
                    "mean_jain_fairness": point.mean_jain_fairness,
                    "mean_total_qos_gap": point.mean_total_qos_gap,
                    "qos_satisfaction_rate": point.qos_satisfaction_rate,
                    "mean_context_transitions": point.mean_context_transitions,
                }
            )
        seed_records.append(per_seed)
        for name in (
            "mean_reward", "mean_sum_rate", "mean_min_user_rate", "mean_jain_fairness",
            "mean_total_qos_gap", "qos_satisfaction_rate", "mean_context_transitions",
        ):
            values = np.asarray([getattr(point, name) for point in points], dtype=np.float64)
            record[f"{name}_mean"] = float(np.mean(values))
            record[f"{name}_std"] = float(np.std(values))
        records.append(record)
    return {"seeds": len(curves), "points": records, "seed_points": seed_records}


def run_meta_ppo_benchmark(
    config: MetaPPOBenchmarkConfig = MetaPPOBenchmarkConfig(),
    *,
    train: Callable[[MetaPPOConfig], object] = train_meta_ppo,
) -> MetaPPOBenchmarkResult:
    """Train paired policy variants and aggregate validation/OOD adaptation curves.

    Every variant uses the same per-seed task IDs and task distribution seed. The
    only ablation is whether PPO receives the learned state/context feature branch.
    """
    if not isinstance(config, MetaPPOBenchmarkConfig):
        raise TypeError("config must be a MetaPPOBenchmarkConfig instance.")
    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    variants: dict[str, dict[str, dict[str, object]]] = {}
    for name, use_learned_context_encoder in (("learned_context", True), ("raw_context", False)):
        curves: dict[str, list[MetaAdaptationCurve]] = {"validation": [], "ood": []}
        for seed in config.seeds:
            run_directory = output_directory / name / f"seed_{seed}"
            training = train(
                MetaPPOConfig(
                    seed=seed,
                    task_seed=config.task_seed,
                    task_ids=config.training_task_ids,
                    total_timesteps=config.total_timesteps,
                    n_steps=config.n_steps,
                    batch_size=config.batch_size,
                    n_epochs=config.n_epochs,
                    use_learned_context_encoder=use_learned_context_encoder,
                    checkpoint_directory=run_directory / "checkpoints",
                    log_directory=run_directory / "logs",
                )
            )
            for split in curves:
                curves[split].append(
                    evaluate_meta_adaptation(
                        training.model,
                        split=split,
                        task_ids=config.evaluation_task_ids,
                        adaptation_episodes=config.adaptation_episodes,
                        evaluation_episodes=config.evaluation_episodes,
                        task_seed=config.task_seed,
                        seed=seed + 100_000,
                    )
                )
        variants[name] = {split: _curve_records(values) for split, values in curves.items()}
    result_path = output_directory / "meta_ppo_benchmark.json"
    payload = {"config": asdict(config), "variants": variants}
    payload["config"]["output_directory"] = str(config.output_directory)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return MetaPPOBenchmarkResult(config=config, variants=variants, result_path=result_path)
