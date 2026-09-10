
""""""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
import json

import numpy as np

from .evaluate_meta_hrl import MetaAdaptationCurve, evaluate_meta_adaptation
from .train_meta_hrl import MetaPPOConfig, train_meta_ppo

__all__ = ["MetaPPOSensitivityConfig", "MetaPPOSensitivityResult", "run_meta_ppo_sensitivity"]


@dataclass(frozen=True, slots=True)
class MetaPPOSensitivityConfig:
    """Configure a reproducible three-factor learned-context sensitivity grid."""

    seeds: tuple[int, ...] = (11, 29, 47)
    training_task_ids: tuple[int, ...] = tuple(range(16))
    evaluation_task_ids: tuple[int, ...] = tuple(range(16))
    context_capacities: tuple[int, ...] = (16, 64, 128)
    context_feature_dims: tuple[int, ...] = (16, 32, 64)
    ood_shift_scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    total_timesteps: int = 100_000
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    adaptation_episodes: tuple[int, ...] = (0, 1, 5, 10)
    evaluation_episodes: int = 5
    task_seed: int = 42
    output_directory: Path = Path("outputs/meta_ppo_sensitivity")

    def __post_init__(self) -> None:
        for name in ("seeds", "context_capacities", "context_feature_dims"):
            values = getattr(self, name)
            if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
                raise ValueError(f"{name} must contain positive integers.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique.")
        for name in ("training_task_ids", "evaluation_task_ids"):
            values = getattr(self, name)
            if not values or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"{name} must contain nonnegative integers.")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must be unique.")
        if not self.ood_shift_scales or any(not np.isfinite(value) or value < 0.0 for value in self.ood_shift_scales):
            raise ValueError("ood_shift_scales must contain finite nonnegative values.")
        if len(set(self.ood_shift_scales)) != len(self.ood_shift_scales):
            raise ValueError("ood_shift_scales must be unique.")
        for name in ("total_timesteps", "n_steps", "batch_size", "n_epochs", "evaluation_episodes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size != 0:
            raise ValueError("batch_size must divide n_steps and not exceed it.")
        if not self.adaptation_episodes or tuple(sorted(self.adaptation_episodes)) != self.adaptation_episodes:
            raise ValueError("adaptation_episodes must be nonempty and increasing.")


@dataclass(frozen=True, slots=True)
class MetaPPOSensitivityResult:
    """Serialized aggregate OOD curves for all three-factor grid cells."""

    config: MetaPPOSensitivityConfig
    cells: list[dict[str, object]]
    result_path: Path


def _aggregate(curves: list[MetaAdaptationCurve]) -> tuple[list[dict[str, float]], list[list[dict[str, float]]]]:
    """Return aggregate and per-seed points for later factor-level inference."""
    results: list[dict[str, float]] = []
    seed_points: list[list[dict[str, float]]] = []
    metrics = (
        "mean_reward", "mean_sum_rate", "mean_min_user_rate", "mean_jain_fairness",
        "mean_total_qos_gap", "qos_satisfaction_rate",
    )
    for index, point in enumerate(curves[0].points):
        record: dict[str, float] = {"adaptation_episodes": float(point.adaptation_episodes)}
        per_seed = [{metric: float(getattr(curve.points[index], metric)) for metric in metrics} for curve in curves]
        seed_points.append(per_seed)
        for metric in metrics:
            values = np.asarray([item[metric] for item in per_seed], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = float(np.std(values))
        results.append(record)
    return results, seed_points


def run_meta_ppo_sensitivity(
    config: MetaPPOSensitivityConfig = MetaPPOSensitivityConfig(),
    *,
    train: Callable[[MetaPPOConfig], object] = train_meta_ppo,
) -> MetaPPOSensitivityResult:
    """Train one learned-context policy per architecture cell and test OOD severity."""
    if not isinstance(config, MetaPPOSensitivityConfig):
        raise TypeError("config must be a MetaPPOSensitivityConfig instance.")
    output = Path(config.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    cells: list[dict[str, object]] = []
    for capacity in config.context_capacities:
        for context_dim in config.context_feature_dims:
            models = []
            for seed in config.seeds:
                run = output / f"capacity_{capacity}" / f"context_dim_{context_dim}" / f"seed_{seed}"
                models.append(train(MetaPPOConfig(
                    seed=seed, task_seed=config.task_seed, task_ids=config.training_task_ids,
                    total_timesteps=config.total_timesteps, n_steps=config.n_steps, batch_size=config.batch_size,
                    n_epochs=config.n_epochs, context_capacity_per_task=capacity,
                    context_feature_dim=context_dim, checkpoint_directory=run / "checkpoints", log_directory=run / "logs",
                )).model)
            for shift in config.ood_shift_scales:
                curves = [evaluate_meta_adaptation(
                    model, split="ood", task_ids=config.evaluation_task_ids,
                    adaptation_episodes=config.adaptation_episodes, evaluation_episodes=config.evaluation_episodes,
                    task_seed=config.task_seed, seed=seed + 100_000, ood_shift_scale=shift,
                ) for model, seed in zip(models, config.seeds)]
                points, seed_points = _aggregate(curves)
                cells.append({
                    "context_capacity": capacity, "context_feature_dim": context_dim,
                    "ood_shift_scale": shift, "seeds": len(curves), "points": points,
                    "seed_points": seed_points,
                })
    result_path = output / "meta_ppo_sensitivity.json"
    payload = {"config": asdict(config), "cells": cells}
    payload["config"]["output_directory"] = str(config.output_directory)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return MetaPPOSensitivityResult(config, cells, result_path)
