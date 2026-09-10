
"""计算 validation/OOD 的 support-episode 适应曲线"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from meta_hrl.envs import ContextualWorkerTrainingEnv, HierarchicalRSMAEnvConfig
from meta_hrl.envs.worker_training_env import PartitionManager

__all__ = [
    "MetaAdaptationPoint",
    "MetaAdaptationCurve",
    "evaluate_meta_adaptation",
]


class PredictiveMetaWorker(Protocol):
    """Minimal deterministic prediction interface implemented by SB3 policies."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


@dataclass(frozen=True, slots=True)
class MetaAdaptationPoint:
    """Aggregate task performance after collecting a fixed support-episode budget."""

    adaptation_episodes: int
    mean_reward: float
    mean_sum_rate: float
    mean_min_user_rate: float
    mean_jain_fairness: float
    mean_total_qos_gap: float
    qos_satisfaction_rate: float
    mean_context_transitions: float


@dataclass(frozen=True, slots=True)
class MetaAdaptationCurve:
    """A reproducible pre-/post-context adaptation curve for one task split."""

    split: str
    task_ids: tuple[int, ...]
    task_seed: int
    evaluation_episodes: int
    points: tuple[MetaAdaptationPoint, ...]


def _validate_inputs(
    policy: PredictiveMetaWorker,
    split: str,
    task_ids: tuple[int, ...],
    adaptation_episodes: tuple[int, ...],
    evaluation_episodes: int,
    task_seed: int,
    seed: int,
) -> None:
    if not hasattr(policy, "predict"):
        raise TypeError("policy must provide predict(observation, deterministic=True).")
    if split not in {"validation", "ood"}:
        raise ValueError("split must be 'validation' or 'ood'.")
    if not task_ids:
        raise ValueError("task_ids must contain at least one task ID.")
    if any(isinstance(task_id, bool) or not isinstance(task_id, int) for task_id in task_ids):
        raise TypeError("task_ids must contain integers.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique.")
    if not adaptation_episodes:
        raise ValueError("adaptation_episodes must contain at least one support budget.")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in adaptation_episodes):
        raise ValueError("adaptation_episodes must contain nonnegative integers.")
    if tuple(sorted(adaptation_episodes)) != adaptation_episodes or len(set(adaptation_episodes)) != len(adaptation_episodes):
        raise ValueError("adaptation_episodes must be strictly increasing.")
    if isinstance(evaluation_episodes, bool) or not isinstance(evaluation_episodes, int) or evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be a positive integer.")
    if isinstance(task_seed, bool) or not isinstance(task_seed, int):
        raise TypeError("task_seed must be an integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")


def _rollout(
    environment: ContextualWorkerTrainingEnv,
    policy: PredictiveMetaWorker,
    *,
    task_id: int,
    task_seed: int,
    episode_seed: int,
) -> tuple[float, float, float, float, float, int]:
    """Run one full deterministic episode and return its physical-layer metrics."""
    observation, _ = environment.reset(
        seed=episode_seed,
        options={"task_id": task_id, "task_seed": task_seed},
    )
    total_reward = 0.0
    sum_rates: list[float] = []
    minimum_rates: list[float] = []
    fairnesses: list[float] = []
    gaps: list[float] = []
    terminated = False
    truncated = False
    context_transitions = 0
    while not terminated and not truncated:
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(action)
        total_reward += float(reward)
        sum_rates.append(float(info["sum_rate"]))
        minimum_rates.append(float(np.min(info["user_rates"])))
        fairnesses.append(float(info["jain_fairness"]))
        gaps.append(float(np.sum(info["qos_gaps"])))
        context_transitions = int(info["meta_context_size"])
    return (
        total_reward,
        float(np.mean(sum_rates)),
        float(np.mean(minimum_rates)),
        float(np.mean(fairnesses)),
        float(np.mean(gaps)),
        context_transitions,
    )


def _evaluate_point(
    policy: PredictiveMetaWorker,
    *,
    env_config: HierarchicalRSMAEnvConfig,
    manager: PartitionManager | None,
    split: str,
    task_ids: tuple[int, ...],
    task_seed: int,
    seed: int,
    adaptation_episodes: int,
    evaluation_episodes: int,
    ood_shift_scale: float,
) -> MetaAdaptationPoint:
    """Evaluate one independent support budget without cross-point context leakage."""
    rewards: list[float] = []
    sum_rates: list[float] = []
    minimum_rates: list[float] = []
    fairnesses: list[float] = []
    gaps: list[float] = []
    context_sizes: list[int] = []
    for task_offset, task_id in enumerate(task_ids):
        environment = ContextualWorkerTrainingEnv(
            env_config, manager, split=split, ood_shift_scale=ood_shift_scale
        )
        base_seed = seed + 10_000 * task_offset
        for support_index in range(adaptation_episodes):
            _rollout(
                environment,
                policy,
                task_id=task_id,
                task_seed=task_seed,
                episode_seed=base_seed + support_index,
            )
        for evaluation_index in range(evaluation_episodes):
            reward, sum_rate, minimum_rate, fairness, gap, context_size = _rollout(
                environment,
                policy,
                task_id=task_id,
                task_seed=task_seed,
                episode_seed=base_seed + 5_000 + evaluation_index,
            )
            rewards.append(reward)
            sum_rates.append(sum_rate)
            minimum_rates.append(minimum_rate)
            fairnesses.append(fairness)
            gaps.append(gap)
            context_sizes.append(context_size)
        environment.close()
    gaps_array = np.asarray(gaps, dtype=np.float64)
    return MetaAdaptationPoint(
        adaptation_episodes=adaptation_episodes,
        mean_reward=float(np.mean(rewards)),
        mean_sum_rate=float(np.mean(sum_rates)),
        mean_min_user_rate=float(np.mean(minimum_rates)),
        mean_jain_fairness=float(np.mean(fairnesses)),
        mean_total_qos_gap=float(np.mean(gaps_array)),
        qos_satisfaction_rate=float(np.mean(gaps_array <= 1e-10)),
        mean_context_transitions=float(np.mean(context_sizes)),
    )


def evaluate_meta_adaptation(
    policy: PredictiveMetaWorker,
    *,
    split: str,
    task_ids: tuple[int, ...] = tuple(range(16)),
    adaptation_episodes: tuple[int, ...] = (0, 1, 5, 10),
    evaluation_episodes: int = 1,
    env_config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
    manager: PartitionManager | None = None,
    task_seed: int = 42,
    seed: int = 10_000,
    ood_shift_scale: float = 1.0,
) -> MetaAdaptationCurve:
    """Measure validation or OOD performance before and after task-local context collection.

    Each curve point rebuilds its environment and task buffer, rolls out the stated
    number of support episodes, then evaluates on held-out fast-fading seeds. This
    prevents an earlier point or probe rollout from contaminating a later budget.
    """
    if not isinstance(env_config, HierarchicalRSMAEnvConfig):
        raise TypeError("env_config must be a HierarchicalRSMAEnvConfig instance.")
    if not np.isfinite(ood_shift_scale) or ood_shift_scale < 0.0:
        raise ValueError("ood_shift_scale must be finite and nonnegative.")
    _validate_inputs(
        policy, split, task_ids, adaptation_episodes, evaluation_episodes, task_seed, seed
    )
    points = tuple(
        _evaluate_point(
            policy,
            env_config=env_config,
            manager=manager,
            split=split,
            task_ids=task_ids,
            task_seed=task_seed,
            seed=seed,
            adaptation_episodes=count,
            evaluation_episodes=evaluation_episodes,
            ood_shift_scale=ood_shift_scale,
        )
        for count in adaptation_episodes
    )
    return MetaAdaptationCurve(
        split=split,
        task_ids=task_ids,
        task_seed=task_seed,
        evaluation_episodes=evaluation_episodes,
        points=points,
    )
