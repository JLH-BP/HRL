
"""评估 Worker/Manager 组合与切换信息"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from meta_hrl.envs import HierarchicalRSMAEnvConfig, WorkerTrainingEnv
from meta_hrl.envs.worker_training_env import PartitionManager


__all__ = ["StagedHRLEvaluationMetrics", "evaluate_staged_hrl_worker"]


class PredictiveWorker(Protocol):
    """Minimal deterministic prediction interface implemented by SB3 policies."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


@dataclass(frozen=True, slots=True)
class StagedHRLEvaluationMetrics:
    """Aggregate Worker outcomes and Manager control diagnostics."""

    mean_reward: float
    mean_sum_rate: float
    mean_min_user_rate: float
    mean_jain_fairness: float
    mean_total_qos_gap: float
    qos_satisfaction_rate: float
    episodes: int
    mean_partition_switches: float
    mean_manager_decisions: float


def evaluate_staged_hrl_worker(
    policy: PredictiveWorker,
    *,
    manager: PartitionManager,
    env_config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig(),
    episodes: int = 100,
    seed: int = 10_000,
) -> StagedHRLEvaluationMetrics:
    """Evaluate a Worker policy under one non-learning Manager baseline."""
    if not hasattr(policy, "predict"):
        raise TypeError("policy must provide predict(observation, deterministic=True).")
    if not hasattr(manager, "select"):
        raise TypeError("manager must provide select(scenario, candidates).")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    environment = WorkerTrainingEnv(env_config, manager)
    rewards: list[float] = []
    sum_rates: list[float] = []
    minimum_rates: list[float] = []
    fairnesses: list[float] = []
    gaps: list[float] = []
    switches: list[int] = []
    decisions: list[int] = []
    for offset in range(episodes):
        observation, _ = environment.reset(seed=seed + offset)
        terminated = False
        episode_reward = 0.0
        episode_sum_rates: list[float] = []
        episode_minimum_rates: list[float] = []
        episode_fairnesses: list[float] = []
        episode_gaps: list[float] = []
        episode_switches = 0
        episode_decisions = 1
        while not terminated:
            action, _ = policy.predict(observation, deterministic=True)
            observation, reward, terminated, _, info = environment.step(action)
            episode_reward += reward
            episode_sum_rates.append(float(info["sum_rate"]))
            episode_minimum_rates.append(float(np.min(info["user_rates"])))
            episode_fairnesses.append(float(info["jain_fairness"]))
            episode_gaps.append(float(np.sum(info["qos_gaps"])))
            episode_switches += int(info["partition_switched"])
            if "manager_candidate_index" in info:
                episode_decisions += 1
        rewards.append(episode_reward)
        sum_rates.append(float(np.mean(episode_sum_rates)))
        minimum_rates.append(float(np.mean(episode_minimum_rates)))
        fairnesses.append(float(np.mean(episode_fairnesses)))
        gaps.append(float(np.mean(episode_gaps)))
        switches.append(episode_switches)
        decisions.append(episode_decisions)
    environment.close()
    gap_array = np.asarray(gaps, dtype=np.float64)
    return StagedHRLEvaluationMetrics(
        mean_reward=float(np.mean(rewards)),
        mean_sum_rate=float(np.mean(sum_rates)),
        mean_min_user_rate=float(np.mean(minimum_rates)),
        mean_jain_fairness=float(np.mean(fairnesses)),
        mean_total_qos_gap=float(np.mean(gap_array)),
        qos_satisfaction_rate=float(np.mean(gap_array <= 1e-10)),
        episodes=episodes,
        mean_partition_switches=float(np.mean(switches)),
        mean_manager_decisions=float(np.mean(decisions)),
    )
