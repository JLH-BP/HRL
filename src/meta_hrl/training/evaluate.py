"""汇总 Flat PPO 的 reward、速率、公平性和 QoS 指标
`evaluate_physical_baselines()`在匹配场景上评估 RZF SDMA 与固定公共 RSMA"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from meta_hrl.envs import OneStepRSMAEnv, RSMAEnvConfig
from meta_hrl.rsma.baselines import equal_power_rzf, fixed_common_rsma

__all__ = ["EvaluationMetrics", "evaluate_flat_policy", "evaluate_physical_baselines"]


class PredictivePolicy(Protocol):
    """Minimal Stable-Baselines3-compatible deterministic policy interface."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate rates, fairness, and QoS outcomes over independent scenarios."""

    mean_reward: float
    mean_sum_rate: float
    mean_min_user_rate: float
    mean_jain_fairness: float
    mean_total_qos_gap: float
    qos_satisfaction_rate: float
    episodes: int


def _validate_episodes(episodes: int) -> int:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("episodes must be a positive integer.")
    return episodes


def _metrics(rewards: list[float], sum_rates: list[float], minimum_rates: list[float], fairnesses: list[float], qos_gaps: list[float]) -> EvaluationMetrics:
    gaps = np.asarray(qos_gaps, dtype=np.float64)
    return EvaluationMetrics(
        mean_reward=float(np.mean(rewards)),
        mean_sum_rate=float(np.mean(sum_rates)),
        mean_min_user_rate=float(np.mean(minimum_rates)),
        mean_jain_fairness=float(np.mean(fairnesses)),
        mean_total_qos_gap=float(np.mean(gaps)),
        qos_satisfaction_rate=float(np.mean(gaps <= 1e-10)),
        episodes=len(rewards),
    )


def evaluate_flat_policy(
    policy: PredictivePolicy, *, env_config: RSMAEnvConfig = RSMAEnvConfig(), episodes: int = 100, seed: int = 10_000
) -> EvaluationMetrics:
    """Evaluate one policy deterministically on a reproducible independent seed range."""
    if not isinstance(env_config, RSMAEnvConfig):
        raise TypeError("env_config must be an RSMAEnvConfig instance.")
    if not hasattr(policy, "predict"):
        raise TypeError("policy must provide a Stable-Baselines3-compatible predict method.")
    episodes = _validate_episodes(episodes)
    environment = OneStepRSMAEnv(env_config)
    rewards: list[float] = []
    sum_rates: list[float] = []
    minimum_rates: list[float] = []
    fairnesses: list[float] = []
    qos_gaps: list[float] = []
    for offset in range(episodes):
        observation, _ = environment.reset(seed=seed + offset)
        action, _ = policy.predict(observation, deterministic=True)
        _, reward, _, _, info = environment.step(action)
        rewards.append(reward)
        sum_rates.append(float(info["sum_rate"]))
        minimum_rates.append(float(np.min(info["user_rates"])))
        fairnesses.append(float(info["jain_fairness"]))
        qos_gaps.append(float(info["total_qos_gap"]))
    environment.close()
    return _metrics(rewards, sum_rates, minimum_rates, fairnesses, qos_gaps)


def evaluate_physical_baselines(
    *, env_config: RSMAEnvConfig = RSMAEnvConfig(), episodes: int = 100, seed: int = 10_000, common_power_fraction: float = 0.2
) -> dict[str, EvaluationMetrics]:
    """Evaluate equal-power RZF SDMA and fixed-common RZF RSMA on shared tasks."""
    if not isinstance(env_config, RSMAEnvConfig):
        raise TypeError("env_config must be an RSMAEnvConfig instance.")
    episodes = _validate_episodes(episodes)
    if not np.isfinite(common_power_fraction) or not 0.0 <= common_power_fraction <= 1.0:
        raise ValueError("common_power_fraction must be within [0, 1].")
    environment = OneStepRSMAEnv(env_config)
    records = {name: ([], [], [], [], []) for name in ("equal_power_rzf", "fixed_common_rzf_rsma")}
    for offset in range(episodes):
        environment.reset(seed=seed + offset)
        scenario = environment.scenario
        base_kwargs = {"total_power": env_config.total_power, "noise_variance": env_config.noise_variance}
        results = {
            "equal_power_rzf": equal_power_rzf(scenario.channels, **base_kwargs),
            "fixed_common_rzf_rsma": fixed_common_rsma(
                scenario.channels,
                common_power_fraction=common_power_fraction,
                private_scheme="rzf",
                **base_kwargs,
            ),
        }
        for name, result in results.items():
            rates = result.rates.user_rates
            fairness = float(np.sum(rates) ** 2 / (rates.size * np.sum(rates**2))) if np.any(rates) else 0.0
            gap = float(np.sum(np.maximum(scenario.qos_rate_targets - rates, 0.0)))
            reward = env_config.sum_rate_weight * result.rates.sum_rate + env_config.fairness_weight * fairness - env_config.qos_gap_weight * gap
            bucket = records[name]
            bucket[0].append(reward)
            bucket[1].append(result.rates.sum_rate)
            bucket[2].append(float(np.min(rates)))
            bucket[3].append(fairness)
            bucket[4].append(gap)
    environment.close()
    return {name: _metrics(*values) for name, values in records.items()}
