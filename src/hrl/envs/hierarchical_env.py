# 多步组 RSMA 环境，显式分离 Manager 和 Worker 接口
"""Gymnasium环境用于多步RSMA Worker控制和Manager分区选择。"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrl.agents.low_level_policy import validate_worker_action
from hrl.grouping.candidate_groups import UserPartition, enumerate_candidate_partitions
from hrl.rsma.group_rsma import (
    active_group_slots,
    group_rsma_precoders,
    group_rsma_rates,
    project_group_action,
    service_masked_partition,
)

from .observation_encoder import encode_observation, observation_size
from .rsma_env import jain_fairness
from .task_sampler import RSMAScenario, ScenarioSamplerConfig, TaskSampler

ServiceMode = Literal["all", "near", "far"]
SERVICE_MODES: tuple[ServiceMode, ...] = ("all", "near", "far")

__all__ = [
    "HierarchicalRSMAEnv",
    "HierarchicalRSMAEnvConfig",
    "SERVICE_MODES",
    "ServiceMode",
]


def _normalize_service_mode(value: object) -> ServiceMode:
    """Validate a public mode name or its compact discrete encoding."""
    if isinstance(value, str):
        if value in SERVICE_MODES:
            return value  # type: ignore[return-value]
        raise ValueError(f"service_mode must be one of {SERVICE_MODES}.")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("service_mode must be a mode name or integer mode index.")
    index = int(value)
    if not 0 <= index < len(SERVICE_MODES):
        raise ValueError(f"service_mode index must be in [0, {len(SERVICE_MODES)}).")
    return SERVICE_MODES[index]


@dataclass(frozen=True, slots=True)
class HierarchicalRSMAEnvConfig:
    """配置静态任务群RSMA管理者和Worker控制。"""

    sampler: ScenarioSamplerConfig = field(default_factory=ScenarioSamplerConfig)
    total_power: float = 1.0
    noise_variance: float = 1.0
    private_scheme: str = "rzf"
    high_level_interval: int = 4
    episode_length: int = 16
    partition_switch_penalty: float = 0.05   # 分组切换的惩罚系数
    sum_rate_weight: float = 1.0
    fairness_weight: float = 1.0
    qos_gap_weight: float = 1.0
    action_logit_bound: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, ScenarioSamplerConfig):
            raise TypeError("sampler must be a ScenarioSamplerConfig instance.")
        if self.private_scheme not in {"mrt", "rzf"}:
            raise ValueError("private_scheme must be 'mrt' or 'rzf'.")
        for name in ("high_level_interval", "episode_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("total_power", "noise_variance", "action_logit_bound"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("partition_switch_penalty", "sum_rate_weight", "fairness_weight", "qos_gap_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")


class HierarchicalRSMAEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Gymnasium环境用于多步RSMA Worker控制和Manager分区选择。"""

    metadata = {"render_modes": []}

    def __init__(self, config: HierarchicalRSMAEnvConfig = HierarchicalRSMAEnvConfig()) -> None:
        self.config = config
        self._sampler = TaskSampler(config.sampler)
        self.candidates = enumerate_candidate_partitions(num_users=config.sampler.num_users)
        # A PPO-compatible scalar encoding of (partition index, service mode).
        # The inverse mapping is exposed below so fixed policies can keep using
        # the clearer tuple form.
        self.manager_action_space = gym.spaces.Discrete(len(self.candidates) * len(SERVICE_MODES))
        self.worker_action_space = gym.spaces.Box(-config.action_logit_bound, config.action_logit_bound, shape=(3 * config.sampler.num_users,), dtype=np.float32)
        self.action_space = self.worker_action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(
                                                observation_size(config.sampler.num_users, config.sampler.ula_config.num_antennas) + 
                                                                 4 * config.sampler.num_users,), dtype=np.float32)
        self._scenario: RSMAScenario | None = None
        self._partition_index: int | None = None
        self._service_mode: ServiceMode = "all"
        self._service_mask = np.ones(config.sampler.num_users, dtype=np.bool_)
        self._steps = 0
        self._last_rates = np.zeros(config.sampler.num_users, dtype=np.float64)
        self._cumulative_user_rates = np.zeros(config.sampler.num_users, dtype=np.float64)
        self._cumulative_qos_gaps = np.zeros(config.sampler.num_users, dtype=np.float64)
        self._cumulative_qos_gap_sum = np.zeros(config.sampler.num_users, dtype=np.float64)
        self._service_counts = np.zeros(config.sampler.num_users, dtype=np.int64)
        self._previous_episode_utility = 0.0
        self._episode_reward = 0.0
        self._partition_switch_count = 0
        self._service_mode_switch_count = 0
        self._manager_decision_count = 0
        self._partition_switched = False
        self._service_mode_switched = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """重置环境并返回初始观察值和信息。"""
        super().reset(seed=seed)
        self._sampler.reset(seed)
        self._scenario = self._sampler.sample()
        self._partition_index = None
        self._service_mode = "all"
        self._service_mask.fill(True)
        self._steps = 0
        self._last_rates.fill(0.0)
        self._cumulative_user_rates.fill(0.0)
        self._cumulative_qos_gaps.fill(0.0)
        self._cumulative_qos_gap_sum.fill(0.0)
        self._service_counts.fill(0)
        self._previous_episode_utility = 0.0
        self._episode_reward = 0.0
        self._partition_switch_count = 0
        self._service_mode_switch_count = 0
        self._manager_decision_count = 0
        self._partition_switched = False
        self._service_mode_switched = False
        return self._observation(), {
            "manager_action_required": True,
            "service_mode": self._service_mode,
            "service_mask": self.service_mask,
            "near_field_mask": self.scenario.near_field_mask.copy(),
            "qos_rate_targets": self.scenario.qos_rate_targets.copy(),
        }

    def encode_manager_action(self, partition_index: int, service_mode: ServiceMode | str | int) -> int:
        """Encode a manager tuple into the scalar action used by discrete PPO."""
        if isinstance(partition_index, (bool, np.bool_)) or not isinstance(partition_index, Integral):
            raise TypeError("partition_index must be an integer.")
        index = int(partition_index)
        if not 0 <= index < len(self.candidates):
            raise ValueError("partition_index must select a candidate partition.")
        mode = _normalize_service_mode(service_mode)
        return index * len(SERVICE_MODES) + SERVICE_MODES.index(mode)

    def decode_manager_action(self, action: int) -> tuple[int, ServiceMode]:
        """Decode a scalar PPO action into ``(partition_index, service_mode)``."""
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral):
            raise TypeError("action must be an integer manager-action encoding.")
        encoded = int(action)
        if not self.manager_action_space.contains(encoded):
            raise ValueError("action must be a valid encoded Manager action.")
        partition_index, mode_index = divmod(encoded, len(SERVICE_MODES))
        return partition_index, SERVICE_MODES[mode_index]

    def set_partition_action(self, partition_index: int) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Select a legacy all-user partition action explicitly.

        This keeps older fixed-partition call sites readable without making a
        scalar action ambiguous now that all scalar actions encode a mode too.
        """
        return self.set_manager_action((partition_index, "all"))

    def set_manager_action(
        self, action: int | tuple[int, ServiceMode | str | int]
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Select a partition and which near/far users receive service.

        Scalar actions use ``partition_index * 3 + service_mode_index`` with
        mode indices ``all=0``, ``near=1``, and ``far=2``.  Fixed policies may
        pass the explicit tuple form instead.
        """
        if self._scenario is None:
            raise RuntimeError("reset() must be called before selecting a partition.")
        if isinstance(action, tuple):
            if len(action) != 2:
                raise ValueError("tuple manager actions must be (partition_index, service_mode).")
            partition_index, raw_mode = action
            if isinstance(partition_index, (bool, np.bool_)) or not isinstance(partition_index, Integral):
                raise TypeError("partition_index must be an integer.")
            partition_index = int(partition_index)
            if not 0 <= partition_index < len(self.candidates):
                raise ValueError("partition_index must select a candidate partition.")
            service_mode = _normalize_service_mode(raw_mode)
        else:
            partition_index, service_mode = self.decode_manager_action(action)
        previous = self._partition_index
        previous_mode = self._service_mode
        self._partition_index = partition_index
        self._service_mode = service_mode
        self._service_mask = self._service_mask_for_mode(service_mode)
        switched = previous is not None and previous != self._partition_index
        mode_switched = previous is not None and previous_mode != self._service_mode
        self._partition_switched = switched
        self._service_mode_switched = mode_switched
        self._partition_switch_count += int(switched)
        self._service_mode_switch_count += int(mode_switched)
        self._manager_decision_count += 1
        return self._observation(), {
            "partition": self.partition,
            "effective_partition": self.effective_partition,
            "partition_index": self._partition_index,
            "service_mode": self._service_mode,
            "service_mode_index": SERVICE_MODES.index(self._service_mode),
            "service_mask": self.service_mask,
            "scheduled_user_count": int(np.count_nonzero(self._service_mask)),
            "switched": switched,
            "partition_switched": switched,
            "service_mode_switched": mode_switched,
        }

    @property
    def scenario(self) -> RSMAScenario:
        """返回当前采样的RSMA场景。"""
        if self._scenario is None:
            raise RuntimeError("reset() must be called before accessing the scenario.")
        return self._scenario

    @property
    def partition(self) -> UserPartition:
        if self._partition_index is None:
            raise RuntimeError("A manager action must select a partition before worker step().")
        return self.candidates[self._partition_index]

    @property
    def service_mode(self) -> ServiceMode:
        """The current high-level service choice."""
        return self._service_mode

    @property
    def service_mask(self) -> NDArray[np.bool_]:
        """A copy of the users permitted to receive downlink resources."""
        return self._service_mask.copy()

    @property
    def effective_partition(self) -> UserPartition:
        """Current partition after forcing all unscheduled users to singleton."""
        return service_masked_partition(
            self.partition,
            service_mask=self._service_mask,
            num_users=self.config.sampler.num_users,
        )

    def _service_mask_for_mode(self, service_mode: ServiceMode) -> NDArray[np.bool_]:
        if service_mode == "all":
            return np.ones(self.config.sampler.num_users, dtype=np.bool_)
        if service_mode == "near":
            return self.scenario.near_field_mask.copy()
        return np.logical_not(self.scenario.near_field_mask)

    def step(self, action: ArrayLike) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # 确保已经重置且已选择分区。
        if self._scenario is None or self._partition_index is None:
            raise RuntimeError("reset() and set_manager_action() are required before worker step().")
        values = validate_worker_action(
            action,
            num_users=self.config.sampler.num_users,
            logit_bound=self.config.action_logit_bound,
        )
        scenario = self._scenario
        users = self.config.sampler.num_users
        service_mask = self._service_mask
        zero_allocation = project_group_action(
            values,
            partition=self.partition,
            num_users=users,
            total_power=self.config.total_power,
            service_mask=service_mask,
        )
        beams = group_rsma_precoders(
            scenario.channels,
            zero_allocation,
            self.partition,
            private_scheme=self.config.private_scheme,
            service_mask=service_mask,
        )
        slots = active_group_slots(
            self.partition, num_users=users, service_mask=service_mask
        )
        received = np.abs(scenario.channels @ beams) ** 2  # 计算接收信号功率
        budgets = np.zeros(users, dtype=np.float64)        # 计算每个用户的速率预算
        for slot, members in slots.items():  
            indices = np.asarray(members)
            signal = received[indices, users + slot]
            interference = np.sum(received[indices, :users], axis=1) + np.sum(received[indices, users:], axis=1) - signal
            budgets[slot] = np.min(np.log2(1.0 + signal / (interference + self.config.noise_variance)))
        allocation = project_group_action(
            values,
            partition=self.partition,
            num_users=users,
            total_power=self.config.total_power,
            group_rate_budgets=budgets,
            service_mask=service_mask,
        )
        beams = group_rsma_precoders(
            scenario.channels,
            allocation,
            self.partition,
            private_scheme=self.config.private_scheme,
            service_mask=service_mask,
        )
        rates = group_rsma_rates(
            scenario.channels,
            beams,
            allocation,
            self.partition,
            noise_variance=self.config.noise_variance,
            service_mask=service_mask,
        )
        self._last_rates = rates.user_rates.copy()
        instantaneous_fairness = jain_fairness(rates.user_rates)
        instantaneous_gap = np.maximum(scenario.qos_rate_targets - rates.user_rates, 0.0)
        self._cumulative_user_rates += rates.user_rates
        self._steps += 1
        cumulative_targets = scenario.qos_rate_targets * self._steps
        self._cumulative_qos_gaps = np.maximum(
            cumulative_targets - self._cumulative_user_rates, 0.0
        )
        self._cumulative_qos_gap_sum += self._cumulative_qos_gaps
        self._service_counts += service_mask.astype(np.int64)

        # Optimize the change in a history-dependent episode objective.  The
        # terminal return therefore equals the utility of the cumulative user
        # throughputs and cumulative QoS deficits, rather than rewarding a
        # high-rate near-field-only slot in isolation.
        cumulative_fairness = jain_fairness(self._cumulative_user_rates)
        episode_utility = (
            self.config.sum_rate_weight * float(np.sum(self._cumulative_user_rates))
            + self.config.fairness_weight * cumulative_fairness
            - self.config.qos_gap_weight * float(np.sum(self._cumulative_qos_gaps))
        )
        switch_penalty = self.config.partition_switch_penalty * float(self._partition_switched)
        reward = episode_utility - self._previous_episode_utility - switch_penalty
        self._previous_episode_utility = episode_utility
        self._episode_reward += reward
        terminated = self._steps >= self.config.episode_length
        manager_required = not terminated and self._steps % self.config.high_level_interval == 0
        average_user_rates = self._cumulative_user_rates / self._steps
        service_fractions = self._service_counts.astype(np.float64) / self._steps
        near_mask = scenario.near_field_mask
        near_average_rate = float(np.mean(average_user_rates[near_mask])) if np.any(near_mask) else 0.0
        far_average_rate = float(np.mean(average_user_rates[~near_mask])) if np.any(~near_mask) else 0.0
        info = {
            "partition": self.partition,
            "effective_partition": self.effective_partition,
            "partition_index": self._partition_index,
            "service_mode": self._service_mode,
            "service_mode_index": SERVICE_MODES.index(self._service_mode),
            "service_mask": self.service_mask,
            "scheduled_user_count": int(np.count_nonzero(service_mask)),
            "partition_switched": self._partition_switched,
            "service_mode_switched": self._service_mode_switched,
            "switch_penalty": switch_penalty,
            "group_common_rates": rates.group_common_rates.copy(),
            "allocation": allocation,
            "user_rates": rates.user_rates.copy(),
            "sum_rate": rates.sum_rate,
            "min_user_rate": float(np.min(rates.user_rates)),
            "jain_fairness": instantaneous_fairness,
            "qos_gaps": instantaneous_gap.copy(),
            "near_field_mask": near_mask.copy(),
            "cumulative_user_rates": self._cumulative_user_rates.copy(),
            "episode_user_rates": self._cumulative_user_rates.copy(),
            "cumulative_qos_targets": cumulative_targets.copy(),
            "cumulative_qos_gaps": self._cumulative_qos_gaps.copy(),
            "episode_qos_gaps": self._cumulative_qos_gaps.copy(),
            "cumulative_qos_gap_sum": self._cumulative_qos_gap_sum.copy(),
            "total_qos_gap": float(np.sum(self._cumulative_qos_gaps)),
            "average_user_rates": average_user_rates.copy(),
            "cumulative_sum_rate": float(np.sum(self._cumulative_user_rates)),
            "average_sum_rate": float(np.sum(average_user_rates)),
            "average_min_user_rate": float(np.min(average_user_rates)),
            "cumulative_jain_fairness": cumulative_fairness,
            "qos_satisfaction_rate": float(
                np.mean(average_user_rates >= scenario.qos_rate_targets)
            ),
            "near_average_user_rate": near_average_rate,
            "far_average_user_rate": far_average_rate,
            "near_far_rate_gap": near_average_rate - far_average_rate,
            "service_counts": self._service_counts.copy(),
            "service_fractions": service_fractions.copy(),
            "partition_switch_count": self._partition_switch_count,
            "service_mode_switch_count": self._service_mode_switch_count,
            "manager_decision_count": self._manager_decision_count,
            "manager_decision_frequency": self._manager_decision_count / self._steps,
            "episode_utility": episode_utility,
            "episode_reward": self._episode_reward,
            "manager_action_required": manager_required,
        }
        self._partition_switched = False
        self._service_mode_switched = False
        return self._observation(), float(reward), terminated, False, info

    def _observation(self) -> NDArray[np.float32]:
        """Encode CSI plus instantaneous, cumulative, and scheduling state.

        The final fixed-width blocks are ``last_rates``,
        ``cumulative_user_rates``, ``cumulative_qos_gaps``, and
        ``service_mask``.  Exposing the cumulative terms keeps the
        history-dependent QoS objective observable to both Worker and Manager.
        """
        if self._scenario is None:
            raise RuntimeError("reset() 必须在访问观察值之前调用。")
        return np.concatenate(
            (
                encode_observation(self._scenario),
                self._last_rates.astype(np.float32),
                self._cumulative_user_rates.astype(np.float32),
                self._cumulative_qos_gaps.astype(np.float32),
                self._service_mask.astype(np.float32),
            )
        )
