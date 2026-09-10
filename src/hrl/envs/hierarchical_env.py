# 多步组 RSMA 环境，显式分离 Manager 和 Worker 接口
"""Gymnasium环境用于多步RSMA Worker控制和Manager分区选择。"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrl.agents.low_level_policy import validate_worker_action
from hrl.grouping.candidate_groups import UserPartition, enumerate_candidate_partitions
from hrl.rsma.group_rsma import active_group_slots, group_rsma_precoders, group_rsma_rates, project_group_action

from .observation_encoder import encode_observation, observation_size
from .rsma_env import jain_fairness
from .task_sampler import RSMAScenario, ScenarioSamplerConfig, TaskSampler

__all__ = ["HierarchicalRSMAEnv", "HierarchicalRSMAEnvConfig"]


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
        self.manager_action_space = gym.spaces.Discrete(len(self.candidates))
        self.worker_action_space = gym.spaces.Box(-config.action_logit_bound, config.action_logit_bound, shape=(3 * config.sampler.num_users,), dtype=np.float32)
        self.action_space = self.worker_action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(
                                                observation_size(config.sampler.num_users, config.sampler.ula_config.num_antennas) + 
                                                                 config.sampler.num_users,), dtype=np.float32)
        self._scenario: RSMAScenario | None = None
        self._partition_index: int | None = None
        self._steps = 0
        self._last_rates = np.zeros(config.sampler.num_users, dtype=np.float64)
        self._partition_switched = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """重置环境并返回初始观察值和信息。"""
        super().reset(seed=seed)
        self._sampler.reset(seed)
        self._scenario = self._sampler.sample()
        self._partition_index = None
        self._steps = 0
        self._last_rates.fill(0.0)
        self._partition_switched = False
        return self._observation(), {"manager_action_required": True}

    def set_manager_action(self, action: int) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """为下一个Worker策略间隔中选择候选分区。"""
        if self._scenario is None:
            raise RuntimeError("reset() must be called before selecting a partition.")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral) or not self.manager_action_space.contains(int(action)):
            raise ValueError("action must be a valid discrete candidate partition index.")
        previous = self._partition_index
        self._partition_index = int(action)
        switched = previous is not None and previous != self._partition_index
        self._partition_switched = switched
        return self._observation(), {"partition": self.partition, "switched": switched}

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
        zero_allocation = project_group_action(values, partition=self.partition, num_users=users, total_power=self.config.total_power)
        beams = group_rsma_precoders(scenario.channels, zero_allocation, self.partition, private_scheme=self.config.private_scheme)  # 预编码
        slots = active_group_slots(self.partition, num_users=users) # 当前选定的用户分组（Partition）转换成程序可遍历的 {组ID: [用户列表]} 字典
        received = np.abs(scenario.channels @ beams) ** 2  # 计算接收信号功率
        budgets = np.zeros(users, dtype=np.float64)        # 计算每个用户的速率预算
        for slot, members in slots.items():  
            indices = np.asarray(members)
            signal = received[indices, users + slot]
            interference = np.sum(received[indices, :users], axis=1) + np.sum(received[indices, users:], axis=1) - signal
            budgets[slot] = np.min(np.log2(1.0 + signal / (interference + self.config.noise_variance)))
        allocation = project_group_action(values, partition=self.partition, num_users=users, total_power=self.config.total_power, group_rate_budgets=budgets)
        beams = group_rsma_precoders(
            scenario.channels, allocation, self.partition, private_scheme=self.config.private_scheme
        )
        rates = group_rsma_rates(scenario.channels, beams, allocation, self.partition, noise_variance=self.config.noise_variance)
        fairness = jain_fairness(rates.user_rates)
        gap = np.maximum(scenario.qos_rate_targets - rates.user_rates, 0.0)
        reward = self.config.sum_rate_weight * rates.sum_rate + self.config.fairness_weight * fairness - self.config.qos_gap_weight * float(np.sum(gap)) - self.config.partition_switch_penalty * float(self._partition_switched)
        self._last_rates = rates.user_rates.copy()
        self._steps += 1
        terminated = self._steps >= self.config.episode_length
        manager_required = not terminated and self._steps % self.config.high_level_interval == 0
        info = {"partition": self.partition, 
                "partition_switched": self._partition_switched, 
                "switch_penalty": self.config.partition_switch_penalty * float(self._partition_switched), 
                "group_common_rates": rates.group_common_rates.copy(), 
                "user_rates": rates.user_rates.copy(), 
                "sum_rate": rates.sum_rate, 
                "jain_fairness": fairness, 
                "qos_gaps": gap.copy(), 
                "manager_action_required": manager_required}
        self._partition_switched = False
        return self._observation(), float(reward), terminated, False, info

    def _observation(self) -> NDArray[np.float32]:
        if self._scenario is None:
            raise RuntimeError("reset() 必须在访问观察值之前调用。")
        return np.concatenate((encode_observation(self._scenario), self._last_rates.astype(np.float32)))
