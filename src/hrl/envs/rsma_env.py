# 实现了混合场单层RSMA的单步Gymnasium环境
""" 
    静态混合近场/远场Rician衰落RSMA资源分配环境。  
    一个episode在重置时采样一个信道实现并只接受一个''2K+1 ''-logit资源动作。
    将平面资源分配基线与以后的分组、层次控制和Meta-RL隔离开来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrl.rsma.constraints import ResourceAllocation, project_action_to_resource_allocation
from hrl.rsma.precoding import one_layer_rsma_precoders
from hrl.rsma.rate import OneLayerRSMARates, one_layer_rsma_rates
from hrl.rsma.signal_model import one_layer_rsma_signal_terms

from .observation_encoder import encode_observation, observation_size
from .task_sampler import RSMAScenario, ScenarioSamplerConfig, TaskSampler

__all__ = ["OneStepRSMAEnv", "RSMAEnvConfig", "jain_fairness"]


def _positive_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real-valued scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def _nonnegative_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real-valued scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True, slots=True)
class RSMAEnvConfig:
    """配置单步单层RSMA资源分配任务。"""

    sampler: ScenarioSamplerConfig = field(default_factory=ScenarioSamplerConfig)
    total_power: float = 1.0
    noise_variance: float = 1.0
    private_scheme: str = "rzf"    # 私有流的预编码方案，支持"mrt"或"rzf"
    sum_rate_weight: float = 1.0
    fairness_weight: float = 1.0
    qos_gap_weight: float = 1.0
    action_logit_bound: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.sampler, ScenarioSamplerConfig):
            raise TypeError("sampler must be a ScenarioSamplerConfig instance.")
        _positive_scalar(self.total_power, "total_power")             # 确保功率和噪声方差为正数
        _positive_scalar(self.noise_variance, "noise_variance")       
        if self.private_scheme not in {"mrt", "rzf"}:
            raise ValueError("private_scheme must be either 'mrt' or 'rzf'.")
        _nonnegative_scalar(self.sum_rate_weight, "sum_rate_weight")
        _nonnegative_scalar(self.fairness_weight, "fairness_weight")
        _nonnegative_scalar(self.qos_gap_weight, "qos_gap_weight")
        _positive_scalar(self.action_logit_bound, "action_logit_bound")


def jain_fairness(user_rates: ArrayLike) -> float:
    """返回Jain公平性指数，将全零情况定义为零。"""
    values = np.asarray(user_rates)
    # 检查是否为复数或非数值类型
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("user_rates must contain real numeric values.")
    # 将输入转换为float64数组
    rates = np.asarray(user_rates, dtype=np.float64)
    # 检查是否为一维数组，且不为空，且所有元素为有限值，且所有元素非负
    if rates.ndim != 1 or rates.size == 0 or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("user_rates must be a finite nonnegative nonempty vector.")
# 计算Jain公平性指数，定义全零情况为零
    squared_sum = float(np.sum(rates**2))
    if squared_sum == 0.0:
        return 0.0
    return float(np.sum(rates) ** 2 / (rates.size * squared_sum))   # ! 公式


class OneStepRSMAEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """采样一个静态Rician衰落RSMA任务并评估一个可行动作。"""

    metadata = {"render_modes": []}   # *不支持渲染

    def __init__(self, config: RSMAEnvConfig = RSMAEnvConfig()) -> None:
        if not isinstance(config, RSMAEnvConfig):
            raise TypeError("config must be an RSMAEnvConfig instance.")
        self.config = config
        self._sampler = TaskSampler(config.sampler)
        self._scenario: RSMAScenario | None = None
        self._has_stepped = False
        user_count = config.sampler.num_users
        antenna_count = config.sampler.ula_config.num_antennas
        self.action_space = gym.spaces.Box(   # ? 未归一化动作值取值范围
            low=-config.action_logit_bound,
            high=config.action_logit_bound,
            shape=(2 * user_count + 1,),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(  # ? 环境返回给智能体的观测（即神经网络的输入）
            low=-np.inf,
            high=np.inf,
            shape=(observation_size(user_count, antenna_count),),
            dtype=np.float32,
        )

    @property  # 将方法转化为只读
    def scenario(self) -> RSMAScenario:
        """暴露重置时的场景以用于确定性分析和测试。"""
        if self._scenario is None:
            raise RuntimeError("reset() must be called before accessing the scenario.")
        return self._scenario

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """负责重置一个episode采样一个新的静态任务并返回其编码后的CSI/几何观测。
        * ：强制要求后面的参数必须以关键字参数的形式传入。reset(send=42),而不是reset(42)
        Args:
            seed (int | None, optional): 随机种子，保证结果可复现. Defaults to None.
            options (dict[str, Any] | None, optional): 暂未调用. Defaults to None.

        Returns:
            tuple[NDArray[np.float32], dict[str, Any]]: _description_
        """        
     
        super().reset(seed=seed)
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary or None.")
        self._sampler.reset(seed)                         # 重置采样
        self._scenario = self._sampler.sample()           # 生成新的场景
        self._has_stepped = False                         # ! 阻止在同一场景下再次执行
        observation = encode_observation(self._scenario)  # 将复杂的物理场景（复数矩阵、距离数组）转换成固定长度的浮点数向量。
        return observation, self._scenario_info()

    def step(self, action: ArrayLike) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """
        负责将智能体输出的动作（一个 2K+1 维的 logit 向量）转化为实际的资源分配，
        并计算相应的性能指标（速率、公平性、QoS 满足度）和奖励。

        Args:
            action (ArrayLike): 智能体输出的动作（一个 2K+1 维的 logit 向量）

        Returns:
            tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]: 实际的资源分配
        """
        # 确保在调用 step() 之前已经调用了 reset()，并且在同一场景下只允许调用一次 step()。        
        if self._scenario is None:
            raise RuntimeError("reset() must be called before step().")
        if self._has_stepped:
            raise RuntimeError("This is a one-step environment; call reset() before step() again.")
        action_vector = self._validated_action(action)  # 验证动作向量的有效性（类型、形状、有限性）
        scenario = self._scenario
        user_count = self.config.sampler.num_users

        # *解耦“功率分配”和“速率分配” 先分配功率，再计算速率，再分配速率（根据物理层功率和信道，测出当前环境下公共流能提供的最大总速率，再进行分配）
        power_allocation = project_action_to_resource_allocation(
            np.concatenate((action_vector[: user_count + 1], np.zeros(user_count))),  # 取前 K+1 个元素作为功率分配的 logits，后 K 个元素置为零（不分配速率）
            num_users=user_count,
            total_power=self.config.total_power,
            common_rate_budget=0.0,
        )
        # 为公共流和每个私有流计算预编码向量
        precoders = one_layer_rsma_precoders(
            scenario.channels, power_allocation, private_scheme=self.config.private_scheme
        )
        # 计算信号项、物理速率和最终的资源分配
        signal_terms = one_layer_rsma_signal_terms(
            scenario.channels, precoders, noise_variance=self.config.noise_variance
        )
        # 计算在当前功率分配下，公共流能够支持的最大速率
        physical_rates = one_layer_rsma_rates(signal_terms, np.zeros(user_count, dtype=np.float64))
        # 同时处理功率和公共速率分配
        allocation = project_action_to_resource_allocation(
            action_vector,
            num_users=user_count,
            total_power=self.config.total_power,
            common_rate_budget=physical_rates.common_rate,
        )
        # 计算最终的速率、Jain公平性指数和QoS差距
        rates = one_layer_rsma_rates(signal_terms, allocation.common_rate_allocations)
        if not rates.is_common_rate_allocation_feasible:
            raise RuntimeError("Action projection unexpectedly violated the common-rate budget.")
        fairness = jain_fairness(rates.user_rates)
        qos_gaps = np.maximum(scenario.qos_rate_targets - rates.user_rates, 0.0)  #! QoS差距，若用户速率低于QoS目标，则为正值，否则为零
        # 计算奖励函数，综合考虑总速率、公平性和QoS差距，权重由配置参数控制
        reward = (
            self.config.sum_rate_weight * rates.sum_rate
            + self.config.fairness_weight * fairness
            - self.config.qos_gap_weight * float(np.sum(qos_gaps))   # ! 奖励函数
        )
        self._has_stepped = True  # 避免重复调用
        info = self._step_info(power_allocation, allocation, rates, fairness, qos_gaps)  # 记录当前步的详细信息，包括功率分配、速率分配、SINR、总速率、公平性和QoS差距等
        return encode_observation(scenario), float(reward), True, False, info

    def _validated_action(self, action: ArrayLike) -> NDArray[np.float64]:
        """将精确的有限实值动作向量转换为float64。"""
        values = np.asarray(action)
        if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
            raise TypeError("action must contain real numeric values.")
        vector = np.asarray(action, dtype=np.float64)
        if vector.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("action must contain only finite values.")
        return vector

    def _scenario_info(self) -> dict[str, Any]:
        """仅提取场景的几何和信道统计信息，用于信息字典"""
        scenario = self.scenario
        return {
            "ranges_m": scenario.ranges_m.copy(),
            "angles_rad": scenario.angles_rad.copy(),
            "near_field_mask": scenario.near_field_mask.copy(),
            "path_loss_linear": scenario.path_loss_linear.copy(),
            "k_factors_linear": scenario.k_factors_linear.copy(),
            "qos_rate_targets": scenario.qos_rate_targets.copy(),
        }

    def _step_info(
        self,
        power_allocation: ResourceAllocation,
        allocation: ResourceAllocation,
        rates: OneLayerRSMARates,
        fairness: float,
        qos_gaps: NDArray[np.float64],
    ) -> dict[str, Any]:
        info = self._scenario_info()
        info.update(
            {
                "power_allocation": power_allocation,
                "allocation": allocation,
                "common_sinrs": rates.common_sinrs.copy(),
                "private_sinrs": rates.private_sinrs.copy(),
                "common_rate": rates.common_rate,
                "private_rates": rates.private_rates.copy(),
                "user_rates": rates.user_rates.copy(),
                "sum_rate": rates.sum_rate,
                "jain_fairness": fairness,
                "qos_gaps": qos_gaps.copy(),
                "total_qos_gap": float(np.sum(qos_gaps)),
            }
        )
        return info
