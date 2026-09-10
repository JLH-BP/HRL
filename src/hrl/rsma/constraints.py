"""
将强化学习输出的动作向量（长度为2K+1的logits）映射为物理可行的资源分配
规范化的单层RSMA资源操作的约束。
实现了一层RSMA资源分配可行性和投影实用程序。
原始的策略-操作顺序是“[K个私有流功率logit，一个公共流功率logit， K个公共流速率logit]”。
功率对数按比例缩放到归一化总功率Pmax，共同速率对数按比例缩放到当前共同速率预算 Rc 。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "ResourceAllocation",
    "ResourceFeasibility",
    "project_action_to_resource_allocation",
    "project_nonnegative_simplex",
    "validate_resource_allocation",
]


def _nonnegative_finite_scalar(value: object, name: str) -> float:
    """Validate and return a finite nonnegative real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real-valued scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and greater than or equal to zero.")
    return result


def _finite_real_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert one finite real nonempty vector to float64."""
    values = np.asarray(value)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain real numeric values.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numeric values.") from error
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _stable_softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """将任意实数向量映射为概率单纯形（非负且和为1）"""
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


@dataclass(frozen=True, slots=True)
class ResourceAllocation:
    """表示一个公共流和每个用户的私有流资源分配。"""

    common_power: float
    private_powers: NDArray[np.float64]
    common_rate_allocations: NDArray[np.float64]

    def __post_init__(self) -> None:
        """规范化标量/矢量类型并验证匹配的用户维度"""
        common_power = _nonnegative_finite_scalar(self.common_power, "common_power")
        private_powers = _finite_real_vector(self.private_powers, "private_powers")
        common_rates = _finite_real_vector(
            self.common_rate_allocations, "common_rate_allocations"
        )
        if np.any(private_powers < 0.0) or np.any(common_rates < 0.0):
            raise ValueError("private_powers and common_rate_allocations must be nonnegative.")
        if private_powers.shape != common_rates.shape:
            raise ValueError(
                "private_powers and common_rate_allocations must have matching shapes."
            )
        object.__setattr__(self, "common_power", common_power)     # 将标准化后的数据写入实例，覆盖掉之前自动生成的占位数据
        object.__setattr__(self, "private_powers", private_powers)
        object.__setattr__(self, "common_rate_allocations", common_rates)


@dataclass(frozen=True, slots=True)
class ResourceFeasibility:
    """返回校验结果，包括是否可行以及功率和速率的剩余量。"""

    is_feasible: bool
    total_power: float
    allocated_power: float
    power_slack: float
    common_rate_budget: float
    allocated_common_rate: float
    common_rate_slack: float


def validate_resource_allocation(allocation: ResourceAllocation,*,total_power: float,
                                common_rate_budget: float,atol: float = 1e-10,) -> ResourceFeasibility:
    """ 根据功率和速率预算验证非负的单层RSMA资源分配是否可行。
        返回一个ResourceFeasibility对象，包含可行性标志和剩余量。"""
    if not isinstance(allocation, ResourceAllocation):
        raise TypeError("allocation must be a ResourceAllocation instance.")
    total_power = _nonnegative_finite_scalar(total_power, "total_power")
    common_rate_budget = _nonnegative_finite_scalar(
        common_rate_budget, "common_rate_budget"
    )
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, Real):
        raise TypeError("atol must be a real-valued scalar.")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and greater than or equal to zero.")

    allocated_power = allocation.common_power + float(np.sum(allocation.private_powers))
    allocated_common_rate = float(np.sum(allocation.common_rate_allocations))
    power_slack = total_power - allocated_power
    common_rate_slack = common_rate_budget - allocated_common_rate
    is_feasible = bool(
        np.all(allocation.private_powers >= -atol)   # 容许一定的数值误差，允许小于零的值在绝对容差范围内
        and np.all(allocation.common_rate_allocations >= -atol)
        and power_slack >= -atol
        and common_rate_slack >= -atol
    )
    return ResourceFeasibility(
        is_feasible=is_feasible,
        total_power=total_power,
        allocated_power=allocated_power,
        power_slack=power_slack,
        common_rate_budget=common_rate_budget,
        allocated_common_rate=allocated_common_rate,
        common_rate_slack=common_rate_slack,
    )


def project_nonnegative_simplex(values: ArrayLike, *, budget: float) -> NDArray[np.float64]:
    """将任意实数向量投影到非负单纯形上，使其和不超过给定的预算。"""
    vector = _finite_real_vector(values, "values")
    budget = _nonnegative_finite_scalar(budget, "budget")
    if budget == 0.0:
        return np.zeros_like(vector)
    if np.all(vector >= 0.0) and float(np.sum(vector)) <= budget:
        return vector.copy()

    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, vector.size + 1, dtype=np.float64)
    valid = sorted_values - (cumulative - budget) / indices > 0.0
    if not np.any(valid):
        return np.zeros_like(vector)
    rho = int(np.nonzero(valid)[0][-1])
    threshold = (cumulative[rho] - budget) / float(rho + 1)
    return np.maximum(vector - threshold, 0.0)


def project_action_to_resource_allocation(action: ArrayLike,*,num_users: int,
                                          total_power: float,common_rate_budget: float) -> ResourceAllocation:
    """将“2K+1”策略逻辑映射为严格可行的单层分配。"""
    if isinstance(num_users, (bool, np.bool_)) or not isinstance(num_users, Integral):
        raise TypeError("num_users must be an integer.")
    if num_users < 1:
        raise ValueError("num_users must be at least one.")
    total_power = _nonnegative_finite_scalar(total_power, "total_power")
    common_rate_budget = _nonnegative_finite_scalar(common_rate_budget, "common_rate_budget")
    action_vector = _finite_real_vector(action, "action")
    expected_size = 2 * num_users + 1
    if action_vector.size != expected_size:
        raise ValueError(f"action must contain exactly {expected_size} entries.")

    power_fractions = _stable_softmax(action_vector[: num_users + 1])
    private_powers = total_power * power_fractions[:num_users]
    common_power = float(total_power * power_fractions[num_users])   # 针对单层RSMA，只有一个公共流
    if common_rate_budget == 0.0:
        common_rates = np.zeros(num_users, dtype=np.float64)
    else:
        common_rates = common_rate_budget * _stable_softmax(action_vector[num_users + 1 :])  # 将公共速率logits映射到非负单纯形上，确保总和不超过公共速率预算
    allocation = ResourceAllocation(
        common_power=common_power,
        private_powers=private_powers,
        common_rate_allocations=common_rates,
    )
    feasibility = validate_resource_allocation(
        allocation,
        total_power=total_power,
        common_rate_budget=common_rate_budget,
    )
    if not feasibility.is_feasible:
        raise RuntimeError("Projected action unexpectedly violates resource constraints.")
    return allocation
