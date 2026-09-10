"""
理想SIC单层RSMA信号项的速率计算
所有用户解码单个公共流，因此其可用速率是每个用户的最小公共可解码速率。
速率使用 log2(1 + SINR) ，单位为bit/s/Hz。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .signal_model import OneLayerRSMASignalTerms

__all__ = ["OneLayerRSMARates", "one_layer_rsma_rates"]


@dataclass(frozen=True, slots=True)
class OneLayerRSMARates:
    """存储单层RSMA SINR、rate和common rate可行性结果"""

    common_sinrs: NDArray[np.float64]
    private_sinrs: NDArray[np.float64]
    common_decodable_rates: NDArray[np.float64]   # 每个用户单独能够解码公共流的最大速率
    private_rates: NDArray[np.float64]
    common_rate: float                            # 公共流的总可达速率瓶颈 = min(common_decodable_rates)，即所有用户都能成功解码的最大公共速率
    common_rate_allocations: NDArray[np.float64]
    allocated_common_rate: float
    common_rate_slack: float
    is_common_rate_allocation_feasible: bool
    user_rates: NDArray[np.float64]
    sum_rate: float


def _finite_nonnegative_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Validate and return a finite nonnegative nonempty real vector."""
    values = np.asarray(value)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain real numeric values.")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numeric values.") from error
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(vector < 0.0):
        raise ValueError(f"{name} must contain values greater than or equal to zero.")
    return vector.copy()


def _positive_noise_variance(value: object) -> float:
    """Validate and return a strictly positive finite real noise variance."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("signal_terms.noise_variance must be a real-valued scalar.")
    variance = float(value)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("signal_terms.noise_variance must be finite and greater than zero.")
    return variance


def _validated_signal_power_vectors(
    signal_terms: OneLayerRSMASignalTerms,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """从 OneLayerRSMASignalTerms 中提取四个功率向量（公共信号、公共干扰、私有信号、私有干扰），并校验它们形状一致、非负有限，同时返回噪声方差。"""
    if not isinstance(signal_terms, OneLayerRSMASignalTerms):
        raise TypeError("signal_terms must be a OneLayerRSMASignalTerms instance.")
    common_signal = _finite_nonnegative_vector(
        signal_terms.common_signal_powers, "common_signal_powers"
    )
    common_interference = _finite_nonnegative_vector(
        signal_terms.common_interference_powers, "common_interference_powers"
    )
    private_signal = _finite_nonnegative_vector(
        signal_terms.private_signal_powers, "private_signal_powers"
    )
    private_interference = _finite_nonnegative_vector(
        signal_terms.private_interference_powers, "private_interference_powers"
    )
    shapes = {
        common_signal.shape,
        common_interference.shape,
        private_signal.shape,
        private_interference.shape,
    }
    if len(shapes) != 1:
        raise ValueError("All signal-term power vectors must have matching shapes.")
    return (
        common_signal,
        common_interference,
        private_signal,
        private_interference,
        _positive_noise_variance(signal_terms.noise_variance),
    )


def one_layer_rsma_rates(
    signal_terms: OneLayerRSMASignalTerms,
    common_rate_allocations: ArrayLike,
    *,
    atol: float = 1e-10,
) -> OneLayerRSMARates:
    """计算理想sic单层RSMA速率和通用速率可行性。
    策略决策的公共速率分配（K维向量，单位为 bit/s/Hz）
    """
    (   common_signal,
        common_interference,
        private_signal,
        private_interference,
        noise_variance,
    ) = _validated_signal_power_vectors(signal_terms)
    allocations = _finite_nonnegative_vector(
        common_rate_allocations, "common_rate_allocations"
    )
    if allocations.shape != common_signal.shape:
        raise ValueError("common_rate_allocations must have one entry per user.")
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, Real):
        raise TypeError("atol must be a real-valued scalar.")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and greater than or equal to zero.")

    common_sinrs = np.asarray(common_signal / (common_interference + noise_variance), dtype=np.float64)    # 公共流SINR
    private_sinrs = np.asarray(private_signal / (private_interference + noise_variance), dtype=np.float64) # 私有流SINR

    common_decodable_rates = np.asarray(np.log2(1.0 + common_sinrs), dtype=np.float64)    # 每个用户单独能够解码公共流的最大速率
    private_rates = np.asarray(np.log2(1.0 + private_sinrs), dtype=np.float64)
    common_rate = float(np.min(common_decodable_rates))   # 公共流的总速率必须 ≤ 最差用户的可解码速率
    allocated_common_rate = float(np.sum(allocations))
    common_rate_slack = common_rate - allocated_common_rate
    user_rates = np.asarray(allocations + private_rates, dtype=np.float64)

    return OneLayerRSMARates(
        common_sinrs=common_sinrs,
        private_sinrs=private_sinrs,
        common_decodable_rates=common_decodable_rates,
        private_rates=private_rates,
        common_rate=common_rate,
        common_rate_allocations=allocations,
        allocated_common_rate=allocated_common_rate,
        common_rate_slack=common_rate_slack,
        is_common_rate_allocation_feasible=bool(common_rate_slack >= -atol),
        user_rates=user_rates,
        sum_rate=float(np.sum(user_rates)),
    )
