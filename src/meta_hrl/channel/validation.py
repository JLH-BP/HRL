"""Pure NumPy diagnostics for steering vectors and user-by-antenna channels.

This module has no random sampling, plotting, logging, or file-system side
effects. It validates normalized array manifolds, compares near- and far-field
models, and summarizes per-user channel energy and spatial correlation.
该模块没有随机抽样、绘图、日志记录或文件系统副作用。
它验证归一化阵列流形，比较近场和远场模型，并总结每个用户的通道能量和空间相关性。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .far_field import ula_los_steering_vector
from .geometry import ULAConfig, classify_near_field, polar_to_cartesian
from .near_field import ula_near_field_los_steering_vector

__all__ = [
    "ChannelDiagnosticsSummary",
    "NearFarFieldComparison",
    "compare_near_and_far_field",
    "steering_vector_norms",
    "summarize_channel_diagnostics",
    "user_channel_correlation_matrix",
]


@dataclass(frozen=True, slots=True)
class NearFarFieldComparison:
    """存储针对每个用户的近场与远场导向矢量（阵列流形）之间的差异诊断。"""

    is_near_field: NDArray[np.bool_]               # True 表示近场，False 表示远场
    coherence: NDArray[np.float64]                 # 近场与远场导向矢量之间的相干性，接近1表示近场和远场导向矢量非常相似，接近0表示它们非常不同
    direct_l2_error: NDArray[np.float64]           # 近场与远场导向矢量之间的直接L2误差，表示它们之间的绝对差异
    phase_invariant_l2_error: NDArray[np.float64]  # 去除整体相位后的误差，更关注波前曲率差异


@dataclass(frozen=True, slots=True)
class ChannelDiagnosticsSummary:
    """存储一个形状为 (用户数, 天线数) 的信道矩阵（每行对应一个用户的信道向量）的统计诊断信息。"""

    num_users: int
    num_antennas: int
    norm_min: float                        # 每个用户信道向量的最小L2范数
    norm_mean: float                       # 每个用户信道向量的平均L2范数
    norm_max: float                        # 每个用户信道向量的最大L2范数
    mean_channel_power: float              # 所有用户信道的平均功率
    max_pairwise_correlation: float        # 用户信道间归一化相关系数绝对值的最大值
    mean_pairwise_correlation: float       # 用户信道间归一化相关系数绝对值的平均值


def _finite_numeric_array(value: ArrayLike, name: str) -> NDArray[np.generic]:
    """将输入转换为 NumPy 数组，并确保所有元素是有限数值（实数或复数）。"""
    values = np.asarray(value)
    if values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain numeric values.")
    try:
        result = np.asarray(value, dtype=np.complex128 if np.iscomplexobj(values) else np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric values.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def steering_vector_norms(steering_vectors: ArrayLike) -> NDArray[np.float64]:
    """计算每个导向矢量或信道向量的 L2 范数"""
    vectors = _finite_numeric_array(steering_vectors, "steering_vectors")
    if vectors.ndim == 0 or vectors.shape[-1] == 0:
        raise ValueError("steering_vectors must have a nonempty final vector axis.")
    return np.asarray(np.linalg.norm(vectors, axis=-1), dtype=np.float64)


def _validate_user_channel_matrix(channels: ArrayLike) -> NDArray[np.complex128]:
    """用于确保输入的信道矩阵符合后续计算所需的格式和数值要求。"""
    matrix = _finite_numeric_array(channels, "channels")
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("channels must have a nonempty two-dimensional (users, antennas) shape.")
    norms = steering_vector_norms(matrix)
    if np.any(norms == 0.0):
        raise ValueError("channels must not contain a zero-norm user channel.")
    return np.asarray(matrix, dtype=np.complex128)


def user_channel_correlation_matrix(channels: ArrayLike) -> NDArray[np.complex128]:
    """计算用户信道之间的归一化复相关系数矩阵"""
    matrix = _validate_user_channel_matrix(channels)
    norms = steering_vector_norms(matrix)
    normalized = matrix / norms[:, np.newaxis]
    return np.asarray(normalized @ normalized.conj().T, dtype=np.complex128) # 矩阵的第i行第j 列元素代表用户i 和用户j 的信道向量之间的复相关系数。


def compare_near_and_far_field(
    config: ULAConfig, range_m: ArrayLike, angle_rad: ArrayLike
) -> NearFarFieldComparison:
    """量化近场球面波导向矢量与远场平面波导向矢量之间的差异"""
    if not isinstance(config, ULAConfig):
        raise TypeError("config must be a ULAConfig instance.")
    _, _ = polar_to_cartesian(range_m, angle_rad)
    ranges, angles = np.broadcast_arrays(
        np.asarray(range_m, dtype=np.float64), np.asarray(angle_rad, dtype=np.float64)
    )
    near_vectors = ula_near_field_los_steering_vector(config, ranges, angles)
    far_vectors = ula_los_steering_vector(config, angles)
    inner_products = np.sum(np.conj(near_vectors) * far_vectors, axis=-1)
    coherence = np.clip(np.abs(inner_products), 0.0, 1.0)  # 计算相干性
    direct_l2_error = steering_vector_norms(near_vectors - far_vectors)  # 计算直接的L2误差
    phase_invariant_l2_error = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * coherence)) # 计算相位不变的L2误差,消除了全局相位旋转的影响
    return NearFarFieldComparison(
        is_near_field=classify_near_field(ranges, config.rayleigh_distance_m),  # 判断是否为近场
        coherence=np.asarray(coherence, dtype=np.float64),  # 计算相干性
        direct_l2_error=direct_l2_error, # 计算直接的L2误差,直观显示差异大小
        phase_invariant_l2_error=np.asarray(phase_invariant_l2_error, dtype=np.float64),
    )


def summarize_channel_diagnostics(channels: ArrayLike) -> ChannelDiagnosticsSummary:
    """对用户信道矩阵进行全面的统计诊断，输出关于信道范数（幅度）、功率以及用户间空间相关性的标量汇总信息。"""
    # 检查生成信道的功率是否均匀、是否存在极端用户（范数过小或过大）
    matrix = _validate_user_channel_matrix(channels)
    norms = steering_vector_norms(matrix)
    correlation = user_channel_correlation_matrix(matrix)
    user_count, antenna_count = matrix.shape
    if user_count == 1:
        max_correlation = 0.0
        mean_correlation = 0.0
    else:
        pairwise = np.abs(correlation[np.triu_indices(user_count, k=1)])
        max_correlation = float(np.max(pairwise))
        mean_correlation = float(np.mean(pairwise))
    return ChannelDiagnosticsSummary(
        num_users=user_count,
        num_antennas=antenna_count,
        norm_min=float(np.min(norms)),
        norm_mean=float(np.mean(norms)),
        norm_max=float(np.max(norms)),
        mean_channel_power=float(np.mean(norms**2)),
        max_pairwise_correlation=max_correlation,
        mean_pairwise_correlation=mean_correlation,
    )
