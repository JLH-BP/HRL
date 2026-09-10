""" 计算 RSMA 环境的观测向量长度。
    观测向量由每个用户的 CSI（复数向量）和6 个几何/信道特征组成：
        1. 对数范围（log1p(ranges_m)）
        2. 归一化角度（angles_rad / π）
        3. 近场标志（near_field_mask.astype(float)）
        4. 对数路径增益（log1p(path_loss_linear)）
        5. 对数 K 因子（log1p(k_factors_linear)）
        6. QoS 速率目标（qos_rate_targets）"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .task_sampler import RSMAScenario

__all__ = ["OBSERVATION_METADATA_SIZE", "encode_observation", "observation_size"]

OBSERVATION_METADATA_SIZE: int = 6


def observation_size(num_users: int, num_antennas: int) -> int:

    if isinstance(num_users, bool) or not isinstance(num_users, int) or num_users < 1:
        raise ValueError("num_users must be a positive integer.")
    if isinstance(num_antennas, bool) or not isinstance(num_antennas, int) or num_antennas < 1:
        raise ValueError("num_antennas must be a positive integer.")
    return num_users * (2 * num_antennas + OBSERVATION_METADATA_SIZE)


def _finite_vector(value: ArrayLike, name: str, size: int, *, nonnegative: bool = False) -> NDArray[np.float64]:
    """验证并返回一个有限的实数向量。"""
    values = np.asarray(value)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain real numeric values.")
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    if nonnegative and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return vector


def encode_observation(scenario: RSMAScenario) -> NDArray[np.float32]:
    """编码正常化后的 CSI 及几何观测为一个有限的 float32 向量。    """
    if not isinstance(scenario, RSMAScenario):
        raise TypeError("scenario must be an RSMAScenario instance.")
    channels = np.asarray(scenario.channels, dtype=np.complex128)
    if channels.ndim != 2 or 0 in channels.shape or not np.all(np.isfinite(channels)):
        raise ValueError("scenario.channels must be a finite nonempty two-dimensional matrix.")
    num_users, _ = channels.shape
    ranges = _finite_vector(scenario.ranges_m, "scenario.ranges_m", num_users, nonnegative=True)
    angles = _finite_vector(scenario.angles_rad, "scenario.angles_rad", num_users)
    path_gains = _finite_vector(scenario.path_loss_linear, "scenario.path_loss_linear", num_users, nonnegative=True)
    k_factors = _finite_vector(scenario.k_factors_linear, "scenario.k_factors_linear", num_users, nonnegative=True)
    qos = _finite_vector(scenario.qos_rate_targets, "scenario.qos_rate_targets", num_users, nonnegative=True)
    near_flags = np.asarray(scenario.near_field_mask)
    if near_flags.shape != (num_users,) or near_flags.dtype != np.bool_:
        raise ValueError("scenario.near_field_mask must be a boolean vector per user.")
    norms = np.linalg.norm(channels, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("scenario.channels must not contain a zero-norm user channel.")
    normalized_channels = channels / norms[:, np.newaxis]
    range_feature = np.log1p(ranges)
    metadata = np.column_stack(
        (
            range_feature,
            angles / np.pi,
            near_flags.astype(np.float64),
            np.log1p(path_gains),
            np.log1p(k_factors),
            qos,
        )
    )
    per_user = np.concatenate(
        (normalized_channels.real, normalized_channels.imag, metadata), axis=1
    )
    observation = np.asarray(per_user.reshape(-1), dtype=np.float32)
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("Observation encoding unexpectedly produced nonfinite values.")
    return observation
