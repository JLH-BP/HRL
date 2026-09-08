"""_summary_
    混合场 Rician 信道采样函数 sample_rician_channel，用于生成从多个用户到 ULA 各阵元的信道系数。
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .far_field import ula_los_steering_vector
from .geometry import ULAConfig, classify_near_field, polar_to_cartesian
from .near_field import ula_near_field_los_steering_vector

__all__ = ["sample_rician_channel"]


def _finite_nonnegative_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert a finite nonnegative real-valued input to a float64 array."""
    values = np.asarray(value)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain real numeric values.")
    if values.ndim == 0 and not isinstance(value, (Real, np.number)):
        raise TypeError(f"{name} must contain real numeric values.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numeric values.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must contain values greater than or equal to zero.")
    return result


def sample_rician_channel(
    config: ULAConfig,
    range_m: ArrayLike,
    angle_rad: ArrayLike,
    k_factor_linear: ArrayLike,
    path_loss_linear: ArrayLike = 1.0,
    *,
    rng: np.random.Generator | None = None,
) -> NDArray[np.complex128]:
    """Sample mixed-field Rician channels with a final antenna axis.

    Args:
        config: Valid centered ULA geometry and carrier configuration.
        range_m: 用户位置（距离和方位角）
        angle_rad: Finite real broadside-referenced azimuths in radians.
        k_factor_linear: Rician K 因子
        path_loss_linear: 路径损耗（线性增益）
        rng: 可选NumPy随机生成器，用于可重复的NLoS绘制

    Returns:
        形状为 (广播后的用户形状..., 天线数) 的复数信道矩阵，最后一维对应 ULA 阵元。
    """
    if not isinstance(config, ULAConfig):
        raise TypeError("config must be a ULAConfig instance.")
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator or None.")

    # This call validates ranges, angles, and their broadcasting convention.
    user_x_m, _ = polar_to_cartesian(range_m, angle_rad)
    ranges = np.asarray(range_m, dtype=np.float64)
    kappa = _finite_nonnegative_array(k_factor_linear, "k_factor_linear")
    beta = _finite_nonnegative_array(path_loss_linear, "path_loss_linear")
    ranges, user_x_m, kappa, beta = np.broadcast_arrays(ranges, user_x_m, kappa, beta)

    near_field = classify_near_field(ranges, config.rayleigh_distance_m)
    near_los = ula_near_field_los_steering_vector(config, ranges, angle_rad)
    far_los = ula_los_steering_vector(config, angle_rad)
    far_los = np.broadcast_to(far_los, near_los.shape)
    los = np.where(near_field[..., np.newaxis], near_los, far_los)

    generator = np.random.default_rng() if rng is None else rng
    nlos_shape = ranges.shape + (config.num_antennas,)
    nlos = (
        generator.standard_normal(nlos_shape)
        + 1j * generator.standard_normal(nlos_shape)
    ) / np.sqrt(2.0 * config.num_antennas)

    los_weight = np.sqrt(kappa / (1.0 + kappa))[..., np.newaxis]
    nlos_weight = np.sqrt(1.0 / (1.0 + kappa))[..., np.newaxis]
    path_gain = np.sqrt(beta)[..., np.newaxis]
    return np.asarray(path_gain * (los_weight * los + nlos_weight * nlos), dtype=np.complex128)
