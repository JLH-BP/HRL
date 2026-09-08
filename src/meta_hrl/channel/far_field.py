# 实现标准化远场ULA LoS导向矢量。
"""
远场平面波导向矢量。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import ULAConfig

__all__ = ["ula_los_steering_vector"]


def _finite_real_angles(angle_rad: ArrayLike) -> NDArray[np.float64]:
    """Convert angle input to a finite float64 array."""
    values = np.asarray(angle_rad)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("angle_rad must contain real numeric values.")
    if values.ndim == 0 and isinstance(angle_rad, (bool, np.bool_)):
        raise TypeError("angle_rad must contain real numeric values.")
    try:
        result = np.asarray(angle_rad, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("angle_rad must contain real numeric values.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("angle_rad must contain only finite values.")
    return result


def ula_los_steering_vector(
    config: ULAConfig, angle_rad: ArrayLike
) -> NDArray[np.complex128]:
    """返回标准化远场LoS矢量用于宽边参考角度。

    Args:
        config: 有效的中心化ULA几何结构和载波配置。
        angle_rad: 有限的实数角度或角度数组，单位为弧度，零点指向宽边（正y轴）。

    Returns:
        具有形状的复相位矢量
    """
    angles = _finite_real_angles(angle_rad)
    positions = config.element_positions_m()
    wavenumber = 2.0 * np.pi / config.wavelength_m
    phase = wavenumber * np.sin(angles)[..., np.newaxis] * positions
    return np.asarray(np.exp(1j * phase) / np.sqrt(config.num_antennas), dtype=np.complex128)  # 除以 np.sqrt(config.num_antennas) 完成归一化
