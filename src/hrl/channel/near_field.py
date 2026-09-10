# 实现了只有相位的球波ULA LoS导向矢量。
"""
近场导向矢量
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import (
    ULAConfig,
    element_to_user_distances_m,
    polar_to_cartesian,
)

__all__ = ["ula_near_field_los_steering_vector"]


def ula_near_field_los_steering_vector(
    config: ULAConfig, range_m: ArrayLike, angle_rad: ArrayLike
) -> NDArray[np.complex128]:
    """返回一个或多个用户的归一化球波LoS矢量。

    Args:
        config: 有效的中心化ULA几何结构和载波配置。
        range_m: 有限的正中心到用户距离或可广播的范围数组，单位为米。
        angle_rad: 有限的实数宽边参考方位角或可广播的方位角数组，单位为弧度。

    Returns:
        具有形状的复相位矢量
    """
    user_x_m, user_y_m = polar_to_cartesian(range_m, angle_rad)
    ranges, user_x_m = np.broadcast_arrays(np.asarray(range_m, dtype=np.float64), user_x_m)
    distances_m = element_to_user_distances_m(
        config.element_positions_m(), user_x_m, user_y_m
    )
    elements_m = config.element_positions_m()

    path_offsets_m = (
        elements_m**2 - 2.0 * user_x_m[..., np.newaxis] * elements_m
    ) / (distances_m + ranges[..., np.newaxis])
    wavenumber = 2.0 * np.pi / config.wavelength_m   # * d_m-r =（d_m^2-r^2）/(d_m+r) 
    return np.asarray(
        np.exp(-1j * wavenumber * path_offsets_m) / np.sqrt(config.num_antennas),
        dtype=np.complex128,
    )
