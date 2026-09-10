# 实现二维ULA几何和近场/远场分类实用程序。
"""
ULA位于x轴上，且以原点(0,0)为中心。宽边是正y轴，极坐标用户(r,theta),theta为方位角（与正y轴的夹角）
x = r * sin(theta)、y = r * cos(theta)。公共API使用米、Hz和弧度。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

SPEED_OF_LIGHT_M_S: float = 299_792_458.0
DEFAULT_NUM_ANTENNAS: int = 128
DEFAULT_CARRIER_FREQUENCY_HZ: float = 28e9
"""
当使用 from 模块名 import * 时，
Python 会导入模块中所有不以下划线开头的名称（变量、函数、类等）
但如果模块中定义了 __all__，则只会导入 __all__ 列表中明确列出的名称。
"""
__all__ = [
    "DEFAULT_CARRIER_FREQUENCY_HZ",
    "DEFAULT_NUM_ANTENNAS",
    "SPEED_OF_LIGHT_M_S",
    "ULAConfig",
    "cartesian_to_polar",
    "classify_near_field",
    "element_to_user_distances_m",
    "polar_to_cartesian",
]


def _positive_finite_scalar(value: object, name: str) -> float:
    """验证并返回一个正的有限实值标量。"""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real-valued scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def _finite_real_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """将数字输入转换为有限的float64 NumPy数组。"""
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain real numeric values.")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numeric values.") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True, slots=True)
class ULAConfig:
    """描述SI单位下的中心化ULA及其Rayleigh距离几何结构。"""
    num_antennas: int = DEFAULT_NUM_ANTENNAS
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ
    element_spacing_m: float | None = None    # if None ,0.5 * wavelengths

    def __post_init__(self) -> None:
        """Validate physical array parameters at construction time."""
        if isinstance(self.num_antennas, (bool, np.bool_)) or not isinstance(
            self.num_antennas, Integral
        ):
            raise TypeError("num_antennas must be an integer.")
        if self.num_antennas < 2:
            raise ValueError("num_antennas must be at least two.")
        _positive_finite_scalar(self.carrier_frequency_hz, "carrier_frequency_hz")
        if self.element_spacing_m is not None:
            _positive_finite_scalar(self.element_spacing_m, "element_spacing_m")

    @property
    def wavelength_m(self) -> float:
        """返回载波波长（以米为单位）。"""
        return SPEED_OF_LIGHT_M_S / float(self.carrier_frequency_hz)  # c/f

    @property
    def resolved_element_spacing_m(self) -> float:
        """返回显式间隔或默认的半波长间隔。"""
        return self.wavelength_m / 2.0 if self.element_spacing_m is None else float(self.element_spacing_m)

    @property
    def aperture_m(self) -> float:
        """返回两个最外层ULA元素之间的距离。"""
        return (self.num_antennas - 1) * self.resolved_element_spacing_m

    @property
    def rayleigh_distance_m(self) -> float:
        """返回Rayleigh距离``2 * aperture**2 / wavelength``（以米为单位）。"""
        return 2.0 * self.aperture_m**2 / self.wavelength_m
    
    def element_positions_m(self) -> NDArray[np.float64]:
        """返回形状为（num_antennas，）的居中ULA元素x坐标。"""
        indices = np.arange(self.num_antennas, dtype=np.float64)
        return (indices - (self.num_antennas - 1) / 2.0) * self.resolved_element_spacing_m


def polar_to_cartesian(
    range_m: ArrayLike, angle_rad: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """将可广播的范围和方位角转换为x/y坐标。"""
    ranges = _finite_real_array(range_m, "range_m")
    angles = _finite_real_array(angle_rad, "angle_rad")
    if np.any(ranges <= 0.0):
        raise ValueError("range_m must contain values greater than zero.")
    ranges, angles = np.broadcast_arrays(ranges, angles)
    return ranges * np.sin(angles), ranges * np.cos(angles)


def cartesian_to_polar(
    x_m: ArrayLike, y_m: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """将可广播的x/y位置转换为范围和宽边参考方位角。"""
    x_values = _finite_real_array(x_m, "x_m")
    y_values = _finite_real_array(y_m, "y_m")
    x_values, y_values = np.broadcast_arrays(x_values, y_values)
    ranges = np.hypot(x_values, y_values)
    if np.any(ranges == 0.0):
        raise ValueError("Azimuth is undefined for a user at the array center.")
    return ranges, np.arctan2(x_values, y_values)


def element_to_user_distances_m(
    element_x_m: ArrayLike, user_x_m: ArrayLike, user_y_m: ArrayLike
) -> NDArray[np.float64]:
    """返回从ULA x坐标到一个或多个用户位置的精确距离。

    输出具有广播用户形状，后跟元素轴。例如，
    ``M`` elements and ``K`` user coordinates produce shape ``(K, M)``.
    """
    elements = _finite_real_array(element_x_m, "element_x_m")
    if elements.ndim != 1:
        raise ValueError("element_x_m must be a one-dimensional element-coordinate array.")
    x_values = _finite_real_array(user_x_m, "user_x_m")
    y_values = _finite_real_array(user_y_m, "user_y_m")
    x_values, y_values = np.broadcast_arrays(x_values, y_values)
    return np.hypot(x_values[..., np.newaxis] - elements, y_values[..., np.newaxis])


def classify_near_field(
    range_m: ArrayLike, rayleigh_distance_m: float
) -> NDArray[np.bool_]:
    """返回 ``range_m < rayleigh_distance_m`` 使用固定的边界约定."""
    ranges = _finite_real_array(range_m, "range_m")
    if np.any(ranges <= 0.0):
        raise ValueError("range_m must contain values greater than zero.")
    boundary = _positive_finite_scalar(rayleigh_distance_m, "rayleigh_distance_m")
    return ranges < boundary
