"""
    在每次 reset() 时随机采样一个混合近场/远场的 Rician 衰落信道实现，并提供所有相关的物理几何和信道参数
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from meta_hrl.channel.geometry import ULAConfig, classify_near_field
from meta_hrl.channel.rician import sample_rician_channel

__all__ = ["RSMAScenario", "ScenarioSamplerConfig", "TaskSampler"]   # Python 模块的公共接口声明。


def _positive_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real-valued scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def _closed_interval(value: tuple[float, float], name: str) -> tuple[float, float]:
    """用于验证一个由两个正数构成的区间,且下限小于上限"""
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple.")
    lower = _positive_scalar(value[0], f"{name}[0]")
    upper = _positive_scalar(value[1], f"{name}[1]")
    if lower > upper:
        raise ValueError(f"{name} lower bound must not exceed its upper bound.")
    return lower, upper


@dataclass(frozen=True, slots=True)
class ScenarioSamplerConfig:
    """场景采样范围与系统参数"""

    num_users: int = 6
    ula_config: ULAConfig = ULAConfig()  # ? 均匀线性阵列
    angle_range_rad: tuple[float, float] = (-np.pi / 3.0, np.pi / 3.0)
    near_range_m: tuple[float, float] = (10.0, 80.0)
    far_range_m: tuple[float, float] = (100.0, 250.0)
    near_field_user_count_range: tuple[int, int] = (0, 6)
    k_factor_linear: float = 10.0
    path_loss_exponent: float = 2.0                  # 路径损耗指数
    path_loss_reference_distance_m: float = 100.0    # 路径损耗参考距离
    qos_rate_targets: tuple[float, ...] = (0.5,) * 6 # 每个用户的 QoS 速率目标

    def __post_init__(self) -> None:
        """
        参数校验：
            num_users 为正整数。
            天线数 ≥ 2。
            角度范围合法。
            近场距离上限 < 瑞利距离 < 远场距离下限。
            近场用户数量范围在 [0, num_users] 内。
            K 因子、路径损耗指数等为正数。
            QoS 目标数组形状正确且非负。
        """
        if isinstance(self.num_users, (bool, np.bool_)) or not isinstance(self.num_users, Integral):
            raise TypeError("num_users must be an integer.")
        if self.num_users < 1:
            raise ValueError("num_users must be at least one.")
        if not isinstance(self.ula_config, ULAConfig):
            raise TypeError("ula_config must be a ULAConfig instance.")
        if self.ula_config.num_antennas < 2:
            raise ValueError("ula_config must have at least two antennas.")
        if not isinstance(self.angle_range_rad, tuple) or len(self.angle_range_rad) != 2:
            raise TypeError("angle_range_rad must be a two-item tuple.")
        angle_lower, angle_upper = (float(value) for value in self.angle_range_rad)
        if not np.isfinite(angle_lower) or not np.isfinite(angle_upper) or angle_lower > angle_upper:
            raise ValueError("angle_range_rad must be finite and ordered.")
        near_range = _closed_interval(self.near_range_m, "near_range_m")
        far_range = _closed_interval(self.far_range_m, "far_range_m")
        if near_range[1] >= self.ula_config.rayleigh_distance_m:
            raise ValueError("near_range_m must lie strictly below the Rayleigh distance.")
        if far_range[0] <= self.ula_config.rayleigh_distance_m:
            raise ValueError("far_range_m must lie strictly above the Rayleigh distance.")
        if (
            not isinstance(self.near_field_user_count_range, tuple)
            or len(self.near_field_user_count_range) != 2
        ):
            raise TypeError("near_field_user_count_range must be a two-item tuple.")
        near_min, near_max = self.near_field_user_count_range
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in (near_min, near_max)):
            raise TypeError("near_field_user_count_range entries must be integers.")
        if not 0 <= near_min <= near_max <= self.num_users:
            raise ValueError("near_field_user_count_range must be within [0, num_users].")
        _positive_scalar(self.k_factor_linear, "k_factor_linear")
        _positive_scalar(self.path_loss_exponent, "path_loss_exponent")
        _positive_scalar(self.path_loss_reference_distance_m, "path_loss_reference_distance_m")
        qos = np.asarray(self.qos_rate_targets, dtype=np.float64)
        if qos.shape != (self.num_users,) or not np.all(np.isfinite(qos)) or np.any(qos < 0.0):
            raise ValueError("qos_rate_targets must contain one finite nonnegative value per user.")


@dataclass(frozen=True, slots=True)
class RSMAScenario:
    """不可变的数据容器（frozen=True），存储一次采样得到的全部信息
    `一个静态场景：CSI、几何、路径损耗、K 因子和 QoS"""

    channels: NDArray[np.complex128]
    ranges_m: NDArray[np.float64]
    angles_rad: NDArray[np.float64]
    near_field_mask: NDArray[np.bool_]
    path_loss_linear: NDArray[np.float64]
    k_factors_linear: NDArray[np.float64]
    qos_rate_targets: NDArray[np.float64]


class TaskSampler:
    """以独立 RNG 生成可复现混合近远场场景"""

    def __init__(self, config: ScenarioSamplerConfig = ScenarioSamplerConfig()) -> None:
        if not isinstance(config, ScenarioSamplerConfig):
            raise TypeError("config must be a ScenarioSamplerConfig instance.")
        self.config = config
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        """重新初始化随机数生成器，允许用户指定种子以获得可重复的采样结果"""
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer or None.")
            self._rng = np.random.default_rng(int(seed))

    def sample(self) -> RSMAScenario:
        """随机采样一个 Rician 衰落信道场景，返回所有相关的物理几何和信道参数"""
        config = self.config
        near_min, near_max = config.near_field_user_count_range
        near_count = int(self._rng.integers(near_min, near_max + 1))  # 随机选择近场用户数量
        far_count = config.num_users - near_count
        ranges = np.concatenate(
            (
                self._rng.uniform(*config.near_range_m, size=near_count),
                self._rng.uniform(*config.far_range_m, size=far_count),
            )
        )
        self._rng.shuffle(ranges)
        angles = self._rng.uniform(*config.angle_range_rad, size=config.num_users)
        near_mask = classify_near_field(ranges, config.ula_config.rayleigh_distance_m)  # 识别近场
        path_gains = (config.path_loss_reference_distance_m / ranges) ** config.path_loss_exponent  # 计算路径损耗
        k_factors = np.full(config.num_users, config.k_factor_linear, dtype=np.float64)             # 初始化 K 因子
        channels = sample_rician_channel(
            config.ula_config,
            ranges,
            angles,
            k_factors,
            path_gains,
            rng=self._rng,
        )
        return RSMAScenario(
            channels=channels,
            ranges_m=np.asarray(ranges, dtype=np.float64),
            angles_rad=np.asarray(angles, dtype=np.float64),
            near_field_mask=np.asarray(near_mask, dtype=np.bool_),
            path_loss_linear=np.asarray(path_gains, dtype=np.float64),
            k_factors_linear=k_factors,
            qos_rate_targets=np.asarray(config.qos_rate_targets, dtype=np.float64),
        )
