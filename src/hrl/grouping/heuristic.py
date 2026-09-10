# 实现了混合几何/信道相关性启发式用户分组。
"""组合几何、近远场标签和信道相关性的亲和度"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from meta_hrl.channel.geometry import ULAConfig, classify_near_field
from meta_hrl.channel.validation import user_channel_correlation_matrix

from .candidate_groups import (
    DEFAULT_NUM_USERS,
    UserPartition,
    canonicalize_partition,
    enumerate_candidate_partitions,
)

__all__ = [
    "GroupingWeights",
    "HeuristicGroupingResult",
    "pairwise_grouping_affinity",
    "score_partition",
    "select_heuristic_partition",
]


@dataclass(frozen=True, slots=True)
class GroupingWeights:
    """信道、角度、范围、近/远场亲成度策等的权重。"""

    channel_correlation: float = 0.50
    angle_similarity: float = 0.25
    range_similarity: float = 0.15
    same_field_bonus: float = 0.10

    def __post_init__(self) -> None:
        """验证权重为非负实数，且至少有一个为正。"""
        values = (
            self.channel_correlation,
            self.angle_similarity,
            self.range_similarity,
            self.same_field_bonus,
        )
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
            raise TypeError("grouping weights must be real-valued.")
        if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("grouping weights must be finite and nonnegative.")
        if sum(float(value) for value in values) <= 0.0:
            raise ValueError("at least one grouping weight must be positive.")

    def as_array(self) -> NDArray[np.float64]:
        """"""
        return np.asarray(
            [
                self.channel_correlation,
                self.angle_similarity,
                self.range_similarity,
                self.same_field_bonus,
            ],
            dtype=np.float64,
        )


def _finite_vector(value: ArrayLike, name: str, *, positive: bool = False) -> NDArray[np.float64]:
    """验证并返回一个有限实数向量。"""
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
    if positive and np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain values greater than zero.")
    return vector


def _gaussian_similarity(difference: NDArray[np.float64], scale: float) -> NDArray[np.float64]:
    """将绝对差转换为Gaussian相似性。"""
    return np.exp(-0.5 * (difference / scale) ** 2)


def _geometry_components(
    ranges: NDArray[np.float64], angles: NDArray[np.float64], config: ULAConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """构建角度、范围、同一場葉、近场组件箱阵。"""
    near_mask = classify_near_field(ranges, config.rayleigh_distance_m)
    angle_difference = np.abs((angles[:, None] - angles[None, :] + np.pi) % (2.0 * np.pi) - np.pi)
    angle_component = _gaussian_similarity(angle_difference, np.deg2rad(10.0))
    same_field = (near_mask[:, None] == near_mask[None, :]).astype(np.float64)
    range_difference = np.abs(ranges[:, None] - ranges[None, :])
    range_component = np.ones((ranges.size, ranges.size), dtype=np.float64)
    near_pair = near_mask[:, None] & near_mask[None, :]
    range_component[near_pair] = _gaussian_similarity(
        range_difference[near_pair], 0.25 * config.rayleigh_distance_m
    )
    mixed_pair = near_mask[:, None] != near_mask[None, :]
    range_component[mixed_pair] = 0.0
    for component in (angle_component, range_component, same_field):
        np.fill_diagonal(component, 0.0)
    return angle_component, range_component, same_field, near_mask


def pairwise_grouping_affinity(
    *,
    ranges_m: ArrayLike,
    angles_rad: ArrayLike,
    channels: ArrayLike | None = None,
    ula_config: ULAConfig | None = None,
    method: str = "hybrid",
    weights: GroupingWeights = GroupingWeights(),
) -> NDArray[np.float64]:
    """返回一个对称零对角线性成对亲成度矩阵。"""
    if method not in {"hybrid", "geometry", "channel"}:
        raise ValueError("method must be 'hybrid', 'geometry', or 'channel'.")
    ranges = _finite_vector(ranges_m, "ranges_m", positive=True)
    angles = _finite_vector(angles_rad, "angles_rad")
    if ranges.shape != angles.shape:
        raise ValueError("ranges_m and angles_rad must have matching shapes.")
    config = ULAConfig() if ula_config is None else ula_config
    if not isinstance(config, ULAConfig):
        raise TypeError("ula_config must be a ULAConfig instance or None.")
    angle_component, range_component, field_component, _ = _geometry_components(ranges, angles, config)
    components = [np.zeros((ranges.size, ranges.size), dtype=np.float64), angle_component, range_component, field_component]
    enabled = np.array([False, True, True, True])
    if channels is not None and method in {"hybrid", "channel"}:
        channel_matrix = np.asarray(channels)
        if channel_matrix.ndim != 2 or channel_matrix.shape[0] != ranges.size:
            raise ValueError("channels must have shape (num_users, num_antennas).")
        components[0] = np.abs(user_channel_correlation_matrix(channel_matrix)).astype(np.float64)
        np.fill_diagonal(components[0], 0.0)
        enabled[0] = True
    elif method == "channel":
        raise ValueError("channels are required for method='channel'.")
    if method == "geometry":
        enabled[:] = False
        enabled[1:] = True
    if method == "hybrid" and channels is None:
        enabled[0] = False
    if method == "channel":
        enabled[:] = False
        enabled[0] = True
    active_weights = weights.as_array() * enabled
    if np.sum(active_weights) <= 0.0:
        raise ValueError("the selected grouping method has no active affinity component.")
    normalized_weights = active_weights / np.sum(active_weights)
    affinity = sum(weight * component for weight, component in zip(normalized_weights, components))
    affinity = np.asarray(np.clip((affinity + affinity.T) / 2.0, 0.0, 1.0), dtype=np.float64)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def score_partition(
    partition: UserPartition,
    affinity: ArrayLike,
    *,
    num_users: int | None = None,
    singleton_score: float = 0.0,
    partition_count_penalty: float = 0.02,
) -> float:
    """使用规范化组内亲成度对可侵分区进行评分。"""
    matrix = np.asarray(affinity)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("affinity must be a nonempty square matrix.")
    if np.iscomplexobj(matrix) or matrix.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("affinity must contain real numeric values.")
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("affinity must contain only finite values.")
    if num_users is None:
        num_users = matrix.shape[0]
    if num_users != matrix.shape[0]:
        raise ValueError("num_users must match affinity dimensions.")
    if not isinstance(singleton_score, Real) or not np.isfinite(float(singleton_score)):
        raise ValueError("singleton_score must be finite.")
    if not isinstance(partition_count_penalty, Real) or not np.isfinite(float(partition_count_penalty)) or partition_count_penalty < 0.0:
        raise ValueError("partition_count_penalty must be finite and nonnegative.")
    canonical = canonicalize_partition(partition, num_users=num_users, allow_singletons=True)
    score = 0.0
    for group in canonical:
        if len(group) == 1:
            score += float(singleton_score)
            continue
        pairs = [matrix[i, j] for offset, i in enumerate(group) for j in group[offset + 1 :]]
        score += float(np.sum(pairs) / max(1, len(group) - 1))
    return score - float(partition_count_penalty) * (len(canonical) - 1)


@dataclass(frozen=True, slots=True)
class HeuristicGroupingResult:
    """存储选择的分区与所有亲成度诊断新息。"""

    partition: UserPartition
    score: float
    candidates: tuple[UserPartition, ...]
    candidate_scores: tuple[float, ...]
    pairwise_affinity: NDArray[np.float64]
    near_field_mask: NDArray[np.bool_]


def select_heuristic_partition(
    *,
    ranges_m: ArrayLike,
    angles_rad: ArrayLike,
    channels: ArrayLike | None = None,
    ula_config: ULAConfig | None = None,
    candidates: list[UserPartition] | tuple[UserPartition, ...] | None = None,
    allow_singletons: bool = True,
    min_group_size: int = 1,
    max_group_size: int | None = None,
    method: str = "hybrid",
    weights: GroupingWeights = GroupingWeights(),
    singleton_score: float = 0.0,
    partition_count_penalty: float = 0.02,
) -> HeuristicGroupingResult:
    """确定性地选择最高评传规范化可侵。"""
    ranges = _finite_vector(ranges_m, "ranges_m", positive=True)
    angles = _finite_vector(angles_rad, "angles_rad")
    if ranges.shape != angles.shape:
        raise ValueError("ranges_m and angles_rad must have matching shapes.")
    config = ULAConfig() if ula_config is None else ula_config
    affinity = pairwise_grouping_affinity(
        ranges_m=ranges,
        angles_rad=angles,
        channels=channels,
        ula_config=config,
        method=method,
        weights=weights,
    )
    near_mask = classify_near_field(ranges, config.rayleigh_distance_m)
    if candidates is None:
        candidate_tuple = enumerate_candidate_partitions(
            num_users=ranges.size,
            allow_singletons=allow_singletons,
            min_group_size=min_group_size,
            max_group_size=max_group_size,
        )
    else:
        normalized = {
            canonicalize_partition(
                candidate,
                num_users=ranges.size,
                allow_singletons=allow_singletons,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
            )
            for candidate in candidates
        }
        if not normalized:
            raise ValueError("candidates must contain at least one valid partition.")
        candidate_tuple = tuple(sorted(normalized))
    scores = tuple(
        score_partition(
            candidate,
            affinity,
            num_users=ranges.size,
            singleton_score=singleton_score,
            partition_count_penalty=partition_count_penalty,
        )
        for candidate in candidate_tuple
    )
    best_index = max(range(len(candidate_tuple)), key=lambda index: (scores[index], tuple(-value for group in candidate_tuple[index] for value in group)))
    best_score = scores[best_index]
    tied = [candidate_tuple[index] for index, score in enumerate(scores) if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
    best_partition = min(tied)
    return HeuristicGroupingResult(
        partition=best_partition,
        score=best_score,
        candidates=candidate_tuple,
        candidate_scores=scores,
        pairwise_affinity=affinity,
        near_field_mask=near_mask,
    )
