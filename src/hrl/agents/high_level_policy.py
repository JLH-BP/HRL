# 实现了预置和启发式非学习Manager，用于戆阶段组RSMA HRL。
"""Manager基线，在训练组RSMA工作者策略时使用。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from hrl.grouping.candidate_groups import UserPartition, canonicalize_partition

if TYPE_CHECKING:
    from hrl.envs.task_sampler import RSMAScenario

__all__ = [
    "FixedPartitionManager",
    "HeuristicPartitionManager",
    "ManagerDecision",
    "NearFieldFirstSequentialManager",
    "select_partition_from_scores",
]


@dataclass(frozen=True, slots=True)
class ManagerDecision:
    """一个Manager选择，由稳定候选重历元索与分区表示。"""

    candidate_index: int
    partition: UserPartition


def select_partition_from_scores(scores: np.ndarray, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
    """选择最大的有限分数与确定性最低索与重抹使。"""
    values = np.asarray(scores)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("scores must contain real numeric values.")
    vector = np.asarray(scores, dtype=np.float64)
    if not candidates or vector.shape != (len(candidates),) or not np.all(np.isfinite(vector)):
        raise ValueError("scores must be finite with one entry per candidate partition.")
    index = int(np.argmax(vector))
    return ManagerDecision(candidate_index=index, partition=candidates[index])


def _candidate_index(partition: UserPartition, candidates: tuple[UserPartition, ...]) -> int:
    try:
        return candidates.index(partition)
    except ValueError as error:
        raise RuntimeError("管理器选择的分区不在配置的候选集合中。") from error


@dataclass(frozen=True, slots=True)
class FixedPartitionManager:
    """始终返回一个预配置的候选分区索下。"""

    candidate_index: int = 0

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> int:
        """验证场景并验证候选边界后返回固定索下。"""
        from hrl.envs.task_sampler import RSMAScenario

        if not isinstance(scenario, RSMAScenario):
            raise TypeError("scenario must be an RSMAScenario instance.")
        if isinstance(self.candidate_index, (bool, np.bool_)) or not isinstance(self.candidate_index, Integral):
            raise TypeError("candidate_index must be an integer.")
        if not 0 <= self.candidate_index < len(candidates):
            raise ValueError("candidate_index must be within the candidate partition set.")
        return int(self.candidate_index)


@dataclass(frozen=True, slots=True)
class HeuristicPartitionManager:
    """使用混合CSI与几何传输亲和度选择候选。"""

    method: str = "hybrid"

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> int:
        """映射确定性启发式分区转为其候选动作索下。"""
        from hrl.envs.task_sampler import RSMAScenario
        from hrl.grouping.heuristic import select_heuristic_partition

        if not isinstance(scenario, RSMAScenario):
            raise TypeError("scenario must be an RSMAScenario instance.")
        result = select_heuristic_partition(
            ranges_m=scenario.ranges_m,
            angles_rad=scenario.angles_rad,
            channels=scenario.channels,
            ula_config=None,
            candidates=candidates,
            method=self.method,
        )
        return _candidate_index(result.partition, candidates)


@dataclass(frozen=True, slots=True)
class NearFieldFirstSequentialManager:
    """Select a deterministic near-field-first sequential grouping baseline.

    Near-field users are handled first and retained as singleton groups.  The
    remaining far-field users are then placed in one group.  This produces a
    legal partition for every near/far composition while keeping the rule
    distinct from the CSI/geometry affinity heuristic.  It is a partition
    baseline only; it does not change the physical-layer transmission order.
    """

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> int:
        """Map the field classification to a canonical candidate partition."""
        from hrl.envs.task_sampler import RSMAScenario

        if not isinstance(scenario, RSMAScenario):
            raise TypeError("scenario must be an RSMAScenario instance.")
        if not candidates:
            raise ValueError("candidates must contain at least one partition.")

        near_mask = np.asarray(scenario.near_field_mask)
        num_users = near_mask.size
        if near_mask.shape != (num_users,) or near_mask.dtype != np.bool_ or num_users < 1:
            raise ValueError("scenario.near_field_mask must be a nonempty one-dimensional boolean array.")

        near_users = tuple(int(user) for user in np.flatnonzero(near_mask))
        far_users = tuple(int(user) for user in np.flatnonzero(~near_mask))
        groups: list[tuple[int, ...]] = [(user,) for user in near_users]
        if far_users:
            groups.append(far_users)
        partition = canonicalize_partition(groups, num_users=num_users)
        return _candidate_index(partition, candidates)
