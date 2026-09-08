# Exposes candidate partitions and deterministic heuristic grouping APIs.
"""User grouping candidates and non-learning grouping strategies."""

from .candidate_groups import (
    DEFAULT_NUM_USERS,
    UserGroup,
    UserPartition,
    canonicalize_partition,
    enumerate_candidate_partitions,
)
from .heuristic import (
    GroupingWeights,
    HeuristicGroupingResult,
    pairwise_grouping_affinity,
    score_partition,
    select_heuristic_partition,
)

__all__ = [
    "DEFAULT_NUM_USERS",
    "GroupingWeights",
    "HeuristicGroupingResult",
    "UserGroup",
    "UserPartition",
    "canonicalize_partition",
    "enumerate_candidate_partitions",
    "pairwise_grouping_affinity",
    "score_partition",
    "select_heuristic_partition",
]
