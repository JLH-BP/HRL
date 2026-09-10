"""High-level partition and service-mode policies for staged HRL.

The Manager decision is deliberately represented independently of a Gym action
encoding. The environment owns the ``partition_index * 3 + service_mode``
encoding; policies return :class:`ManagerDecision` so the same policy can be
used by Worker pre-training, deterministic baselines, and Manager evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from hrl.grouping.candidate_groups import UserPartition, canonicalize_partition

if TYPE_CHECKING:
    from hrl.envs.task_sampler import RSMAScenario

ServiceMode: TypeAlias = Literal["all", "near", "far"]
SERVICE_MODES: tuple[ServiceMode, ...] = ("all", "near", "far")
ManagerSelection: TypeAlias = "ManagerDecision | int | tuple[int, ServiceMode | int]"

__all__ = [
    "ExplicitFixedPartitionManager",
    "FixedPartitionManager",
    "HeuristicPartitionManager",
    "ManagerDecision",
    "ManagerSelection",
    "NearFieldFirstSequentialManager",
    "RandomCompositeActionManager",
    "SERVICE_MODES",
    "ServiceMode",
    "coerce_manager_decision",
    "legal_service_modes",
    "normalize_service_mode",
    "select_partition_from_scores",
    "service_mask_for_mode",
]


def normalize_service_mode(service_mode: ServiceMode | str | int) -> ServiceMode:
    """Return a validated canonical service-mode name.

    Integer mode identifiers are accepted to make policies easy to compose with
    the environment's discrete Manager action codec: ``0=all``, ``1=near``,
    and ``2=far``.
    """
    if isinstance(service_mode, (bool, np.bool_)):
        raise TypeError("service_mode must be 'all', 'near', 'far', or its integer identifier.")
    if isinstance(service_mode, Integral):
        index = int(service_mode)
        if not 0 <= index < len(SERVICE_MODES):
            raise ValueError("service_mode integer identifier must be in [0, 2].")
        return SERVICE_MODES[index]
    if not isinstance(service_mode, str):
        raise TypeError("service_mode must be 'all', 'near', 'far', or its integer identifier.")
    normalized = service_mode.lower()
    if normalized not in SERVICE_MODES:
        raise ValueError("service_mode must be one of: all, near, far.")
    return normalized  # type: ignore[return-value]


def _validate_scenario(scenario: RSMAScenario) -> np.ndarray:
    """Validate and return the scenario's near-field membership vector."""
    from hrl.envs.task_sampler import RSMAScenario

    if not isinstance(scenario, RSMAScenario):
        raise TypeError("scenario must be an RSMAScenario instance.")
    near_mask = np.asarray(scenario.near_field_mask)
    if near_mask.ndim != 1 or near_mask.size < 1 or near_mask.dtype != np.bool_:
        raise ValueError("scenario.near_field_mask must be a nonempty one-dimensional boolean array.")
    return near_mask


def service_mask_for_mode(scenario: RSMAScenario, service_mode: ServiceMode | str | int) -> np.ndarray:
    """Return the users which are eligible for service under one mode."""
    near_mask = _validate_scenario(scenario)
    mode = normalize_service_mode(service_mode)
    if mode == "all":
        return np.ones(near_mask.size, dtype=np.bool_)
    if mode == "near":
        return near_mask.copy()
    return np.logical_not(near_mask)


def legal_service_modes(scenario: RSMAScenario) -> tuple[ServiceMode, ...]:
    """Return nonempty modes available for a sampled mixed-field scenario."""
    near_mask = _validate_scenario(scenario)
    modes: list[ServiceMode] = ["all"]
    if np.any(near_mask):
        modes.append("near")
    if np.any(~near_mask):
        modes.append("far")
    return tuple(modes)


@dataclass(frozen=True, slots=True)
class ManagerDecision:
    """A partition choice together with a transmission service mode.

    ``__index__`` intentionally preserves use of a decision in legacy code
    which indexes the candidate list directly. New code should use the
    explicit ``candidate_index`` and ``service_mode`` fields.
    """

    candidate_index: int
    partition: UserPartition
    service_mode: ServiceMode | str | int = "all"

    def __post_init__(self) -> None:
        if isinstance(self.candidate_index, (bool, np.bool_)) or not isinstance(self.candidate_index, Integral):
            raise TypeError("candidate_index must be an integer.")
        if int(self.candidate_index) < 0:
            raise ValueError("candidate_index must be nonnegative.")
        object.__setattr__(self, "candidate_index", int(self.candidate_index))
        object.__setattr__(self, "service_mode", normalize_service_mode(self.service_mode))

    def __index__(self) -> int:
        return self.candidate_index

    def __int__(self) -> int:
        return self.candidate_index


def select_partition_from_scores(scores: np.ndarray, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
    """Select the first maximum-scoring candidate under full-user service."""
    values = np.asarray(scores)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("scores must contain real numeric values.")
    vector = np.asarray(scores, dtype=np.float64)
    if not candidates or vector.shape != (len(candidates),) or not np.all(np.isfinite(vector)):
        raise ValueError("scores must be finite with one entry per candidate partition.")
    index = int(np.argmax(vector))
    return ManagerDecision(candidate_index=index, partition=candidates[index], service_mode="all")


def _candidate_index(partition: UserPartition, candidates: tuple[UserPartition, ...]) -> int:
    try:
        return candidates.index(partition)
    except ValueError as error:
        raise RuntimeError("The Manager selected a partition outside the configured candidate set.") from error


def _decision_for_index(
    candidate_index: int,
    candidates: tuple[UserPartition, ...],
    service_mode: ServiceMode | str | int,
) -> ManagerDecision:
    if isinstance(candidate_index, (bool, np.bool_)) or not isinstance(candidate_index, Integral):
        raise TypeError("candidate_index must be an integer.")
    index = int(candidate_index)
    if not 0 <= index < len(candidates):
        raise ValueError("candidate_index must be within the candidate partition set.")
    return ManagerDecision(index, candidates[index], normalize_service_mode(service_mode))


def coerce_manager_decision(
    selection: ManagerSelection,
    scenario: RSMAScenario,
    candidates: tuple[UserPartition, ...],
) -> ManagerDecision:
    """Normalize legacy and composite policy outputs to ``ManagerDecision``.

    An integer remains a legacy all-user partition selection. New fixed or
    learned policy adapters may return ``(partition_index, service_mode)``.
    This keeps previous custom Managers usable without silently assigning a
    near/far service mode to an old integer-only implementation.
    """
    _validate_scenario(scenario)
    if isinstance(selection, ManagerDecision):
        decision = _decision_for_index(selection.candidate_index, candidates, selection.service_mode)
        if decision.partition != selection.partition:
            raise ValueError("ManagerDecision partition does not match candidate_index.")
        return decision
    if isinstance(selection, tuple):
        if len(selection) != 2:
            raise ValueError("A composite Manager selection must be (partition_index, service_mode).")
        return _decision_for_index(selection[0], candidates, selection[1])
    return _decision_for_index(selection, candidates, "all")


@dataclass(frozen=True, slots=True)
class FixedPartitionManager:
    """Use one explicitly configured partition and service mode.

    Prefer ``partition=((0, 1), (2, 3), (4, 5))`` for the paper's fixed-group
    baseline. ``candidate_index`` remains for backwards compatibility, but the
    default index zero is deliberately only a legacy convenience and is not a
    valid definition of the fixed multi-user grouping baseline.
    """

    candidate_index: int | None = 0
    partition: UserPartition | None = None
    service_mode: ServiceMode | str | int = "all"

    def __post_init__(self) -> None:
        if self.partition is not None and self.candidate_index is not None:
            # The explicit partition is authoritative only when the old default
            # was not supplied. Requiring one source prevents ambiguous specs.
            if self.candidate_index != 0:
                raise ValueError("Specify either partition or candidate_index, not both.")
        if self.partition is None and self.candidate_index is None:
            raise ValueError("Specify a fixed partition or candidate_index.")
        normalize_service_mode(self.service_mode)

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
        near_mask = _validate_scenario(scenario)
        if not candidates:
            raise ValueError("candidates must contain at least one partition.")
        if self.partition is not None:
            partition = canonicalize_partition(self.partition, num_users=near_mask.size)
            index = _candidate_index(partition, candidates)
            return ManagerDecision(index, partition, self.service_mode)
        assert self.candidate_index is not None
        return _decision_for_index(self.candidate_index, candidates, self.service_mode)

    select_action = select


@dataclass(frozen=True, slots=True)
class ExplicitFixedPartitionManager:
    """Named fixed multi-user grouping Manager for main-experiment configs."""

    partition: UserPartition
    service_mode: ServiceMode | str | int = "all"

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
        return FixedPartitionManager(
            candidate_index=None, partition=self.partition, service_mode=self.service_mode
        ).select(scenario, candidates)

    select_action = select


@dataclass(frozen=True, slots=True)
class HeuristicPartitionManager:
    """Choose a CSI/geometry affinity partition while serving all users."""

    method: str = "hybrid"
    service_mode: ServiceMode | str | int = "all"

    def __post_init__(self) -> None:
        if normalize_service_mode(self.service_mode) != "all":
            raise ValueError("HeuristicPartitionManager is an all-user grouping baseline.")

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
        from hrl.grouping.heuristic import select_heuristic_partition

        _validate_scenario(scenario)
        result = select_heuristic_partition(
            ranges_m=scenario.ranges_m,
            angles_rad=scenario.angles_rad,
            channels=scenario.channels,
            ula_config=None,
            candidates=candidates,
            method=self.method,
        )
        index = _candidate_index(result.partition, candidates)
        return ManagerDecision(index, result.partition, "all")

    select_action = select


@dataclass(slots=True)
class NearFieldFirstSequentialManager:
    """Alternate actual near-only and far-only service intervals.

    The deterministic grouping keeps near-field users as singleton groups and
    places far-field users in one group. Crucially, the returned service mode
    is no longer merely a grouping label: ``near`` and ``far`` are consumed by
    ``HierarchicalRSMAEnv.service_mask``. ``reset(seed=...)`` starts every
    episode with a near interval and enables alternating thereafter.
    """

    _next_mode: ServiceMode = field(default="near", init=False, repr=False)
    _cycling_enabled: bool = field(default=False, init=False, repr=False)

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._next_mode = "near"
        self._cycling_enabled = True

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
        near_mask = _validate_scenario(scenario)
        if not candidates:
            raise ValueError("candidates must contain at least one partition.")
        near_users = tuple(int(user) for user in np.flatnonzero(near_mask))
        far_users = tuple(int(user) for user in np.flatnonzero(~near_mask))
        groups: list[tuple[int, ...]] = [(user,) for user in near_users]
        if far_users:
            groups.append(far_users)
        partition = canonicalize_partition(groups, num_users=near_mask.size)
        available = legal_service_modes(scenario)
        if "near" in available and "far" in available:
            mode = self._next_mode if self._cycling_enabled else "near"
            if self._cycling_enabled:
                self._next_mode = "far" if mode == "near" else "near"
        elif "near" in available:
            mode = "near"
        elif "far" in available:
            mode = "far"
        else:  # ``all`` is always legal, included for completeness.
            mode = "all"
        return ManagerDecision(_candidate_index(partition, candidates), partition, mode)

    select_action = select


@dataclass(slots=True)
class RandomCompositeActionManager:
    """Sample legal partition/service-mode pairs for Worker pre-training."""

    seed: int | None = None
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.seed, (bool, np.bool_)) or (self.seed is not None and not isinstance(self.seed, Integral)):
            raise TypeError("seed must be an integer or None.")
        self._rng = np.random.default_rng(self.seed)

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer or None.")
            self._rng = np.random.default_rng(int(seed))

    def select(self, scenario: RSMAScenario, candidates: tuple[UserPartition, ...]) -> ManagerDecision:
        _validate_scenario(scenario)
        if not candidates:
            raise ValueError("candidates must contain at least one partition.")
        index = int(self._rng.integers(len(candidates)))
        modes = legal_service_modes(scenario)
        mode = modes[int(self._rng.integers(len(modes)))]
        return ManagerDecision(index, candidates[index], mode)

    select_action = select
