# Implements fixed-slot group-common RSMA resources, beams, and rates.
"""Group-common RSMA physical layer for canonical user partitions.

Every non-singleton group owns one public stream placed in the slot indexed by
its smallest user identifier. This deterministic sparse representation keeps
resource actions fixed at ``3K`` entries despite changes in partition topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hrl.grouping.candidate_groups import UserPartition, canonicalize_partition
from .precoding import mrt_private_directions, rzf_private_directions

__all__ = [
    "GroupRSMAAllocation",
    "GroupRSMARates",
    "active_group_slots",
    "normalize_service_mask",
    "service_masked_partition",
    "group_rsma_precoders",
    "group_rsma_rates",
    "project_group_action",
]


def _vector(value: ArrayLike, name: str, size: int, *, nonnegative: bool = False) -> NDArray[np.float64]:
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain real numeric values.")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}.")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real-valued.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def normalize_service_mask(
    service_mask: ArrayLike | None, *, num_users: int
) -> NDArray[np.bool_]:
    """Return a validated boolean scheduled-user mask.

    ``None`` denotes the legacy all-user service case.  Keeping this validation
    beside the resource projection prevents a caller from accidentally treating
    numeric resource weights as a scheduling mask.
    """
    if service_mask is None:
        return np.ones(num_users, dtype=np.bool_)
    raw = np.asarray(service_mask)
    if raw.shape != (num_users,) or raw.dtype != np.bool_:
        raise ValueError(f"service_mask must be a boolean vector of length {num_users}.")
    return np.asarray(raw, dtype=np.bool_)


def service_masked_partition(
    partition: UserPartition, *, service_mask: ArrayLike | None, num_users: int
) -> UserPartition:
    """Make unscheduled users singleton while retaining active groups only.

    A group-common stream may therefore only ever serve users selected by the
    current scheduling mask.  This also handles a group's original minimum
    member being unscheduled: the active subgroup receives its own deterministic
    fixed slot through canonicalization.
    """
    canonical = canonicalize_partition(partition, num_users=num_users)
    mask = normalize_service_mask(service_mask, num_users=num_users)
    groups: list[tuple[int, ...]] = []
    for group in canonical:
        active_members = tuple(user for user in group if mask[user])
        if active_members:
            groups.append(active_members)
        groups.extend((user,) for user in group if not mask[user])
    return canonicalize_partition(groups, num_users=num_users)


def active_group_slots(
    partition: UserPartition,
    *,
    num_users: int,
    service_mask: ArrayLike | None = None,
) -> dict[int, tuple[int, ...]]:
    """Map each scheduled non-singleton group to its minimum-user slot."""
    canonical = service_masked_partition(
        partition, service_mask=service_mask, num_users=num_users
    )
    return {group[0]: group for group in canonical if len(group) > 1}


@dataclass(frozen=True, slots=True)
class GroupRSMAAllocation:
    """Private and fixed-slot group-common resources for one canonical partition."""

    private_powers: NDArray[np.float64]
    group_common_powers: NDArray[np.float64]
    common_rate_allocations: NDArray[np.float64]

    def __post_init__(self) -> None:
        users = len(self.private_powers)
        private = _vector(self.private_powers, "private_powers", users, nonnegative=True)
        group_powers = _vector(self.group_common_powers, "group_common_powers", users, nonnegative=True)
        common_rates = _vector(self.common_rate_allocations, "common_rate_allocations", users, nonnegative=True)
        object.__setattr__(self, "private_powers", private)
        object.__setattr__(self, "group_common_powers", group_powers)
        object.__setattr__(self, "common_rate_allocations", common_rates)


def project_group_action(
    action: ArrayLike,
    *,
    partition: UserPartition,
    num_users: int,
    total_power: float,
    group_rate_budgets: ArrayLike | None = None,
    service_mask: ArrayLike | None = None,
) -> GroupRSMAAllocation:
    """Project fixed ``3K`` logits into power and per-group common-rate resources.

    Entries are ``[K private-power, K group-power, K user common-rate logits]``.
    Inactive group slots, singleton users, and unscheduled users receive zero
    group-common resources.  If no user is scheduled, the transmitter idles and
    the projected allocation is identically zero.
    """
    total_power = _positive(total_power, "total_power")
    logits = _vector(action, "action", 3 * num_users)
    mask = normalize_service_mask(service_mask, num_users=num_users)
    slots = active_group_slots(partition, num_users=num_users, service_mask=mask)
    active_power_indices = list(np.flatnonzero(mask)) + [num_users + slot for slot in slots]
    power = np.zeros(2 * num_users, dtype=np.float64)
    if active_power_indices:
        power[active_power_indices] = total_power * _softmax(logits[active_power_indices])
    rate_allocations = np.zeros(num_users, dtype=np.float64)
    if group_rate_budgets is not None:
        budgets = _vector(group_rate_budgets, "group_rate_budgets", num_users, nonnegative=True)
        for slot, members in slots.items():
            member_indices = np.asarray(members, dtype=np.int64)
            rate_allocations[member_indices] = budgets[slot] * _softmax(logits[2 * num_users + member_indices])
    return GroupRSMAAllocation(power[:num_users], power[num_users:], rate_allocations)


def group_rsma_precoders(
    channels: ArrayLike,
    allocation: GroupRSMAAllocation,
    partition: UserPartition,
    *,
    private_scheme: str = "rzf",
    service_mask: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Build columns ``[K private streams, K fixed-slot group-common streams]``."""
    matrix = np.asarray(channels, dtype=np.complex128)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(np.linalg.norm(matrix, axis=1) == 0.0):
        raise ValueError("channels must be a finite nonzero (users, antennas) matrix.")
    users, antennas = matrix.shape
    if allocation.private_powers.shape != (users,):
        raise ValueError("allocation dimensions must match channels.")
    mask = normalize_service_mask(service_mask, num_users=users)
    slots = active_group_slots(partition, num_users=users, service_mask=mask)
    if private_scheme == "mrt":
        private = mrt_private_directions(matrix)
    elif private_scheme == "rzf":
        private = rzf_private_directions(matrix)
    else:
        raise ValueError("private_scheme must be either 'mrt' or 'rzf'.")
    precoders = np.zeros((antennas, 2 * users), dtype=np.complex128)
    private_powers = np.where(mask, allocation.private_powers, 0.0)
    precoders[:, :users] = private * np.sqrt(private_powers)[np.newaxis, :]
    for slot, members in slots.items():
        direction = np.sum(private[:, members], axis=1)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("Group common beam direction unexpectedly cancels.")
        precoders[:, users + slot] = np.sqrt(allocation.group_common_powers[slot]) * direction / norm
    return precoders


@dataclass(frozen=True, slots=True)
class GroupRSMARates:
    """Rates and SINRs for private streams and fixed-slot group-common streams."""

    group_common_rates: NDArray[np.float64]
    group_common_sinrs: NDArray[np.float64]
    private_sinrs: NDArray[np.float64]
    private_rates: NDArray[np.float64]
    user_rates: NDArray[np.float64]
    sum_rate: float


def group_rsma_rates(
    channels: ArrayLike,
    precoders: ArrayLike,
    allocation: GroupRSMAAllocation,
    partition: UserPartition,
    *,
    noise_variance: float,
    service_mask: ArrayLike | None = None,
) -> GroupRSMARates:
    """Compute group-common bottlenecks and post-SIC private rates.

    A group member decodes only its own group-common stream. Other group-common
    streams and every private stream are interference at that stage; after SIC,
    all group-common streams are removed before private decoding.
    """
    matrix = np.asarray(channels, dtype=np.complex128)
    beams = np.asarray(precoders, dtype=np.complex128)
    if matrix.ndim != 2 or beams.shape != (matrix.shape[1], 2 * matrix.shape[0]):
        raise ValueError("channels and precoders have incompatible group-RSMA shapes.")
    noise = _positive(noise_variance, "noise_variance")
    users = matrix.shape[0]
    mask = normalize_service_mask(service_mask, num_users=users)
    slots = active_group_slots(partition, num_users=users, service_mask=mask)
    # Rate evaluation enforces the scheduling contract even if a caller hands
    # in an externally-built beam matrix instead of group_rsma_precoders().
    masked_beams = beams.copy()
    masked_beams[:, np.flatnonzero(~mask)] = 0.0
    active_group_columns = np.zeros(users, dtype=np.bool_)
    active_group_columns[list(slots)] = True
    masked_beams[:, users + np.flatnonzero(~active_group_columns)] = 0.0
    received = np.abs(matrix @ masked_beams) ** 2
    private_received = received[:, :users]
    group_received = received[:, users:]
    private_signal = np.diag(private_received)
    private_interference = np.sum(private_received, axis=1) - private_signal
    private_sinrs = private_signal / (private_interference + noise)
    private_rates = np.log2(1.0 + private_sinrs)
    private_sinrs[~mask] = 0.0
    private_rates[~mask] = 0.0
    group_rates = np.zeros(users, dtype=np.float64)
    group_sinrs = np.zeros(users, dtype=np.float64)
    common_rate_users = np.zeros(users, dtype=np.bool_)
    for slot, members in slots.items():
        member_indices = np.asarray(members, dtype=np.int64)
        common_rate_users[member_indices] = True
        signal = group_received[member_indices, slot]
        interference = np.sum(private_received[member_indices], axis=1) + np.sum(group_received[member_indices], axis=1) - signal
        sinrs = signal / (interference + noise)
        group_sinrs[slot] = float(np.min(sinrs))
        group_rates[slot] = float(np.min(np.log2(1.0 + sinrs)))
        if not np.isclose(np.sum(allocation.common_rate_allocations[member_indices]), group_rates[slot], atol=1e-8):
            raise ValueError("Each active group's common-rate allocations must equal its physical bottleneck rate.")
    # A caller cannot manufacture a common rate for a scheduled singleton: only
    # members of an active non-singleton group decode a group-common stream.
    common_allocations = np.where(
        common_rate_users, allocation.common_rate_allocations, 0.0
    )
    user_rates = private_rates + common_allocations
    return GroupRSMARates(
        group_rates,
        group_sinrs,
        private_sinrs,
        private_rates,
        user_rates,
        float(np.sum(user_rates)),
    )
