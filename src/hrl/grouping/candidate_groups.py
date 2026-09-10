
"""
分组和分区类型别名：`UserGroup`、`UserPartition`
验证并规范化分区，消除组标签置换的影响。`canonicalize_partition()`
枚举满足约束的无标签候选用户分区：`enumerate_candidate_partitions()`

"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral

import numpy as np

DEFAULT_NUM_USERS: int = 6
UserGroup = tuple[int, ...]
UserPartition = tuple[UserGroup, ...]

__all__ = [
    "DEFAULT_NUM_USERS",
    "UserGroup",
    "UserPartition",
    "canonicalize_partition",
    "enumerate_candidate_partitions",
]


def _validate_partition_parameters(*,num_users: int,allow_singletons: bool,
                                   min_group_size: int,max_group_size: int | None,) -> tuple[int, int, int]:
    """Validate grouping constraints and return normalized size bounds."""
    if isinstance(num_users, (bool, np.bool_)) or not isinstance(num_users, Integral):
        raise TypeError("num_users must be an integer.")
    if num_users < 1:
        raise ValueError("num_users must be at least one.")
    if not isinstance(allow_singletons, (bool, np.bool_)):
        raise TypeError("allow_singletons must be a boolean.")
    if isinstance(min_group_size, (bool, np.bool_)) or not isinstance(min_group_size, Integral):
        raise TypeError("min_group_size must be an integer.")
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least one.")
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least one.")
    if max_group_size is None:
        effective_maximum = int(num_users)
    else:
        if isinstance(max_group_size, (bool, np.bool_)) or not isinstance(
            max_group_size, Integral
        ):
            raise TypeError("max_group_size must be an integer or None.")
        if max_group_size < min_group_size:
            raise ValueError("max_group_size must be at least min_group_size.")
        effective_maximum = min(int(max_group_size), int(num_users))
    effective_minimum = 1 if allow_singletons else max(2, int(min_group_size))
    if effective_minimum > num_users:
        return int(num_users), effective_minimum, effective_maximum
    return int(num_users), effective_minimum, effective_maximum


def canonicalize_partition(groups: Iterable[Iterable[int]],*,num_users: int = DEFAULT_NUM_USERS,
                           allow_singletons: bool = True,min_group_size: int = 1,max_group_size: int | None = None,) -> UserPartition:
    """验证并以规范元组形式返回一个完整的分区。"""
    num_users, minimum, maximum = _validate_partition_parameters(
        num_users=num_users,
        allow_singletons=allow_singletons,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
    )
    try:
        group_iterable = iter(groups)
    except TypeError as error:
        raise TypeError("groups must be an iterable of user-group iterables.") from error

    canonical_groups: list[UserGroup] = []
    assigned_users: list[int] = []
    for group in group_iterable:
        try:
            members = tuple(group)
        except TypeError as error:
            raise TypeError("each group must be an iterable of user identifiers.") from error
        if not members:
            raise ValueError("groups must not be empty.")
        for user in members:
            if isinstance(user, (bool, np.bool_)) or not isinstance(user, Integral):
                raise TypeError("user identifiers must be integers.")
            if not 0 <= user < num_users:
                raise ValueError("user identifiers must be within the configured user range.")
        normalized_group = tuple(sorted(int(user) for user in members))
        if len(set(normalized_group)) != len(normalized_group):
            raise ValueError("groups must not contain duplicate user identifiers.")
        if not minimum <= len(normalized_group) <= maximum:
            raise ValueError("group size violates the configured size constraints.")
        canonical_groups.append(normalized_group)
        assigned_users.extend(normalized_group)

    if not canonical_groups:
        raise ValueError("groups must contain at least one user group.")
    if len(set(assigned_users)) != len(assigned_users):
        raise ValueError("user identifiers must not appear in more than one group.")
    if set(assigned_users) != set(range(num_users)):
        raise ValueError("groups must contain every configured user exactly once.")
    return tuple(sorted(canonical_groups))


def _generate_partitions(num_users: int) -> Iterable[list[list[int]]]:
    """Yield each unlabeled set partition using restricted-growth recursion."""
    def recurse(next_user: int, groups: list[list[int]]) -> Iterable[list[list[int]]]:
        if next_user == num_users:
            yield [group.copy() for group in groups]
            return
        for index in range(len(groups)):
            groups[index].append(next_user)
            yield from recurse(next_user + 1, groups)
            groups[index].pop()
        groups.append([next_user])
        yield from recurse(next_user + 1, groups)
        groups.pop()

    yield from recurse(0, [])


def enumerate_candidate_partitions(*,num_users: int = DEFAULT_NUM_USERS,allow_singletons: bool = True,
                                   min_group_size: int = 1,max_group_size: int | None = None,) -> tuple[UserPartition, ...]:
    """以确定的顺序返回所有完整的规范化候选分区"""
    num_users, minimum, maximum = _validate_partition_parameters(
        num_users=num_users,
        allow_singletons=allow_singletons,
        min_group_size=min_group_size,
        max_group_size=max_group_size,
    )
    partitions: list[UserPartition] = []
    for groups in _generate_partitions(num_users):
        group_sizes = [len(group) for group in groups]
        if not all(minimum <= size <= maximum for size in group_sizes):
            continue
        partitions.append(
            canonicalize_partition(
                groups,
                num_users=num_users,
                allow_singletons=allow_singletons,
                min_group_size=min_group_size,
                max_group_size=max_group_size,
            )
        )
    return tuple(sorted(partitions))
