# Tests canonical user partitions and deterministic candidate enumeration.
from collections import Counter

import numpy as np
import pytest

from hrl.grouping.candidate_groups import (
    canonicalize_partition,
    enumerate_candidate_partitions,
)


def _size_signature(partition: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return tuple(sorted(map(len, partition), reverse=True))


def test_default_six_user_enumeration_has_all_203_unique_partitions() -> None:
    partitions = enumerate_candidate_partitions()

    assert len(partitions) == 203
    assert len(set(partitions)) == 203
    assert partitions == enumerate_candidate_partitions()
    assert all(
        set(user for group in partition for user in group) == set(range(6))
        for partition in partitions
    )
    assert all(group == tuple(sorted(group)) for partition in partitions for group in partition)
    assert all(partition == tuple(sorted(partition)) for partition in partitions)


def test_default_candidates_cover_expected_group_size_signatures() -> None:
    signatures = Counter(_size_signature(partition) for partition in enumerate_candidate_partitions())

    assert signatures[(6,)] == 1
    assert signatures[(4, 2)] == 15
    assert signatures[(3, 3)] == 10
    assert signatures[(2, 2, 2)] == 15
    assert signatures[(5, 1)] == 6
    assert signatures[(2, 2, 1, 1)] == 45
    assert signatures[(1, 1, 1, 1, 1, 1)] == 1


def test_canonicalization_removes_group_and_member_order() -> None:
    first = canonicalize_partition(((3, 1), (5, 4, 2, 0)))
    second = canonicalize_partition(((0, 2, 4, 5), (1, 3)))

    assert first == ((0, 2, 4, 5), (1, 3))
    assert first == second


def test_without_singletons_returns_41_partitions() -> None:
    partitions = enumerate_candidate_partitions(
        allow_singletons=False, min_group_size=2
    )

    assert len(partitions) == 41
    assert all(len(group) >= 2 for partition in partitions for group in partition)


def test_small_user_counts_and_generator_input() -> None:
    assert enumerate_candidate_partitions(num_users=1) == (((0,),),)
    assert enumerate_candidate_partitions(num_users=1, allow_singletons=False, min_group_size=2) == ()
    assert enumerate_candidate_partitions(num_users=2, allow_singletons=False, min_group_size=2) == (((0, 1),),)
    generated = canonicalize_partition((group for group in ((1,), (0,))), num_users=2)
    assert generated == ((0,), (1,))


@pytest.mark.parametrize(
    "groups",
    [
        ((0, 1),),
        ((0, 1, 2), (3, 4)),
        ((), (0, 1, 2, 3, 4, 5)),
        ((0, 0), (1, 2, 3, 4, 5)),
        ((0, 1), (1, 2, 3, 4, 5)),
        ((0, 1), (2, 3, 4)),
        ((0, 1), (2, 3, 4, 5, 6)),
    ],
)
def test_canonicalization_rejects_incomplete_duplicate_empty_or_out_of_range_groups(
    groups: object,
) -> None:
    with pytest.raises(ValueError):
        canonicalize_partition(groups)  # type: ignore[arg-type]


@pytest.mark.parametrize("groups", [None, ["users"], [[0, "1"], [2, 3, 4, 5]], [[True], [0, 1, 2, 3, 4]]])
def test_canonicalization_rejects_invalid_group_types(groups: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonicalize_partition(groups)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"num_users": 0}, ValueError),
        ({"num_users": True}, TypeError),
        ({"num_users": 2.5}, TypeError),
        ({"allow_singletons": 1}, TypeError),
        ({"min_group_size": 0}, ValueError),
        ({"max_group_size": 0}, ValueError),
        ({"min_group_size": 3, "max_group_size": 2}, ValueError),
    ],
)
def test_partition_parameters_are_validated(kwargs: dict[str, object], exception: type[Exception]) -> None:
    with pytest.raises(exception):
        enumerate_candidate_partitions(**kwargs)  # type: ignore[arg-type]


def test_custom_maximum_group_size_is_enforced() -> None:
    partitions = enumerate_candidate_partitions(max_group_size=3)

    assert all(max(map(len, partition)) <= 3 for partition in partitions)
    assert len(partitions) == 166


def test_numpy_integer_ids_are_supported_but_boolean_ids_are_not() -> None:
    assert canonicalize_partition(
        ((np.int64(0),), (np.int64(1),), (np.int64(2),), (3, 4, 5))
    ) == ((0,), (1,), (2,), (3, 4, 5))
    with pytest.raises(TypeError):
        canonicalize_partition(((True,), (1,), (2,), (3, 4, 5)))
