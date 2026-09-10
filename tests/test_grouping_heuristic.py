# Tests hybrid geometry/channel affinity and deterministic heuristic grouping.
import numpy as np
import pytest

from hrl.channel.geometry import ULAConfig
from hrl.grouping.heuristic import (
    GroupingWeights,
    pairwise_grouping_affinity,
    score_partition,
    select_heuristic_partition,
)


def test_geometry_affinity_is_symmetric_bounded_and_zero_diagonal() -> None:
    affinity = pairwise_grouping_affinity(
        ranges_m=[20.0, 30.0, 120.0],
        angles_rad=[0.0, 0.05, 0.8],
        method="geometry",
    )

    assert affinity.shape == (3, 3)
    assert affinity.dtype == np.float64
    np.testing.assert_allclose(affinity, affinity.T)
    np.testing.assert_array_equal(np.diag(affinity), np.zeros(3))
    assert np.all((affinity >= 0.0) & (affinity <= 1.0))
    assert affinity[0, 1] > affinity[0, 2]


def test_channel_affinity_uses_normalized_correlation_and_requires_channels() -> None:
    channels = np.array([[1.0, 1.0j], [2.0, 2.0j], [1.0, -1.0j]])
    affinity = pairwise_grouping_affinity(
        ranges_m=[20.0, 30.0, 120.0],
        angles_rad=[0.0, 0.5, 0.8],
        channels=channels,
        method="channel",
    )

    assert affinity[0, 1] == pytest.approx(1.0)
    assert affinity[0, 2] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pairwise_grouping_affinity(
            ranges_m=[20.0, 30.0], angles_rad=[0.0, 0.1], method="channel"
        )


def test_hybrid_without_channels_renormalizes_geometry_features() -> None:
    hybrid = pairwise_grouping_affinity(
        ranges_m=[20.0, 30.0, 120.0],
        angles_rad=[0.0, 0.05, 0.8],
        method="hybrid",
    )
    geometry = pairwise_grouping_affinity(
        ranges_m=[20.0, 30.0, 120.0],
        angles_rad=[0.0, 0.05, 0.8],
        method="geometry",
    )

    np.testing.assert_allclose(hybrid, geometry)


def test_near_field_distance_similarity_and_far_field_distance_invariance() -> None:
    config = ULAConfig()
    near = pairwise_grouping_affinity(
        ranges_m=[10.0, 20.0, 120.0],
        angles_rad=[0.0, 0.0, 0.0],
        ula_config=config,
        method="geometry",
    )
    far = pairwise_grouping_affinity(
        ranges_m=[120.0, 220.0, 320.0],
        angles_rad=[0.0, 0.0, 0.0],
        ula_config=config,
        method="geometry",
    )

    assert near[0, 1] > near[0, 2]
    assert far[0, 1] == pytest.approx(far[0, 2])


def test_angle_wrap_treats_opposite_periodic_edges_as_close() -> None:
    affinity = pairwise_grouping_affinity(
        ranges_m=[120.0, 120.0],
        angles_rad=[np.pi - 1e-3, -np.pi + 1e-3],
        method="geometry",
    )

    assert affinity[0, 1] > 0.99


def test_score_is_invariant_to_partition_order_and_rewards_high_affinity_pair() -> None:
    affinity = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    grouped = score_partition(((0, 1), (2,)), affinity, singleton_score=0.0)
    reordered = score_partition(((2,), (1, 0)), affinity, singleton_score=0.0)
    separated = score_partition(((0,), (1,), (2,)), affinity, singleton_score=0.0)

    assert grouped == pytest.approx(reordered)
    assert grouped > separated


def test_select_heuristic_partition_is_deterministic_and_returns_diagnostics() -> None:
    channels = np.array([[1.0, 1.0j], [2.0, 2.0j], [1.0, -1.0j]])
    kwargs = dict(
        ranges_m=[20.0, 25.0, 120.0],
        angles_rad=[0.0, 0.01, 0.8],
        channels=channels,
        method="hybrid",
    )

    first = select_heuristic_partition(**kwargs)
    second = select_heuristic_partition(**kwargs)

    assert first.partition == second.partition
    assert first.score == pytest.approx(second.score)
    assert first.candidates == second.candidates
    assert first.candidate_scores == second.candidate_scores
    assert first.pairwise_affinity.shape == (3, 3)
    assert first.near_field_mask.shape == (3,)


def test_singleton_user_is_private_only_candidate_without_error() -> None:
    result = select_heuristic_partition(
        ranges_m=[20.0, 120.0], angles_rad=[0.0, 0.8], method="geometry"
    )

    assert all(set(user for group in result.partition for user in group) == {0, 1} for _ in [0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ranges_m": [1.0], "angles_rad": [0.0], "method": "bad"},
        {"ranges_m": [1.0], "angles_rad": [0.0, 0.1]},
        {"ranges_m": [0.0], "angles_rad": [0.0]},
        {"ranges_m": [1.0, 2.0], "angles_rad": [0.0, 0.1], "channels": [[1.0, 0.0]]},
    ],
)
def test_grouping_rejects_invalid_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        pairwise_grouping_affinity(**kwargs)  # type: ignore[arg-type]


def test_grouping_weights_must_have_positive_finite_mass() -> None:
    with pytest.raises(ValueError):
        GroupingWeights(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        GroupingWeights(channel_correlation=-1.0)
