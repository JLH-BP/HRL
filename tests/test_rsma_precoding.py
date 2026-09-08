# Tests MRT, RZF, common-beam, and power-weighted RSMA precoding.
import numpy as np
import pytest

from meta_hrl.rsma.constraints import ResourceAllocation
from meta_hrl.rsma.precoding import (
    mrt_private_directions,
    normalized_channel_sum_common_direction,
    one_layer_rsma_precoders,
    rzf_private_directions,
)


def test_mrt_directions_match_conjugated_user_rows() -> None:
    channels = np.array([[1.0 + 1.0j, 2.0], [2.0j, -1.0]], dtype=np.complex128)

    directions = mrt_private_directions(channels)
    expected = channels.conj().T / np.linalg.norm(channels, axis=1)

    assert directions.shape == (2, 2)
    assert directions.dtype == np.complex128
    np.testing.assert_allclose(directions, expected)
    np.testing.assert_allclose(np.sum(np.abs(directions) ** 2, axis=0), 1.0)


def test_rzf_directions_are_finite_unit_norm_and_match_mrt_for_one_user() -> None:
    channels = np.array([[1.0 + 2.0j, -3.0j]], dtype=np.complex128)

    rzf = rzf_private_directions(channels)

    np.testing.assert_allclose(rzf, mrt_private_directions(channels))
    np.testing.assert_allclose(np.sum(np.abs(rzf) ** 2, axis=0), 1.0)
    assert np.all(np.isfinite(rzf))


def test_rzf_reduces_cross_user_leakage_for_full_rank_channels() -> None:
    channels = np.array([[1.0, 0.3], [0.3, 1.0]], dtype=np.complex128)
    mrt_effective = channels @ mrt_private_directions(channels)
    rzf_effective = channels @ rzf_private_directions(channels)

    mrt_leakage = np.abs(mrt_effective[0, 1]) + np.abs(mrt_effective[1, 0])
    rzf_leakage = np.abs(rzf_effective[0, 1]) + np.abs(rzf_effective[1, 0])

    assert rzf_leakage < mrt_leakage


def test_rzf_handles_collinear_channels_with_regularization() -> None:
    channels = np.array([[1.0, 1.0j], [2.0, 2.0j]], dtype=np.complex128)

    directions = rzf_private_directions(channels)

    assert directions.shape == (2, 2)
    assert np.all(np.isfinite(directions))
    np.testing.assert_allclose(np.sum(np.abs(directions) ** 2, axis=0), 1.0)


def test_common_direction_uses_sum_of_normalized_user_directions() -> None:
    channels = np.array([[1.0, 0.0], [0.0, 4.0]], dtype=np.complex128)

    common = normalized_channel_sum_common_direction(channels)

    np.testing.assert_allclose(common, np.array([1.0, 1.0]) / np.sqrt(2.0))
    np.testing.assert_allclose(
        common,
        normalized_channel_sum_common_direction(channels * np.array([[3.0], [7.0]])),
    )


def test_common_direction_rejects_exactly_cancelling_user_directions() -> None:
    with pytest.raises(ValueError):
        normalized_channel_sum_common_direction([[1.0, 0.0], [-1.0, 0.0]])


def test_power_weighted_precoders_match_resource_allocation() -> None:
    channels = np.array([[1.0, 0.0, 1.0j], [0.0, 2.0, 1.0]], dtype=np.complex128)
    allocation = ResourceAllocation(
        common_power=0.2,
        private_powers=np.array([0.3, 0.5]),
        common_rate_allocations=np.array([0.4, 0.6]),
    )

    precoders = one_layer_rsma_precoders(channels, allocation, private_scheme="rzf")

    assert precoders.shape == (3, 3)
    np.testing.assert_allclose(
        np.sum(np.abs(precoders) ** 2, axis=0), [0.2, 0.3, 0.5]
    )
    assert np.sum(np.abs(precoders) ** 2) == pytest.approx(1.0)


def test_zero_power_streams_keep_their_columns_and_are_zero() -> None:
    channels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    allocation = ResourceAllocation(
        common_power=0.0,
        private_powers=np.array([1.0, 0.0]),
        common_rate_allocations=np.array([0.0, 0.0]),
    )

    precoders = one_layer_rsma_precoders(channels, allocation)

    np.testing.assert_array_equal(precoders[:, 0], np.zeros(2))
    np.testing.assert_array_equal(precoders[:, 2], np.zeros(2))
    assert np.sum(np.abs(precoders[:, 1]) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "channels",
    [
        [1.0, 2.0],
        np.empty((0, 2)),
        np.empty((2, 0)),
        [[0.0, 0.0]],
        [[np.nan, 1.0]],
        [[True, False]],
        [["x", "y"]],
    ],
)
def test_precoding_rejects_invalid_channel_matrices(channels: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        mrt_private_directions(channels)


def test_precoder_assembly_rejects_mismatched_allocation_and_unknown_scheme() -> None:
    channels = np.eye(2, dtype=np.complex128)
    mismatched = ResourceAllocation(0.2, np.array([0.8]), np.array([0.0]))
    matching = ResourceAllocation(0.2, np.array([0.4, 0.4]), np.array([0.0, 0.0]))

    with pytest.raises(ValueError):
        one_layer_rsma_precoders(channels, mismatched)
    with pytest.raises(ValueError):
        one_layer_rsma_precoders(channels, matching, private_scheme="unknown")
    with pytest.raises(TypeError):
        one_layer_rsma_precoders(channels, object())  # type: ignore[arg-type]
