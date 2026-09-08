# Tests deterministic channel validation and diagnostic utilities.
import numpy as np
import pytest

from meta_hrl.channel.far_field import ula_los_steering_vector
from meta_hrl.channel.geometry import ULAConfig
from meta_hrl.channel.near_field import ula_near_field_los_steering_vector
from meta_hrl.channel.validation import (
    compare_near_and_far_field,
    steering_vector_norms,
    summarize_channel_diagnostics,
    user_channel_correlation_matrix,
)


def test_steering_vector_norms_supports_far_and_near_field_batches() -> None:
    config = ULAConfig(num_antennas=4)
    far_vectors = ula_los_steering_vector(config, [0.0, 0.2, -0.3])
    near_vectors = ula_near_field_los_steering_vector(
        config, np.array([[1.0], [2.0]]), np.array([0.0, 0.2, -0.3])
    )

    np.testing.assert_allclose(steering_vector_norms(far_vectors), 1.0)
    assert steering_vector_norms(near_vectors).shape == (2, 3)
    np.testing.assert_allclose(steering_vector_norms(near_vectors), 1.0)


def test_steering_vector_norms_match_manual_complex_values() -> None:
    vectors = np.array([[3.0 + 4.0j, 0.0], [1.0j, -1.0j]])

    np.testing.assert_allclose(steering_vector_norms(vectors), [5.0, np.sqrt(2.0)])


def test_user_channel_correlation_matrix_for_orthogonal_and_collinear_users() -> None:
    channels = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0j]], dtype=np.complex128)

    correlation = user_channel_correlation_matrix(channels)

    np.testing.assert_allclose(np.diag(correlation), 1.0)
    np.testing.assert_allclose(correlation, correlation.conj().T)
    assert correlation.dtype == np.complex128
    assert correlation[0, 1] == pytest.approx(0.0)

    collinear = user_channel_correlation_matrix([[1.0, 1.0j], [2.0, 2.0j]])
    assert abs(collinear[0, 1]) == pytest.approx(1.0)


def test_user_correlation_is_invariant_to_positive_per_user_scaling() -> None:
    channels = np.array([[1.0, 1.0j], [2.0 - 1.0j, 1.0]])
    scaled = channels * np.array([[2.0], [7.0]])

    np.testing.assert_allclose(
        user_channel_correlation_matrix(channels),
        user_channel_correlation_matrix(scaled),
    )


def test_near_far_comparison_reports_convergence_boundary_and_broadcasting() -> None:
    config = ULAConfig()
    comparison = compare_near_and_far_field(
        config,
        np.array([[1.0], [100_000.0 * config.rayleigh_distance_m]]),
        np.array([0.0, 0.25]),
    )

    assert comparison.is_near_field.shape == (2, 2)
    assert comparison.coherence.shape == (2, 2)
    assert comparison.direct_l2_error.shape == (2, 2)
    assert comparison.phase_invariant_l2_error.shape == (2, 2)
    assert comparison.is_near_field[0, 0]
    assert comparison.coherence[1, 1] == pytest.approx(1.0, abs=1e-6)
    assert comparison.direct_l2_error[1, 1] < 3e-6

    boundary = compare_near_and_far_field(config, config.rayleigh_distance_m, 0.2)
    assert not bool(boundary.is_near_field)


def test_channel_summary_reports_expected_energy_and_pairwise_statistics() -> None:
    channels = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.complex128)

    summary = summarize_channel_diagnostics(channels)

    assert summary.num_users == 2
    assert summary.num_antennas == 2
    assert summary.norm_min == pytest.approx(2.0)
    assert summary.norm_mean == pytest.approx(3.5)
    assert summary.norm_max == pytest.approx(5.0)
    assert summary.mean_channel_power == pytest.approx(14.5)
    assert summary.max_pairwise_correlation == pytest.approx(0.8)
    assert summary.mean_pairwise_correlation == pytest.approx(0.8)


def test_channel_summary_uses_zero_pairwise_metrics_for_one_user() -> None:
    summary = summarize_channel_diagnostics([[1.0, 1.0j]])

    assert summary.max_pairwise_correlation == 0.0
    assert summary.mean_pairwise_correlation == 0.0


@pytest.mark.parametrize(
    "vectors",
    [1.0, [], np.array([np.nan]), np.array([np.inf]), np.array([True]), ["x"]],
)
def test_steering_vector_norms_reject_invalid_inputs(vectors: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        steering_vector_norms(vectors)


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
def test_channel_correlation_rejects_invalid_inputs(channels: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        user_channel_correlation_matrix(channels)


def test_near_far_comparison_rejects_invalid_config_and_positions() -> None:
    with pytest.raises(TypeError):
        compare_near_and_far_field(object(), 1.0, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compare_near_and_far_field(ULAConfig(), 0.0, 0.0)
    with pytest.raises(TypeError):
        compare_near_and_far_field(ULAConfig(), 1.0, True)
