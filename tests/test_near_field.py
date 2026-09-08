# Tests near-field spherical-wave ULA LoS steering-vector behavior and validation.
import math

import numpy as np
import pytest

from meta_hrl.channel.far_field import ula_los_steering_vector
from meta_hrl.channel.geometry import ULAConfig
from meta_hrl.channel.near_field import ula_near_field_los_steering_vector


def test_near_field_vector_matches_explicit_spherical_wave_formula() -> None:
    config = ULAConfig(num_antennas=3)
    range_m = 2.0
    angle_rad = math.pi / 6.0
    positions = config.element_positions_m()
    distances = np.hypot(
        range_m * math.sin(angle_rad) - positions,
        range_m * math.cos(angle_rad),
    )
    expected = np.exp(
        -1j * 2.0 * np.pi * (distances - range_m) / config.wavelength_m
    ) / math.sqrt(config.num_antennas)

    vector = ula_near_field_los_steering_vector(config, range_m, angle_rad)

    np.testing.assert_allclose(vector, expected, atol=1e-12)


def test_broadside_near_field_vector_has_symmetric_phase_curvature() -> None:
    config = ULAConfig(num_antennas=5)
    vector = ula_near_field_los_steering_vector(config, 1.0, 0.0)

    assert vector[2] == pytest.approx(1.0 / math.sqrt(5))
    np.testing.assert_allclose(vector, vector[::-1])
    assert not np.allclose(vector[0], vector[2])


def test_near_field_vectors_support_broadcasting_dtype_and_unit_norm() -> None:
    config = ULAConfig(num_antennas=4)
    ranges = np.array([[1.0], [2.0]])
    angles = np.array([0.0, 0.2, -0.3])

    vectors = ula_near_field_los_steering_vector(config, ranges, angles)

    assert vectors.shape == (2, 3, 4)
    assert vectors.dtype == np.complex128
    np.testing.assert_allclose(np.sum(np.abs(vectors) ** 2, axis=-1), 1.0)


def test_near_field_converges_to_far_field_with_shared_center_reference() -> None:
    config = ULAConfig()
    angle = 0.31
    large_range_m = 100_000.0 * config.rayleigh_distance_m

    near_field = ula_near_field_los_steering_vector(config, large_range_m, angle)
    far_field = ula_los_steering_vector(config, angle)

    np.testing.assert_allclose(near_field, far_field, atol=5e-7, rtol=5e-6)


@pytest.mark.parametrize(
    ("range_m", "angle_rad"),
    [
        (0.0, 0.0),
        (-1.0, 0.0),
        (np.nan, 0.0),
        (np.inf, 0.0),
        (1.0 + 1.0j, 0.0),
        (True, 0.0),
        ("1.0", 0.0),
        (1.0, np.nan),
        (1.0, np.inf),
        (1.0, 1.0 + 1.0j),
        (1.0, True),
        (1.0, "0.0"),
    ],
)
def test_near_field_steering_rejects_invalid_inputs(
    range_m: object, angle_rad: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ula_near_field_los_steering_vector(ULAConfig(), range_m, angle_rad)


def test_near_field_steering_rejects_unbroadcastable_inputs() -> None:
    with pytest.raises(ValueError):
        ula_near_field_los_steering_vector(
            ULAConfig(), np.ones(2), np.ones(3)
        )
