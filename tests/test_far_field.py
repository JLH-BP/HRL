# Tests far-field ULA LoS steering-vector conventions and input validation.
import math

import numpy as np
import pytest

from hrl.channel.far_field import ula_los_steering_vector
from hrl.channel.geometry import ULAConfig


def test_broadside_vector_has_uniform_phase() -> None:
    config = ULAConfig(num_antennas=4)

    vector = ula_los_steering_vector(config, 0.0)

    np.testing.assert_allclose(vector, np.ones(4, dtype=np.complex128) / 2.0)
    assert vector.dtype == np.complex128


def test_vector_matches_explicit_positive_phase_formula() -> None:
    config = ULAConfig(num_antennas=3)
    angle = math.pi / 6.0
    positions = config.element_positions_m()
    expected = np.exp(
        1j * 2.0 * np.pi * positions * np.sin(angle) / config.wavelength_m
    ) / math.sqrt(config.num_antennas)

    vector = ula_los_steering_vector(config, angle)

    np.testing.assert_allclose(vector, expected)


def test_opposite_angles_produce_complex_conjugate_vectors() -> None:
    config = ULAConfig(num_antennas=5)
    angle = 0.37

    np.testing.assert_allclose(
        ula_los_steering_vector(config, -angle),
        np.conj(ula_los_steering_vector(config, angle)),
    )


def test_vectorized_angles_preserve_batch_shape_dtype_and_norm() -> None:
    config = ULAConfig(num_antennas=4)
    angles = np.array([[0.0, 0.1], [-0.2, 0.3]])

    vectors = ula_los_steering_vector(config, angles)

    assert vectors.shape == (2, 2, 4)
    assert vectors.dtype == np.complex128
    np.testing.assert_allclose(np.sum(np.abs(vectors) ** 2, axis=-1), 1.0)


def test_custom_ula_configuration_controls_vector_length_and_phase() -> None:
    config = ULAConfig(
        num_antennas=3,
        carrier_frequency_hz=30e9,
        element_spacing_m=0.01,
    )
    angle = 0.2

    vector = ula_los_steering_vector(config, angle)
    expected = np.exp(
        1j
        * 2.0
        * np.pi
        * config.element_positions_m()
        * np.sin(angle)
        / config.wavelength_m
    ) / math.sqrt(3)

    assert vector.shape == (3,)
    np.testing.assert_allclose(vector, expected)


@pytest.mark.parametrize(
    "angle",
    [np.nan, np.inf, 1.0 + 1.0j, True, "0.1", np.array([False, True])],
)
def test_steering_vector_rejects_invalid_angles(angle: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ula_los_steering_vector(ULAConfig(), angle)
