# Tests ULA geometry, coordinate transforms, distances, and near/far-field classification.
import math

import numpy as np
import pytest

from hrl.channel.geometry import (
    SPEED_OF_LIGHT_M_S,
    ULAConfig,
    cartesian_to_polar,
    classify_near_field,
    element_to_user_distances_m,
    polar_to_cartesian,
)


def test_default_ula_physical_quantities() -> None:
    config = ULAConfig()
    wavelength = SPEED_OF_LIGHT_M_S / 28e9

    assert config.wavelength_m == pytest.approx(wavelength)
    assert config.resolved_element_spacing_m == pytest.approx(wavelength / 2.0)
    assert config.aperture_m == pytest.approx(127 * wavelength / 2.0)
    assert config.rayleigh_distance_m == pytest.approx(
        2.0 * config.aperture_m**2 / wavelength
    )
    assert config.rayleigh_distance_m == pytest.approx(86.304, rel=1e-3)


def test_centered_ula_positions_are_symmetric_and_evenly_spaced() -> None:
    config = ULAConfig()
    positions = config.element_positions_m()

    assert positions.shape == (128,)
    assert positions.dtype == np.float64
    assert positions.mean() == pytest.approx(0.0, abs=1e-15)
    assert positions[0] == pytest.approx(-63.5 * config.resolved_element_spacing_m)
    assert positions[-1] == pytest.approx(63.5 * config.resolved_element_spacing_m)
    np.testing.assert_allclose(np.diff(positions), config.resolved_element_spacing_m)


def test_odd_array_has_a_center_element_at_origin() -> None:
    positions = ULAConfig(num_antennas=5).element_positions_m()
    assert positions[2] == pytest.approx(0.0)


def test_broadside_coordinate_convention_and_round_trip() -> None:
    ranges = np.array([10.0, 10.0, 10.0])
    angles = np.array([0.0, math.pi / 2.0, -math.pi / 2.0])
    x_values, y_values = polar_to_cartesian(ranges, angles)

    np.testing.assert_allclose(x_values, [0.0, 10.0, -10.0], atol=1e-12)
    np.testing.assert_allclose(y_values, [10.0, 0.0, 0.0], atol=1e-12)
    recovered_ranges, recovered_angles = cartesian_to_polar(x_values, y_values)
    np.testing.assert_allclose(recovered_ranges, ranges)
    np.testing.assert_allclose(recovered_angles, angles)


def test_coordinate_transforms_broadcast() -> None:
    ranges = np.array([[10.0], [20.0]])
    angles = np.array([0.0, math.pi / 6.0])
    x_values, y_values = polar_to_cartesian(ranges, angles)
    recovered_ranges, recovered_angles = cartesian_to_polar(x_values, y_values)

    assert x_values.shape == (2, 2)
    np.testing.assert_allclose(recovered_ranges, np.broadcast_to(ranges, (2, 2)))
    np.testing.assert_allclose(recovered_angles, np.broadcast_to(angles, (2, 2)))


def test_element_to_user_distances_supports_single_and_multiple_users() -> None:
    elements = np.array([-1.0, 0.0, 1.0])
    single_user = element_to_user_distances_m(elements, 0.0, 2.0)
    multiple_users = element_to_user_distances_m(elements, [0.0, 1.0], [2.0, 2.0])

    np.testing.assert_allclose(single_user, [math.sqrt(5.0), 2.0, math.sqrt(5.0)])
    assert multiple_users.shape == (2, 3)
    np.testing.assert_allclose(multiple_users[0], single_user)
    np.testing.assert_allclose(multiple_users[1], [math.sqrt(8.0), math.sqrt(5.0), 2.0])


def test_near_field_classification_uses_strict_rayleigh_boundary() -> None:
    boundary = ULAConfig().rayleigh_distance_m
    result = classify_near_field([boundary - 1e-9, boundary, boundary + 1e-9], boundary)

    np.testing.assert_array_equal(result, [True, False, False])


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"num_antennas": 1}, ValueError),
        ({"num_antennas": 2.5}, TypeError),
        ({"num_antennas": True}, TypeError),
        ({"carrier_frequency_hz": 0.0}, ValueError),
        ({"element_spacing_m": -1.0}, ValueError),
    ],
)
def test_ula_config_rejects_invalid_parameters(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        ULAConfig(**kwargs)


@pytest.mark.parametrize(
    "function_args",
    [
        ((0.0, 0.0), polar_to_cartesian),
        (([1.0, np.nan], 0.0), polar_to_cartesian),
        (([1.0 + 1.0j], 0.0), polar_to_cartesian),
        ((0.0, 0.0), cartesian_to_polar),
        (([1.0, -1.0], ULAConfig().rayleigh_distance_m), classify_near_field),
    ],
)
def test_geometry_functions_reject_invalid_values(function_args: tuple[tuple[object, object], object]) -> None:
    args, function = function_args
    with pytest.raises((TypeError, ValueError)):
        function(*args)
