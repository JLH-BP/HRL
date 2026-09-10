# Tests mixed-field Rician channel sampling, normalization, and validation.
import numpy as np
import pytest

from hrl.channel.far_field import ula_los_steering_vector
from hrl.channel.geometry import ULAConfig
from hrl.channel.near_field import ula_near_field_los_steering_vector
from hrl.channel.rician import sample_rician_channel


@pytest.mark.parametrize(
    ("range_m", "angle_rad", "expected_los"),
    [
        (20.0, 0.2, "near"),
        (200.0, -0.2, "far"),
    ],
)
def test_large_k_factor_selects_expected_los_model(
    range_m: float, angle_rad: float, expected_los: str
) -> None:
    config = ULAConfig()
    kappa = 1e15
    beta = 4.0

    channel = sample_rician_channel(
        config,
        range_m,
        angle_rad,
        kappa,
        beta,
        rng=np.random.default_rng(1),
    )
    steering = (
        ula_near_field_los_steering_vector(config, range_m, angle_rad)
        if expected_los == "near"
        else ula_los_steering_vector(config, angle_rad)
    )

    np.testing.assert_allclose(channel, np.sqrt(beta) * steering, atol=1e-7)


def test_rayleigh_boundary_uses_far_field_los_vector() -> None:
    config = ULAConfig()
    angle = 0.19

    channel = sample_rician_channel(
        config,
        config.rayleigh_distance_m,
        angle,
        1e15,
        rng=np.random.default_rng(2),
    )

    np.testing.assert_allclose(
        channel, ula_los_steering_vector(config, angle), atol=1e-7
    )


def test_channels_support_broadcasting_and_complex128_output() -> None:
    config = ULAConfig(num_antennas=4)
    channels = sample_rician_channel(
        config,
        np.array([[1.0], [100.0]]),
        np.array([0.0, 0.2, -0.3]),
        k_factor_linear=np.array([0.0, 2.0, 10.0]),
        path_loss_linear=np.array([[1.0], [0.25]]),
        rng=np.random.default_rng(3),
    )

    assert channels.shape == (2, 3, 4)
    assert channels.dtype == np.complex128


def test_path_loss_scales_same_random_realization_and_zero_gain() -> None:
    config = ULAConfig(num_antennas=5)
    arguments = (config, [20.0, 200.0], [0.1, -0.2], [1.0, 4.0])

    unit_gain = sample_rician_channel(
        *arguments, 1.0, rng=np.random.default_rng(4)
    )
    quarter_gain = sample_rician_channel(
        *arguments, 0.25, rng=np.random.default_rng(4)
    )
    zero_gain = sample_rician_channel(
        *arguments, 0.0, rng=np.random.default_rng(4)
    )

    np.testing.assert_allclose(quarter_gain, 0.5 * unit_gain)
    np.testing.assert_array_equal(zero_gain, np.zeros_like(zero_gain))


def test_rng_seed_reproducibility_and_stream_progression() -> None:
    config = ULAConfig(num_antennas=4)
    arguments = (config, 20.0, 0.1, 3.0)

    first = sample_rician_channel(*arguments, rng=np.random.default_rng(5))
    repeated = sample_rician_channel(*arguments, rng=np.random.default_rng(5))
    shared_rng = np.random.default_rng(5)
    advanced = sample_rician_channel(*arguments, rng=shared_rng)
    next_sample = sample_rician_channel(*arguments, rng=shared_rng)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, advanced)
    assert not np.array_equal(advanced, next_sample)


def test_rayleigh_nlos_has_unit_expected_total_energy() -> None:
    config = ULAConfig(num_antennas=16)
    sample_count = 12_000
    channels = sample_rician_channel(
        config,
        np.full(sample_count, 20.0),
        0.1,
        k_factor_linear=0.0,
        rng=np.random.default_rng(6),
    )

    assert np.mean(channels.real) == pytest.approx(0.0, abs=0.01)
    assert np.mean(channels.imag) == pytest.approx(0.0, abs=0.01)
    assert np.var(channels.real) == pytest.approx(1.0 / (2.0 * 16), rel=0.04)
    assert np.var(channels.imag) == pytest.approx(1.0 / (2.0 * 16), rel=0.04)
    assert np.mean(np.sum(np.abs(channels) ** 2, axis=-1)) == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"k_factor_linear": -1.0}, ValueError),
        ({"k_factor_linear": np.nan}, ValueError),
        ({"k_factor_linear": 1.0 + 1.0j}, TypeError),
        ({"k_factor_linear": True}, TypeError),
        ({"path_loss_linear": -0.1}, ValueError),
        ({"path_loss_linear": np.inf}, ValueError),
        ({"path_loss_linear": 1.0 + 1.0j}, TypeError),
        ({"path_loss_linear": False}, TypeError),
        ({"rng": 42}, TypeError),
    ],
)
def test_rician_channel_rejects_invalid_arguments(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    arguments: dict[str, object] = {"k_factor_linear": 1.0}
    arguments.update(kwargs)
    with pytest.raises(exception):
        sample_rician_channel(ULAConfig(), 20.0, 0.1, **arguments)


def test_rician_channel_rejects_unbroadcastable_user_inputs() -> None:
    with pytest.raises(ValueError):
        sample_rician_channel(
            ULAConfig(),
            range_m=np.ones(2),
            angle_rad=np.ones(3),
            k_factor_linear=1.0,
        )
