# Tests one-layer RSMA effective gains and ideal-SIC signal terms.
import numpy as np
import pytest

from meta_hrl.rsma.constraints import ResourceAllocation
from meta_hrl.rsma.precoding import one_layer_rsma_precoders
from meta_hrl.rsma.signal_model import one_layer_rsma_signal_terms


def test_signal_terms_match_manual_complex_matrix_calculation() -> None:
    channels = np.array([[1.0 + 1.0j, 2.0], [1.0, -1.0j]], dtype=np.complex128)
    precoders = np.array(
        [[1.0, 1.0j, 0.0], [0.5j, 0.0, 2.0]], dtype=np.complex128
    )
    expected_gains = channels @ precoders
    expected_powers = np.abs(expected_gains) ** 2

    terms = one_layer_rsma_signal_terms(
        channels, precoders, noise_variance=0.5
    )

    np.testing.assert_allclose(terms.effective_gains, expected_gains)
    np.testing.assert_allclose(terms.received_powers, expected_powers)
    assert terms.effective_gains.shape == (2, 3)
    assert terms.effective_gains.dtype == np.complex128
    assert terms.received_powers.dtype == np.float64
    assert terms.noise_variance == 0.5

    np.testing.assert_allclose(terms.common_signal_powers, expected_powers[:, 0])
    np.testing.assert_allclose(terms.common_interference_powers, expected_powers[:, 1:].sum(axis=1))
    np.testing.assert_allclose(terms.private_signal_powers, [expected_powers[0, 1], expected_powers[1, 2]])
    np.testing.assert_allclose(
        terms.private_interference_powers,
        [expected_powers[0, 2], expected_powers[1, 1]],
    )


def test_single_user_private_interference_is_zero_after_ideal_sic() -> None:
    terms = one_layer_rsma_signal_terms(
        np.array([[1.0, 2.0j]]),
        np.array([[1.0, 0.5], [0.0, 0.5j]]),
        noise_variance=1.0,
    )

    np.testing.assert_allclose(terms.private_interference_powers, [0.0])
    assert terms.common_interference_powers[0] == pytest.approx(
        terms.private_signal_powers[0]
    )


def test_zero_power_precoder_columns_are_valid_signal_terms() -> None:
    channels = np.eye(2, dtype=np.complex128)
    precoders = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)

    terms = one_layer_rsma_signal_terms(channels, precoders, noise_variance=1.0)

    np.testing.assert_array_equal(terms.common_signal_powers, [0.0, 0.0])
    np.testing.assert_array_equal(terms.private_signal_powers, [1.0, 0.0])
    np.testing.assert_array_equal(terms.private_interference_powers, [0.0, 0.0])


def test_signal_model_integrates_with_power_weighted_precoders() -> None:
    channels = np.array([[1.0, 0.0, 1.0j], [0.0, 1.0, 1.0]], dtype=np.complex128)
    allocation = ResourceAllocation(0.2, np.array([0.3, 0.5]), np.array([0.4, 0.6]))
    precoders = one_layer_rsma_precoders(channels, allocation, private_scheme="mrt")

    terms = one_layer_rsma_signal_terms(channels, precoders, noise_variance=1.0)

    assert terms.effective_gains.shape == (2, 3)
    assert terms.received_powers.shape == (2, 3)
    assert terms.common_signal_powers.shape == (2,)
    assert terms.private_signal_powers.shape == (2,)


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
def test_signal_model_rejects_invalid_channels(channels: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_layer_rsma_signal_terms(channels, np.ones((2, 2)), noise_variance=1.0)


@pytest.mark.parametrize(
    "precoders",
    [
        [1.0, 2.0],
        np.empty((0, 3)),
        [[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[True, False, False], [False, False, False]],
        np.ones((3, 3)),
        np.ones((2, 2)),
    ],
)
def test_signal_model_rejects_invalid_precoders(precoders: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_layer_rsma_signal_terms(
            np.eye(2, dtype=np.complex128), precoders, noise_variance=1.0
        )


@pytest.mark.parametrize("noise_variance", [0.0, -1.0, np.nan, np.inf, True, 1.0 + 1.0j, [1.0]])
def test_signal_model_rejects_invalid_noise_variance(noise_variance: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_layer_rsma_signal_terms(
            np.eye(1, dtype=np.complex128),
            np.ones((1, 2), dtype=np.complex128),
            noise_variance=noise_variance,
        )
