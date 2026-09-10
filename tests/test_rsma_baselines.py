# Tests deterministic equal-power SDMA and fixed-common one-layer RSMA baselines.
import numpy as np
import pytest

from hrl.rsma.baselines import (
    equal_power_mrt,
    equal_power_rzf,
    fixed_common_rsma,
)
from hrl.rsma.rate import one_layer_rsma_rates


@pytest.fixture
def channels() -> np.ndarray:
    return np.array([[1.0, 0.2, 1.0j], [0.1, 1.0, 1.0]], dtype=np.complex128)


@pytest.mark.parametrize(
    ("baseline_function", "expected_scheme"),
    [(equal_power_mrt, "mrt"), (equal_power_rzf, "rzf")],
)
def test_equal_power_sdma_baselines_have_no_common_stream(
    channels: np.ndarray, baseline_function: object, expected_scheme: str
) -> None:
    result = baseline_function(channels, total_power=1.0, noise_variance=1.0)  # type: ignore[operator]

    assert result.private_scheme == expected_scheme
    assert result.common_power_fraction == 0.0
    assert result.power_allocation.common_power == 0.0
    np.testing.assert_allclose(result.power_allocation.private_powers, [0.5, 0.5])
    np.testing.assert_array_equal(result.allocation.common_rate_allocations, [0.0, 0.0])
    np.testing.assert_array_equal(result.precoders[:, 0], np.zeros(3))
    np.testing.assert_allclose(result.rates.user_rates, result.rates.private_rates)
    assert np.sum(np.abs(result.precoders) ** 2) == pytest.approx(1.0)


def test_fixed_common_rsma_uses_power_before_common_rate_and_default_equal_shares(
    channels: np.ndarray,
) -> None:
    result = fixed_common_rsma(
        channels,
        total_power=1.0,
        noise_variance=1.0,
        common_power_fraction=0.2,
        private_scheme="mrt",
    )
    initial = one_layer_rsma_rates(result.signal_terms, [0.0, 0.0])

    assert result.power_allocation.common_power == pytest.approx(0.2)
    np.testing.assert_allclose(result.power_allocation.private_powers, [0.4, 0.4])
    np.testing.assert_allclose(
        np.sum(np.abs(result.precoders) ** 2, axis=0), [0.2, 0.4, 0.4]
    )
    np.testing.assert_allclose(result.common_rate_shares, [0.5, 0.5])
    np.testing.assert_allclose(
        result.allocation.common_rate_allocations,
        [initial.common_rate / 2.0, initial.common_rate / 2.0],
    )
    assert result.rates.common_rate == pytest.approx(initial.common_rate)
    assert result.rates.is_common_rate_allocation_feasible
    assert result.rates.common_rate_slack == pytest.approx(0.0, abs=1e-12)


def test_fixed_common_rsma_honors_custom_common_rate_shares(channels: np.ndarray) -> None:
    result = fixed_common_rsma(
        channels,
        total_power=2.0,
        noise_variance=0.5,
        common_power_fraction=0.25,
        private_scheme="rzf",
        common_rate_shares=[0.75, 0.25],
    )

    assert result.private_scheme == "rzf"
    np.testing.assert_allclose(result.common_rate_shares, [0.75, 0.25])
    np.testing.assert_allclose(
        result.allocation.common_rate_allocations,
        result.rates.common_rate * np.array([0.75, 0.25]),
    )
    assert result.rates.is_common_rate_allocation_feasible


def test_fixed_common_rsma_supports_sdma_and_pure_common_endpoints(channels: np.ndarray) -> None:
    sdma = fixed_common_rsma(
        channels, total_power=1.0, noise_variance=1.0, common_power_fraction=0.0
    )
    common_only = fixed_common_rsma(
        channels, total_power=1.0, noise_variance=1.0, common_power_fraction=1.0
    )

    assert sdma.power_allocation.common_power == 0.0
    np.testing.assert_array_equal(sdma.precoders[:, 0], np.zeros(3))
    np.testing.assert_allclose(common_only.power_allocation.private_powers, [0.0, 0.0])
    np.testing.assert_allclose(np.sum(np.abs(common_only.precoders[:, 1:]) ** 2), 0.0)
    assert common_only.rates.is_common_rate_allocation_feasible


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("total_power", 0.0),
        ("total_power", -1.0),
        ("total_power", np.nan),
        ("noise_variance", 0.0),
        ("noise_variance", np.inf),
        ("common_power_fraction", -0.1),
        ("common_power_fraction", 1.1),
        ("common_power_fraction", True),
    ],
)
def test_fixed_common_rsma_rejects_invalid_scalar_parameters(
    channels: np.ndarray, keyword: str, value: object
) -> None:
    kwargs: dict[str, object] = {
        "total_power": 1.0,
        "noise_variance": 1.0,
        "common_power_fraction": 0.2,
    }
    kwargs[keyword] = value
    with pytest.raises((TypeError, ValueError)):
        fixed_common_rsma(channels, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "shares",
    [[1.0], [-0.1, 1.1], [0.2, 0.2], [np.nan, 1.0], [[0.5, 0.5]], [True, False]],
)
def test_fixed_common_rsma_rejects_invalid_common_rate_shares(
    channels: np.ndarray, shares: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        fixed_common_rsma(
            channels,
            total_power=1.0,
            noise_variance=1.0,
            common_power_fraction=0.2,
            common_rate_shares=shares,
        )


def test_fixed_common_rsma_rejects_unknown_private_scheme(channels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fixed_common_rsma(
            channels,
            total_power=1.0,
            noise_variance=1.0,
            common_power_fraction=0.2,
            private_scheme="unknown",
        )
