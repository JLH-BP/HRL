# Tests one-layer RSMA SINR, rates, common bottleneck, and allocation feasibility.
import numpy as np
import pytest

from meta_hrl.rsma.constraints import ResourceAllocation
from meta_hrl.rsma.precoding import one_layer_rsma_precoders
from meta_hrl.rsma.rate import one_layer_rsma_rates
from meta_hrl.rsma.signal_model import (
    OneLayerRSMASignalTerms,
    one_layer_rsma_signal_terms,
)


def _signal_terms(
    common_signal: object = (4.0, 1.0),
    common_interference: object = (1.0, 3.0),
    private_signal: object = (9.0, 4.0),
    private_interference: object = (3.0, 0.0),
    noise_variance: float = 1.0,
) -> OneLayerRSMASignalTerms:
    common_signal_array = np.asarray(common_signal, dtype=np.float64)
    user_count = common_signal_array.size
    return OneLayerRSMASignalTerms(
        effective_gains=np.zeros((user_count, user_count + 1), dtype=np.complex128),
        received_powers=np.zeros((user_count, user_count + 1), dtype=np.float64),
        noise_variance=noise_variance,
        common_signal_powers=common_signal_array,
        common_interference_powers=np.asarray(common_interference, dtype=np.float64),
        private_signal_powers=np.asarray(private_signal, dtype=np.float64),
        private_interference_powers=np.asarray(private_interference, dtype=np.float64),
    )


def test_rates_match_manual_sinr_and_weakest_common_user() -> None:
    result = one_layer_rsma_rates(_signal_terms(), [0.1, 0.2])

    np.testing.assert_allclose(result.common_sinrs, [2.0, 0.25])
    np.testing.assert_allclose(result.private_sinrs, [2.25, 4.0])
    np.testing.assert_allclose(result.common_decodable_rates, np.log2([3.0, 1.25]))
    np.testing.assert_allclose(result.private_rates, np.log2([3.25, 5.0]))
    assert result.common_rate == pytest.approx(np.log2(1.25))
    assert result.allocated_common_rate == pytest.approx(0.3)
    assert result.common_rate_slack == pytest.approx(np.log2(1.25) - 0.3)
    assert result.is_common_rate_allocation_feasible
    np.testing.assert_allclose(result.user_rates, result.private_rates + [0.1, 0.2])
    assert result.sum_rate == pytest.approx(result.user_rates.sum())


def test_common_rate_allocation_boundary_tolerance_and_infeasibility() -> None:
    terms = _signal_terms(common_signal=(1.0,), common_interference=(0.0,), private_signal=(0.0,), private_interference=(0.0,))
    common_rate = np.log2(2.0)

    exact = one_layer_rsma_rates(terms, [common_rate])
    within_tolerance = one_layer_rsma_rates(terms, [common_rate + 5e-11])
    infeasible = one_layer_rsma_rates(terms, [common_rate + 1e-4])

    assert exact.is_common_rate_allocation_feasible
    assert within_tolerance.is_common_rate_allocation_feasible
    assert within_tolerance.common_rate_slack < 0.0
    assert not infeasible.is_common_rate_allocation_feasible


def test_zero_common_rate_and_zero_private_stream_boundaries() -> None:
    zero_common = _signal_terms(
        common_signal=(0.0, 0.0),
        common_interference=(0.0, 0.0),
        private_signal=(1.0, 0.0),
        private_interference=(0.0, 0.0),
    )

    feasible = one_layer_rsma_rates(zero_common, [0.0, 0.0])
    infeasible = one_layer_rsma_rates(zero_common, [0.0, 0.1])

    np.testing.assert_allclose(feasible.common_sinrs, [0.0, 0.0])
    np.testing.assert_allclose(feasible.common_decodable_rates, [0.0, 0.0])
    assert feasible.common_rate == 0.0
    assert feasible.is_common_rate_allocation_feasible
    assert not infeasible.is_common_rate_allocation_feasible
    np.testing.assert_allclose(feasible.private_rates, [1.0, 0.0])


def test_rate_model_integrates_with_resource_precoding_and_signal_layers() -> None:
    channels = np.array([[1.0, 0.0, 1.0j], [0.0, 1.0, 1.0]], dtype=np.complex128)
    allocation = ResourceAllocation(0.2, np.array([0.3, 0.5]), np.array([0.0, 0.0]))
    terms = one_layer_rsma_signal_terms(
        channels,
        one_layer_rsma_precoders(channels, allocation, private_scheme="rzf"),
        noise_variance=1.0,
    )
    initial = one_layer_rsma_rates(terms, [0.0, 0.0])
    result = one_layer_rsma_rates(terms, np.full(2, initial.common_rate / 2.0))

    assert result.common_sinrs.shape == (2,)
    assert result.private_sinrs.shape == (2,)
    assert result.user_rates.dtype == np.float64
    assert np.all(result.common_sinrs >= 0.0)
    assert np.all(result.private_sinrs >= 0.0)
    assert result.is_common_rate_allocation_feasible


@pytest.mark.parametrize(
    "allocations",
    [[-0.1, 0.0], [np.nan, 0.0], [np.inf, 0.0], [True, False], [1.0 + 1.0j, 0.0], [[0.0, 0.0]], []],
)
def test_rate_model_rejects_invalid_common_rate_allocations(allocations: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_layer_rsma_rates(_signal_terms(), allocations)


@pytest.mark.parametrize("atol", [-1.0, np.nan, np.inf, True, 1.0 + 1.0j, [1e-10]])
def test_rate_model_rejects_invalid_tolerance(atol: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        one_layer_rsma_rates(_signal_terms(), [0.0, 0.0], atol=atol)


def test_rate_model_rejects_invalid_or_corrupted_signal_terms() -> None:
    with pytest.raises(TypeError):
        one_layer_rsma_rates(object(), [0.0, 0.0])  # type: ignore[arg-type]

    corrupted = _signal_terms(private_signal=(-1.0, 1.0))
    with pytest.raises(ValueError):
        one_layer_rsma_rates(corrupted, [0.0, 0.0])

    mismatched = _signal_terms(private_interference=(0.0,))
    with pytest.raises(ValueError):
        one_layer_rsma_rates(mismatched, [0.0, 0.0])
