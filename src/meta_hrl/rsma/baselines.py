"""实现确定性等功率SDMA和固定通用单层RSMA基线。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constraints import ResourceAllocation, validate_resource_allocation
from .precoding import one_layer_rsma_precoders
from .rate import OneLayerRSMARates, one_layer_rsma_rates
from .signal_model import OneLayerRSMASignalTerms, one_layer_rsma_signal_terms

__all__ = [
    "OneLayerRSMABaseline",
    "equal_power_mrt",
    "equal_power_rzf",
    "fixed_common_rsma",
    "one_layer_rsma_baseline",
]


@dataclass(frozen=True, slots=True)
class OneLayerRSMABaseline:
    """Store a deterministic one-layer RSMA baseline result and intermediates."""

    name: str
    private_scheme: str
    total_power: float
    noise_variance: float
    common_power_fraction: float
    common_rate_shares: NDArray[np.float64]
    power_allocation: ResourceAllocation
    allocation: ResourceAllocation
    precoders: NDArray[np.complex128]
    signal_terms: OneLayerRSMASignalTerms
    rates: OneLayerRSMARates


def _positive_finite_scalar(value: object, name: str) -> float:
    """Validate and return a finite strictly positive real scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real-valued scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


def _common_power_fraction(value: object) -> float:
    """Validate and return a finite common-power fraction in the closed unit interval."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("common_power_fraction must be a real-valued scalar.")
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("common_power_fraction must be finite and between zero and one.")
    return fraction


def _validated_shares(
    shares: ArrayLike | None, user_count: int
) -> NDArray[np.float64]:
    """Return user common-rate shares as a finite nonnegative probability vector."""
    if shares is None:
        return np.full(user_count, 1.0 / user_count, dtype=np.float64)
    values = np.asarray(shares)
    if np.iscomplexobj(values) or values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("common_rate_shares must contain real numeric values.")
    try:
        vector = np.asarray(shares, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError("common_rate_shares must contain real numeric values.") from error
    if vector.ndim != 1 or vector.shape != (user_count,):
        raise ValueError("common_rate_shares must have one entry per channel user.")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("common_rate_shares must be finite and nonnegative.")
    if not np.isclose(np.sum(vector), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("common_rate_shares must sum to one.")
    return vector.copy()


def one_layer_rsma_baseline(
    channels: ArrayLike,
    *,
    total_power: float,
    noise_variance: float,
    common_power_fraction: float,
    private_scheme: str,
    common_rate_shares: ArrayLike | None = None,
) -> OneLayerRSMABaseline:
    """Evaluate a fixed-power one-layer RSMA baseline with MRT or RZF private beams."""
    total_power = _positive_finite_scalar(total_power, "total_power")
    noise_variance = _positive_finite_scalar(noise_variance, "noise_variance")
    common_power_fraction = _common_power_fraction(common_power_fraction)
    if private_scheme not in {"mrt", "rzf"}:
        raise ValueError("private_scheme must be either 'mrt' or 'rzf'.")

    channel_matrix = np.asarray(channels)
    if channel_matrix.ndim != 2 or channel_matrix.shape[0] == 0:
        raise ValueError("channels must have a nonempty two-dimensional user axis.")
    user_count = channel_matrix.shape[0]
    common_rate_shares_vector = _validated_shares(common_rate_shares, user_count)
    common_power = common_power_fraction * total_power
    private_powers = np.full(
        user_count, (1.0 - common_power_fraction) * total_power / user_count, dtype=np.float64
    )
    power_allocation = ResourceAllocation(
        common_power=common_power,
        private_powers=private_powers,
        common_rate_allocations=np.zeros(user_count, dtype=np.float64),
    )
    power_feasibility = validate_resource_allocation(
        power_allocation, total_power=total_power, common_rate_budget=0.0
    )
    if not power_feasibility.is_feasible:
        raise RuntimeError("Baseline power allocation unexpectedly violates its budget.")

    precoders = one_layer_rsma_precoders(
        channels, power_allocation, private_scheme=private_scheme
    )
    signal_terms = one_layer_rsma_signal_terms(
        channels, precoders, noise_variance=noise_variance
    )
    initial_rates = one_layer_rsma_rates(
        signal_terms, np.zeros(user_count, dtype=np.float64)
    )
    final_allocation = ResourceAllocation(
        common_power=common_power,
        private_powers=private_powers,
        common_rate_allocations=initial_rates.common_rate * common_rate_shares_vector,
    )
    rates = one_layer_rsma_rates(signal_terms, final_allocation.common_rate_allocations)
    final_feasibility = validate_resource_allocation(
        final_allocation,
        total_power=total_power,
        common_rate_budget=rates.common_rate,
    )
    if not final_feasibility.is_feasible or not rates.is_common_rate_allocation_feasible:
        raise RuntimeError("Baseline common-rate allocation unexpectedly violates its budget.")

    return OneLayerRSMABaseline(
        name=f"fixed_common_{private_scheme}_rsma",
        private_scheme=private_scheme,
        total_power=total_power,
        noise_variance=noise_variance,
        common_power_fraction=common_power_fraction,
        common_rate_shares=common_rate_shares_vector,
        power_allocation=power_allocation,
        allocation=final_allocation,
        precoders=precoders,
        signal_terms=signal_terms,
        rates=rates,
    )


def equal_power_mrt(
    channels: ArrayLike, *, total_power: float, noise_variance: float
) -> OneLayerRSMABaseline:
    """Evaluate an equal-private-power MRT SDMA baseline with no common stream."""
    baseline = one_layer_rsma_baseline(
        channels,
        total_power=total_power,
        noise_variance=noise_variance,
        common_power_fraction=0.0,
        private_scheme="mrt",
        common_rate_shares=np.array([1.0 / np.asarray(channels).shape[0]] * np.asarray(channels).shape[0]),
    )
    return OneLayerRSMABaseline(
        name="equal_power_mrt_sdma",
        private_scheme=baseline.private_scheme,
        total_power=baseline.total_power,
        noise_variance=baseline.noise_variance,
        common_power_fraction=baseline.common_power_fraction,
        common_rate_shares=np.zeros_like(baseline.common_rate_shares),
        power_allocation=baseline.power_allocation,
        allocation=ResourceAllocation(
            baseline.allocation.common_power,
            baseline.allocation.private_powers,
            np.zeros_like(baseline.allocation.common_rate_allocations),
        ),
        precoders=baseline.precoders,
        signal_terms=baseline.signal_terms,
        rates=one_layer_rsma_rates(
            baseline.signal_terms,
            np.zeros_like(baseline.rates.common_rate_allocations),
        ),
    )


def equal_power_rzf(
    channels: ArrayLike, *, total_power: float, noise_variance: float
) -> OneLayerRSMABaseline:
    """Evaluate an equal-private-power RZF SDMA baseline with no common stream."""
    baseline = one_layer_rsma_baseline(
        channels,
        total_power=total_power,
        noise_variance=noise_variance,
        common_power_fraction=0.0,
        private_scheme="rzf",
        common_rate_shares=np.array([1.0 / np.asarray(channels).shape[0]] * np.asarray(channels).shape[0]),
    )
    return OneLayerRSMABaseline(
        name="equal_power_rzf_sdma",
        private_scheme=baseline.private_scheme,
        total_power=baseline.total_power,
        noise_variance=baseline.noise_variance,
        common_power_fraction=baseline.common_power_fraction,
        common_rate_shares=np.zeros_like(baseline.common_rate_shares),
        power_allocation=baseline.power_allocation,
        allocation=ResourceAllocation(
            baseline.allocation.common_power,
            baseline.allocation.private_powers,
            np.zeros_like(baseline.allocation.common_rate_allocations),
        ),
        precoders=baseline.precoders,
        signal_terms=baseline.signal_terms,
        rates=one_layer_rsma_rates(
            baseline.signal_terms,
            np.zeros_like(baseline.rates.common_rate_allocations),
        ),
    )


def fixed_common_rsma(
    channels: ArrayLike,
    *,
    total_power: float,
    noise_variance: float,
    common_power_fraction: float,
    private_scheme: str = "mrt",
    common_rate_shares: ArrayLike | None = None,
) -> OneLayerRSMABaseline:
    """Evaluate a fixed-common-power one-layer RSMA baseline."""
    return one_layer_rsma_baseline(
        channels,
        total_power=total_power,
        noise_variance=noise_variance,
        common_power_fraction=common_power_fraction,
        private_scheme=private_scheme,
        common_rate_shares=common_rate_shares,
    )
