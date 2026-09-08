# Tests one-layer RSMA resource constraints and action projection utilities.
import numpy as np
import pytest

from meta_hrl.rsma.constraints import (
    ResourceAllocation,
    project_action_to_resource_allocation,
    project_nonnegative_simplex,
    validate_resource_allocation,
)


def test_validate_feasible_allocation_reports_correct_slack() -> None:
    allocation = ResourceAllocation(
        common_power=0.2,
        private_powers=np.array([0.1, 0.3]),
        common_rate_allocations=np.array([0.4, 0.2]),
    )

    feasibility = validate_resource_allocation(
        allocation, total_power=1.0, common_rate_budget=1.0
    )

    assert feasibility.is_feasible
    assert feasibility.allocated_power == pytest.approx(0.6)
    assert feasibility.power_slack == pytest.approx(0.4)
    assert feasibility.allocated_common_rate == pytest.approx(0.6)
    assert feasibility.common_rate_slack == pytest.approx(0.4)


def test_validate_boundary_and_over_budget_allocations() -> None:
    boundary = ResourceAllocation(
        common_power=0.25,
        private_powers=np.array([0.25, 0.5]),
        common_rate_allocations=np.array([0.3, 0.7]),
    )
    over_budget = ResourceAllocation(
        common_power=0.25,
        private_powers=np.array([0.25, 0.6]),
        common_rate_allocations=np.array([0.3, 0.8]),
    )

    assert validate_resource_allocation(
        boundary, total_power=1.0, common_rate_budget=1.0
    ).is_feasible
    result = validate_resource_allocation(
        over_budget, total_power=1.0, common_rate_budget=1.0
    )
    assert not result.is_feasible
    assert result.power_slack < 0.0
    assert result.common_rate_slack < 0.0


def test_zero_common_rate_budget_requires_zero_allocation() -> None:
    zero_rates = ResourceAllocation(0.2, np.array([0.8]), np.array([0.0]))
    nonzero_rates = ResourceAllocation(0.2, np.array([0.8]), np.array([0.1]))

    assert validate_resource_allocation(
        zero_rates, total_power=1.0, common_rate_budget=0.0
    ).is_feasible
    assert not validate_resource_allocation(
        nonzero_rates, total_power=1.0, common_rate_budget=0.0
    ).is_feasible


def test_simplex_projection_preserves_feasible_values_and_repairs_invalid_values() -> None:
    feasible = np.array([0.1, 0.2, 0.3])
    projected_feasible = project_nonnegative_simplex(feasible, budget=1.0)
    projected_invalid = project_nonnegative_simplex([-2.0, 1.0, 4.0], budget=1.0)

    np.testing.assert_array_equal(projected_feasible, feasible)
    assert np.all(projected_invalid >= 0.0)
    assert projected_invalid.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(project_nonnegative_simplex([3.0, -1.0], budget=0.0), [0.0, 0.0])


def test_logits_action_projection_is_feasible_for_six_users() -> None:
    action = np.linspace(-2.0, 2.0, 13)

    allocation = project_action_to_resource_allocation(
        action, num_users=6, total_power=1.0, common_rate_budget=3.5
    )
    feasibility = validate_resource_allocation(
        allocation, total_power=1.0, common_rate_budget=3.5
    )

    assert allocation.private_powers.shape == (6,)
    assert allocation.common_rate_allocations.shape == (6,)
    assert allocation.common_power + allocation.private_powers.sum() == pytest.approx(1.0)
    assert allocation.common_rate_allocations.sum() == pytest.approx(3.5)
    assert feasibility.is_feasible


def test_logits_action_projection_handles_zero_rate_budget_and_is_deterministic() -> None:
    action = np.arange(13, dtype=np.float64)

    first = project_action_to_resource_allocation(
        action, num_users=6, total_power=1.0, common_rate_budget=0.0
    )
    second = project_action_to_resource_allocation(
        action, num_users=6, total_power=1.0, common_rate_budget=0.0
    )

    np.testing.assert_array_equal(first.private_powers, second.private_powers)
    np.testing.assert_array_equal(first.common_rate_allocations, np.zeros(6))
    assert first.common_power == second.common_power


@pytest.mark.parametrize(
    "kwargs",
    [
        {"common_power": -0.1, "private_powers": [0.1], "common_rate_allocations": [0.0]},
        {"common_power": 0.1, "private_powers": [-0.1], "common_rate_allocations": [0.0]},
        {"common_power": 0.1, "private_powers": [0.1], "common_rate_allocations": [-0.1]},
        {"common_power": 0.1, "private_powers": [0.1], "common_rate_allocations": [0.0, 0.1]},
        {"common_power": 0.1, "private_powers": [], "common_rate_allocations": []},
    ],
)
def test_resource_allocation_rejects_invalid_data(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResourceAllocation(**kwargs)


@pytest.mark.parametrize(
    "action",
    [np.zeros(12), np.full(13, np.nan), np.full(13, np.inf), np.ones(13, dtype=bool), np.ones(13) + 1j],
)
def test_action_projection_rejects_invalid_actions(action: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        project_action_to_resource_allocation(
            action, num_users=6, total_power=1.0, common_rate_budget=1.0
        )


@pytest.mark.parametrize("values", [[np.nan], [True], [1.0 + 1.0j], []])
def test_simplex_projection_rejects_invalid_vectors(values: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        project_nonnegative_simplex(values, budget=1.0)
