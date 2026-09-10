# Tests one-step Gymnasium RSMA resource-allocation behavior.
import numpy as np
import pytest

from hrl.envs.rsma_env import OneStepRSMAEnv, jain_fairness


def test_one_step_environment_returns_feasible_rate_diagnostics() -> None:
    environment = OneStepRSMAEnv()
    observation, reset_info = environment.reset(seed=7)
    next_observation, reward, terminated, truncated, info = environment.step(
        np.linspace(-1.0, 1.0, 13, dtype=np.float32)
    )

    assert environment.observation_space.contains(observation)
    assert environment.observation_space.contains(next_observation)
    assert set(reset_info) >= {"ranges_m", "near_field_mask", "qos_rate_targets"}
    assert np.isfinite(reward)
    assert terminated
    assert not truncated
    assert info["power_allocation"].common_power + info["power_allocation"].private_powers.sum() == pytest.approx(1.0)
    assert info["allocation"].common_rate_allocations.sum() == pytest.approx(info["common_rate"])
    assert info["sum_rate"] == pytest.approx(info["user_rates"].sum())


def test_one_step_environment_is_reset_seed_reproducible() -> None:
    environment = OneStepRSMAEnv()
    first_observation, _ = environment.reset(seed=23)
    second_observation, _ = environment.reset(seed=23)

    np.testing.assert_array_equal(first_observation, second_observation)


def test_one_step_environment_rejects_second_step_and_invalid_actions() -> None:
    environment = OneStepRSMAEnv()
    environment.reset(seed=2)
    with pytest.raises(ValueError):
        environment.step(np.zeros(12, dtype=np.float32))
    environment.step(np.zeros(13, dtype=np.float32))
    with pytest.raises(RuntimeError):
        environment.step(np.zeros(13, dtype=np.float32))


def test_jain_fairness_handles_balanced_and_zero_rates() -> None:
    assert jain_fairness([2.0, 2.0]) == pytest.approx(1.0)
    assert jain_fairness([0.0, 0.0]) == 0.0
