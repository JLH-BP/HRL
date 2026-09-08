# Tests reproducible mixed-field scenario sampling.
import numpy as np

from meta_hrl.envs.task_sampler import TaskSampler


def test_task_sampler_seed_reproduces_complete_scenario() -> None:
    sampler = TaskSampler()
    sampler.reset(seed=11)
    first = sampler.sample()
    sampler.reset(seed=11)
    second = sampler.sample()

    np.testing.assert_array_equal(first.channels, second.channels)
    np.testing.assert_array_equal(first.ranges_m, second.ranges_m)
    np.testing.assert_array_equal(first.angles_rad, second.angles_rad)
    np.testing.assert_array_equal(first.near_field_mask, second.near_field_mask)
    assert first.channels.shape == (6, 128)
    assert first.qos_rate_targets.shape == (6,)
