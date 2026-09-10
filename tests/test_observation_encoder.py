# Tests the fixed-length CSI and geometry observation encoder.
import numpy as np

from hrl.envs.observation_encoder import encode_observation, observation_size
from hrl.envs.task_sampler import TaskSampler


def test_observation_encoder_returns_finite_expected_flat_shape() -> None:
    sampler = TaskSampler()
    sampler.reset(seed=3)
    observation = encode_observation(sampler.sample())

    assert observation.dtype == np.float32
    assert observation.shape == (observation_size(6, 128),)
    assert np.all(np.isfinite(observation))
