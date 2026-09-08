# Tests meta-task sampling, task-isolated transition context, and conditional observations.
import numpy as np

from meta_hrl.agents.meta_context import META_CONTEXT_SIZE, MetaContextEncoder
from meta_hrl.agents.replay_buffer import TaskContextBuffer
from meta_hrl.envs.contextual_worker_env import ContextualWorkerTrainingEnv
from meta_hrl.envs.meta_task_sampler import MetaTaskSampler


def test_meta_task_sampling_is_reproducible_and_ood_separated() -> None:
    sampler = MetaTaskSampler()
    first = sampler.sample(split="train", task_id=2, seed=41)
    second = sampler.sample(split="train", task_id=2, seed=41)
    ood = sampler.sample(split="ood", task_id=2, seed=41)

    assert first.sampler_config == second.sampler_config
    assert first.noise_variance == second.noise_variance
    assert 3.0 <= first.sampler_config.k_factor_linear <= 15.0
    assert ood.sampler_config.k_factor_linear >= 25.0
    assert ood.noise_variance >= 1.5


def test_context_buffer_is_task_isolated_and_encoder_handles_empty_context() -> None:
    buffer = TaskContextBuffer(capacity_per_task=2)
    encoder = MetaContextEncoder()
    np.testing.assert_array_equal(encoder.encode(()), np.zeros(META_CONTEXT_SIZE, dtype=np.float32))
    buffer.add(task_id=1, observation=[1.0, 2.0], action=[0.0], reward=1.0, next_observation=[2.0, 3.0], terminated=False)
    buffer.add(task_id=2, observation=[3.0, 4.0], action=[1.0], reward=2.0, next_observation=[4.0, 5.0], terminated=True)

    assert len(buffer.recent(1)) == 1
    assert len(buffer.recent(2)) == 1
    assert encoder.encode(buffer.recent(1)).shape == (META_CONTEXT_SIZE,)


def test_contextual_worker_appends_and_updates_task_context() -> None:
    environment = ContextualWorkerTrainingEnv()
    observation, info = environment.reset(seed=9, options={"task_id": 4})
    assert environment.observation_space.contains(observation)
    assert info["meta_task_id"] == 4
    before_context = observation[-META_CONTEXT_SIZE:].copy()
    next_observation, _, _, _, step_info = environment.step(np.zeros(18, dtype=np.float32))

    assert step_info["meta_context_size"] == 1
    assert environment.observation_space.contains(next_observation)
    assert np.any(next_observation[-META_CONTEXT_SIZE:] != before_context)
