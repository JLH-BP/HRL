# Tests high-level Manager interval rollout under a frozen Worker.
import numpy as np

from hrl.envs.hierarchical_env import HierarchicalRSMAEnvConfig
from hrl.envs.manager_training_env import ManagerTrainingEnv
from hrl.envs.task_sampler import ScenarioSamplerConfig


class ZeroWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(18, dtype=np.float32), None


def test_manager_environment_rolls_one_full_high_level_interval() -> None:
    environment = ManagerTrainingEnv(
        ZeroWorker(), HierarchicalRSMAEnvConfig(high_level_interval=2, episode_length=5)
    )
    observation, info = environment.reset(seed=13)
    assert environment.observation_space.contains(observation)
    assert info["manager_action_required"]

    next_observation, reward, terminated, truncated, info = environment.step(0)
    assert environment.observation_space.contains(next_observation)
    assert info["manager_interval_steps"] == 2
    assert info["manager_interval_reward"] == reward
    assert not terminated
    assert not truncated


def test_first_manager_choice_does_not_pay_switch_penalty() -> None:
    environment = ManagerTrainingEnv(ZeroWorker())
    environment.reset(seed=5)
    _, _, _, _, info = environment.step(0)
    assert not info["partition_switched"]
    assert info["switch_penalty"] == 0.0


def test_manager_environment_decodes_partition_and_service_mode_together() -> None:
    environment = ManagerTrainingEnv(
        ZeroWorker(),
        HierarchicalRSMAEnvConfig(
            sampler=ScenarioSamplerConfig(near_field_user_count_range=(3, 3)),
            high_level_interval=1,
            episode_length=2,
        ),
    )
    environment.reset(seed=19)
    action = environment.hierarchical_env.encode_manager_action(0, "near")
    _, _, _, _, info = environment.step(action)

    assert environment.action_space.contains(action)
    assert info["manager_candidate_index"] == 0
    assert info["manager_service_mode"] == "near"
    np.testing.assert_array_equal(
        info["service_mask"], environment.hierarchical_env.scenario.near_field_mask
    )
