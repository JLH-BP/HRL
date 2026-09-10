# Smoke-tests task cycling and context-conditioned Meta-PPO Worker training.
from pathlib import Path

import numpy as np
import pytest

from hrl.training import MetaPPOConfig, TaskCyclingContextualWorkerEnv, train_meta_ppo


def test_task_cycling_environment_keeps_task_contexts_isolated() -> None:
    environment = TaskCyclingContextualWorkerEnv(task_ids=(3, 7), task_seed=19)
    _, first_info = environment.reset(seed=31)
    assert first_info["meta_task_id"] == 3
    assert first_info["meta_episode_index"] == 0
    environment.step(np.zeros(18, dtype=np.float32))

    _, second_info = environment.reset(seed=32)
    assert second_info["meta_task_id"] == 7
    assert second_info["meta_episode_index"] == 1
    assert len(environment.context_buffer.recent(3)) == 1
    assert len(environment.context_buffer.recent(7)) == 0

    _, third_info = environment.reset(seed=33)
    assert third_info["meta_task_id"] == 3
    assert third_info["meta_episode_index"] == 2
    assert len(environment.context_buffer.recent(3)) == 1
    environment.close()


def test_meta_ppo_config_rejects_invalid_task_and_rollout_settings() -> None:
    with pytest.raises(ValueError, match="task_ids"):
        MetaPPOConfig(task_ids=())
    with pytest.raises(ValueError, match="unique"):
        MetaPPOConfig(task_ids=(1, 1))
    with pytest.raises(ValueError, match="batch_size"):
        MetaPPOConfig(n_steps=8, batch_size=3)
    with pytest.raises(ValueError, match="split"):
        MetaPPOConfig(split="test")


def test_task_cycling_environment_rejects_reset_options() -> None:
    environment = TaskCyclingContextualWorkerEnv()
    with pytest.raises(ValueError, match="reset options"):
        environment.reset(options={"task_id": 2})
    environment.close()


@pytest.mark.filterwarnings("ignore:.*truncated mini-batch.*")
def test_meta_ppo_smoke_train_and_save(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    result = train_meta_ppo(
        MetaPPOConfig(
            task_ids=(0, 1),
            total_timesteps=8,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            policy_net_arch=(16,),
            checkpoint_directory=tmp_path / "checkpoints",
            log_directory=tmp_path / "logs",
        )
    )

    assert result.model_path.is_file()
    assert result.model.policy.features_extractor.__class__.__name__ == "ContextConditionedFeaturesExtractor"
    action, _ = result.model.predict(np.zeros(1639, dtype=np.float32), deterministic=True)
    assert action.shape == (18,)
