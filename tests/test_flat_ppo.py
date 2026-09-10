# Smoke-tests Flat PPO training and deterministic baseline evaluation.
from pathlib import Path

import numpy as np
import pytest

from hrl.training.evaluate import evaluate_flat_policy, evaluate_physical_baselines
from hrl.training.train_flat_rl import FlatPPOConfig, train_flat_ppo


def test_physical_baseline_evaluation_is_reproducible() -> None:
    first = evaluate_physical_baselines(episodes=3, seed=71)
    second = evaluate_physical_baselines(episodes=3, seed=71)

    assert set(first) == {"equal_power_rzf", "fixed_common_rzf_rsma"}
    assert first == second
    assert first["equal_power_rzf"].episodes == 3


@pytest.mark.filterwarnings("ignore:.*truncated mini-batch.*")
def test_flat_ppo_smoke_train_save_load_and_evaluate(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    config = FlatPPOConfig(
        total_timesteps=8,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_net_arch=(16,),
        checkpoint_directory=tmp_path / "checkpoints",
        log_directory=tmp_path / "logs",
        evaluation_directory=tmp_path / "evaluations",
        evaluation_interval=8,
        evaluation_episodes=2,
    )
    result = train_flat_ppo(config)

    assert result.model_path.is_file()
    assert result.evaluation_csv_path.is_file()
    assert result.evaluation_svg_path.is_file()
    assert "timesteps,mean_reward" in result.evaluation_csv_path.read_text(encoding="utf-8")
    action, _ = result.model.predict(np.zeros(1572, dtype=np.float32), deterministic=True)
    assert action.shape == (13,)
    metrics = evaluate_flat_policy(result.model, episodes=2, seed=19)
    assert metrics.episodes == 2
    assert np.isfinite(metrics.mean_reward)
