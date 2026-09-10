# Smoke-tests discrete PPO Manager training under a frozen zero Worker.
from pathlib import Path

import numpy as np
import pytest

from hrl.training.train_hrl import ManagerPPOConfig, train_manager_ppo


class ZeroWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(18, dtype=np.float32), None


@pytest.mark.filterwarnings("ignore:.*truncated mini-batch.*")
def test_manager_ppo_smoke_train_and_save(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    result = train_manager_ppo(
        ManagerPPOConfig(
            worker=ZeroWorker(),
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
    action, _ = result.model.predict(
        np.zeros(result.model.observation_space.shape, dtype=np.float32), deterministic=True
    )
    assert isinstance(int(action), int)
