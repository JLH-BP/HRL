# Smoke-tests staged group-RSMA Worker PPO training.
from pathlib import Path

import numpy as np
import pytest

from meta_hrl.training.train_hrl import StagedHRLConfig, train_staged_hrl_worker


@pytest.mark.filterwarnings("ignore:.*truncated mini-batch.*")
def test_staged_hrl_worker_ppo_smoke_train(tmp_path: Path) -> None:
    pytest.importorskip("stable_baselines3")
    config = StagedHRLConfig(
        total_timesteps=8,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        policy_net_arch=(16,),
        checkpoint_directory=tmp_path / "checkpoints",
        log_directory=tmp_path / "logs",
    )
    result = train_staged_hrl_worker(config)

    assert result.model_path.is_file()
    action, _ = result.model.predict(np.zeros(1614, dtype=np.float32), deterministic=True)
    assert action.shape == (18,)
