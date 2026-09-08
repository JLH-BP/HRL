# Tests three-factor learned-context sensitivity experiments and OOD severity.
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

from meta_hrl.envs.meta_task_sampler import MetaTaskSampler
from meta_hrl.training import MetaPPOSensitivityConfig, run_meta_ppo_sensitivity


class ZeroMetaWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(18, dtype=np.float32), None


def test_ood_shift_scale_changes_task_distribution_reproducibly() -> None:
    mild = MetaTaskSampler(ood_shift_scale=0.5).sample(split="ood", task_id=2, seed=31)
    severe = MetaTaskSampler(ood_shift_scale=1.5).sample(split="ood", task_id=2, seed=31)

    assert severe.noise_variance > mild.noise_variance
    assert severe.sampler_config.k_factor_linear > mild.sampler_config.k_factor_linear


def test_meta_ppo_sensitivity_runs_grid_and_writes_json(tmp_path: Path) -> None:
    received = []

    def fake_train(config):
        received.append(config)
        return SimpleNamespace(model=ZeroMetaWorker())

    result = run_meta_ppo_sensitivity(
        MetaPPOSensitivityConfig(
            seeds=(7,), training_task_ids=(0,), evaluation_task_ids=(1,),
            context_capacities=(2,), context_feature_dims=(4,), ood_shift_scales=(0.5, 1.5),
            total_timesteps=8, n_steps=8, batch_size=8, n_epochs=1,
            adaptation_episodes=(0,), evaluation_episodes=1, output_directory=tmp_path,
        ),
        train=fake_train,
    )

    assert [config.context_capacity_per_task for config in received] == [2]
    assert [config.context_feature_dim for config in received] == [4]
    assert len(result.cells) == 2
    assert result.result_path.is_file()
    assert len(json.loads(result.result_path.read_text(encoding="utf-8"))["cells"]) == 2
