# Tests multi-seed Meta-PPO benchmark aggregation and JSON artifacts.
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hrl.training import MetaPPOBenchmarkConfig, run_meta_ppo_benchmark


class ZeroMetaWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(18, dtype=np.float32), None


def test_meta_ppo_benchmark_aggregates_paired_variants_and_writes_json(tmp_path: Path) -> None:
    received_configs = []

    def fake_train(config):
        received_configs.append(config)
        return SimpleNamespace(model=ZeroMetaWorker())

    result = run_meta_ppo_benchmark(
        MetaPPOBenchmarkConfig(
            seeds=(17,),
            training_task_ids=(0,),
            evaluation_task_ids=(1,),
            total_timesteps=8,
            n_steps=8,
            batch_size=8,
            n_epochs=1,
            adaptation_episodes=(0, 1),
            evaluation_episodes=1,
            output_directory=tmp_path,
        ),
        train=fake_train,
    )

    assert [config.use_learned_context_encoder for config in received_configs] == [True, False]
    assert result.result_path.is_file()
    payload = json.loads(result.result_path.read_text(encoding="utf-8"))
    assert set(payload["variants"]) == {"learned_context", "raw_context"}
    assert len(payload["variants"]["learned_context"]["validation"]["points"]) == 2
    assert payload["variants"]["raw_context"]["ood"]["seeds"] == 1


def test_meta_ppo_benchmark_rejects_invalid_paired_experiment_settings() -> None:
    with np.testing.assert_raises_regex(ValueError, "seeds"):
        MetaPPOBenchmarkConfig(seeds=())
    with np.testing.assert_raises_regex(ValueError, "unique"):
        MetaPPOBenchmarkConfig(seeds=(1, 1))
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        MetaPPOBenchmarkConfig(adaptation_episodes=(1, 0))
