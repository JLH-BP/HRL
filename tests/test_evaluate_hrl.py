# Tests staged HRL Worker evaluation metrics.
import numpy as np

from hrl.agents.high_level_policy import FixedPartitionManager, NearFieldFirstSequentialManager
from hrl.training.evaluate_hrl import evaluate_staged_hrl_worker


class ZeroWorker:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(18, dtype=np.float32), None


def test_staged_hrl_evaluation_reports_manager_diagnostics() -> None:
    metrics = evaluate_staged_hrl_worker(
        ZeroWorker(), manager=FixedPartitionManager(), episodes=2, seed=31
    )
    assert metrics.episodes == 2
    assert metrics.mean_manager_decisions >= 1.0
    assert metrics.mean_partition_switches == 0.0
    assert np.isfinite(metrics.mean_reward)


def test_staged_hrl_evaluation_accepts_near_field_first_manager() -> None:
    metrics = evaluate_staged_hrl_worker(
        ZeroWorker(), manager=NearFieldFirstSequentialManager(), episodes=2, seed=37
    )
    assert metrics.episodes == 2
    assert metrics.mean_partition_switches == 0.0
    assert np.isfinite(metrics.mean_sum_rate)
