# Tests CSV, SVG, and paired-difference Meta-PPO benchmark analysis artifacts.
import json
from pathlib import Path

import pytest

from hrl.training import analyze_meta_ppo_benchmark


def _payload() -> dict[str, object]:
    def point(budget: int, reward: float) -> dict[str, float]:
        return {
            "adaptation_episodes": float(budget),
            "mean_reward_mean": reward,
            "mean_reward_std": 0.2,
            "mean_sum_rate_mean": reward + 1.0,
            "mean_sum_rate_std": 0.1,
            "mean_min_user_rate_mean": reward / 2.0,
            "mean_min_user_rate_std": 0.05,
            "mean_jain_fairness_mean": 0.8,
            "mean_jain_fairness_std": 0.02,
            "mean_total_qos_gap_mean": 0.3,
            "mean_total_qos_gap_std": 0.01,
            "qos_satisfaction_rate_mean": 0.6,
            "qos_satisfaction_rate_std": 0.1,
            "mean_context_transitions_mean": 16.0,
            "mean_context_transitions_std": 0.0,
        }

    return {
        "variants": {
            "learned_context": {
                split: {"seeds": 2, "points": [point(0, 2.0), point(1, 3.0)]}
                for split in ("validation", "ood")
            },
            "raw_context": {
                split: {"seeds": 2, "points": [point(0, 1.5), point(1, 2.5)]}
                for split in ("validation", "ood")
            },
        }
    }


def test_meta_ppo_analysis_writes_csv_svg_and_paired_differences(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    result = analyze_meta_ppo_benchmark(source, output_directory=tmp_path / "analysis")

    assert result.csv_path.is_file()
    assert set(result.svg_paths) == {
        "validation_mean_reward", "validation_mean_sum_rate",
        "ood_mean_reward", "ood_mean_sum_rate",
    }
    assert all(path.read_text(encoding="utf-8").startswith("<svg") for path in result.svg_paths.values())
    paired = json.loads(result.paired_summary_path.read_text(encoding="utf-8"))
    assert paired["validation"][0]["mean_reward_difference"] == 0.5


def test_meta_ppo_analysis_rejects_missing_or_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_meta_ppo_benchmark(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="variants"):
        analyze_meta_ppo_benchmark(invalid)
