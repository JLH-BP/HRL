# Tests CSV, SVG, and bootstrap trend reports for sensitivity experiments.
import json
from pathlib import Path

import pytest

from hrl.training import analyze_meta_ppo_sensitivity


def _cell(capacity: int, width: int, shift: float, reward: float) -> dict[str, object]:
    point = {"adaptation_episodes": 0.0, "mean_reward_mean": reward, "mean_reward_std": 0.1}
    seed_point = [{"mean_reward": reward - 0.1}, {"mean_reward": reward + 0.1}]
    return {
        "context_capacity": capacity, "context_feature_dim": width, "ood_shift_scale": shift,
        "seeds": 2, "points": [point], "seed_points": [seed_point],
    }


def test_sensitivity_analysis_writes_factor_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "sensitivity.json"
    source.write_text(json.dumps({"cells": [_cell(16, 32, 0.5, 2.0), _cell(64, 32, 1.0, 3.0)]}), encoding="utf-8")

    result = analyze_meta_ppo_sensitivity(source, output_directory=tmp_path / "analysis", bootstrap_samples=100)

    assert result.csv_path.is_file()
    assert set(result.svg_paths) == {"context_capacity", "context_feature_dim", "ood_shift_scale"}
    assert all(path.read_text(encoding="utf-8").startswith("<svg") for path in result.svg_paths.values())
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["metric"] == "mean_reward"
    assert summary["factors"]["context_capacity"][0]["level"] == 16.0


def test_sensitivity_analysis_requires_seed_level_results(tmp_path: Path) -> None:
    source = tmp_path / "sensitivity.json"
    source.write_text(json.dumps({"cells": [{"context_capacity": 16, "context_feature_dim": 32, "ood_shift_scale": 1.0, "seeds": 1, "points": [{}]}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="seed_points"):
        analyze_meta_ppo_sensitivity(source)
