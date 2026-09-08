# Tests parsing and artifact-generating CLI commands without training workloads.
import json
from pathlib import Path

from meta_hrl.cli import build_parser, main


def _benchmark_payload() -> dict[str, object]:
    point = {
        "adaptation_episodes": 0.0, "mean_reward_mean": 1.0, "mean_reward_std": 0.1,
        "mean_sum_rate_mean": 2.0, "mean_sum_rate_std": 0.1,
        "mean_min_user_rate_mean": 1.0, "mean_min_user_rate_std": 0.1,
        "mean_jain_fairness_mean": 1.0, "mean_jain_fairness_std": 0.1,
        "mean_total_qos_gap_mean": 1.0, "mean_total_qos_gap_std": 0.1,
        "qos_satisfaction_rate_mean": 1.0, "qos_satisfaction_rate_std": 0.1,
    }
    return {
        "variants": {
            variant: {split: {"points": [point], "seed_points": [[{
                "mean_reward": 1.0, "mean_sum_rate": 2.0, "mean_min_user_rate": 1.0,
                "mean_jain_fairness": 1.0, "mean_total_qos_gap": 1.0, "qos_satisfaction_rate": 1.0,
            }]]} for split in ("validation", "ood")}
            for variant in ("learned_context", "raw_context")
        }
    }


def test_cli_parses_csv_options() -> None:
    args = build_parser().parse_args([
        "sensitivity", "--seeds", "3,5", "--context-capacities", "8,16",
        "--ood-shift-scales", "0.5,1.5",
    ])
    assert args.seeds == (3, 5)
    assert args.context_capacities == (8, 16)
    assert args.ood_shift_scales == (0.5, 1.5)


def test_cli_parses_flat_training_options() -> None:
    args = build_parser().parse_args([
        "train-flat", "--timesteps", "1_000", "--evaluation-interval", "200",
    ])
    assert args.timesteps == 1_000
    assert args.evaluation_interval == 200


def test_cli_runs_analysis_and_significance_commands(tmp_path: Path, capsys) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(_benchmark_payload()), encoding="utf-8")

    assert main(["analyze-benchmark", str(source), "--output-directory", str(tmp_path / "analysis")]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()
    assert main(["significance", str(source), "--bootstrap-samples", "10"]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()
