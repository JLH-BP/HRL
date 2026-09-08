# Tests paired bootstrap and sign-flip inference on seed-level Meta-PPO results.
import json
from pathlib import Path

import pytest

from meta_hrl.training import analyze_meta_ppo_significance


def _payload() -> dict[str, object]:
    def split_records(offset: float) -> dict[str, object]:
        points = [{"adaptation_episodes": 0.0}, {"adaptation_episodes": 1.0}]
        learned = [[{"mean_reward": 3.0 + offset}, {"mean_reward": 5.0 + offset}], [{"mean_reward": 4.0 + offset}, {"mean_reward": 6.0 + offset}]]
        raw = [[{"mean_reward": 1.0 + offset}, {"mean_reward": 2.0 + offset}], [{"mean_reward": 2.0 + offset}, {"mean_reward": 3.0 + offset}]]
        for records in (learned, raw):
            for point in records:
                for item in point:
                    item.update({
                        "mean_sum_rate": item["mean_reward"], "mean_min_user_rate": item["mean_reward"],
                        "mean_jain_fairness": item["mean_reward"], "mean_total_qos_gap": item["mean_reward"],
                        "qos_satisfaction_rate": item["mean_reward"],
                    })
        return {"points": points, "learned": learned, "raw": raw}

    validation, ood = split_records(0.0), split_records(1.0)
    return {
        "variants": {
            "learned_context": {
                split: {"points": source["points"], "seed_points": source["learned"]}
                for split, source in (("validation", validation), ("ood", ood))
            },
            "raw_context": {
                split: {"points": source["points"], "seed_points": source["raw"]}
                for split, source in (("validation", validation), ("ood", ood))
            },
        }
    }


def test_meta_ppo_significance_reports_exact_paired_inference(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")

    result = analyze_meta_ppo_significance(source, bootstrap_samples=100, seed=9)

    report = json.loads(result.result_path.read_text(encoding="utf-8"))
    reward = report["validation"][0]["metrics"]["mean_reward"]
    assert reward["paired_seeds"] == 2
    assert reward["mean_difference"] == 2.5
    assert reward["permutation_method"] == "exact_sign_flip"
    assert reward["two_sided_p_value"] == 0.5


def test_meta_ppo_significance_rejects_invalid_inference_options(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ValueError, match="bootstrap_samples"):
        analyze_meta_ppo_significance(source, bootstrap_samples=0)
    with pytest.raises(ValueError, match="confidence"):
        analyze_meta_ppo_significance(source, confidence=1.0)
