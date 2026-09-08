
"""bootstrap CI 和双侧 sign-flip 检验"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
import json

import numpy as np

__all__ = ["MetaPPOSignificanceResult", "analyze_meta_ppo_significance"]

_METRICS = (
    "mean_reward",
    "mean_sum_rate",
    "mean_min_user_rate",
    "mean_jain_fairness",
    "mean_total_qos_gap",
    "qos_satisfaction_rate",
)


@dataclass(frozen=True, slots=True)
class MetaPPOSignificanceResult:
    """Path to paired learned-minus-raw inference records."""

    result_path: Path


def _load_benchmark(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark JSON was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("benchmark_path must contain valid JSON.") from error
    variants = payload.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Benchmark JSON must contain a variants object.")
    if not {"learned_context", "raw_context"}.issubset(variants):
        raise ValueError("Benchmark JSON must contain learned_context and raw_context variants.")
    return payload


def _paired_values(payload: dict[str, Any], split: str, point_index: int, metric: str) -> np.ndarray:
    try:
        learned = payload["variants"]["learned_context"][split]["seed_points"][point_index]
        raw = payload["variants"]["raw_context"][split]["seed_points"][point_index]
    except (KeyError, IndexError) as error:
        raise ValueError(
            "Benchmark JSON must contain paired seed_points. Re-run the benchmark with the current pipeline."
        ) from error
    if not isinstance(learned, list) or not isinstance(raw, list) or len(learned) != len(raw) or not learned:
        raise ValueError("Each variant must have matching nonempty seed_points.")
    return np.asarray([float(left[metric]) - float(right[metric]) for left, right in zip(learned, raw)], dtype=np.float64)


def _bootstrap_interval(differences: np.ndarray, samples: int, confidence: float, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(differences, size=(samples, differences.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def _permutation_p_value(differences: np.ndarray, permutations: int, rng: np.random.Generator) -> tuple[float, str]:
    observed = abs(float(np.mean(differences)))
    if differences.size <= 16:
        signs = np.asarray(tuple(product((-1.0, 1.0), repeat=differences.size)), dtype=np.float64)
        null = np.abs((signs * differences).mean(axis=1))
        return float(np.mean(null >= observed - 1e-12)), "exact_sign_flip"
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(permutations, differences.size))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null >= observed - 1e-12) + 1) / (permutations + 1)), "monte_carlo_sign_flip"


def analyze_meta_ppo_significance(
    benchmark_path: Path | str,
    *,
    output_path: Path | str | None = None,
    bootstrap_samples: int = 10_000,
    monte_carlo_permutations: int = 100_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> MetaPPOSignificanceResult:
    """Report paired effect sizes, bootstrap CIs, and two-sided sign-flip p-values."""
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")
    if isinstance(monte_carlo_permutations, bool) or not isinstance(monte_carlo_permutations, int) or monte_carlo_permutations < 1:
        raise ValueError("monte_carlo_permutations must be a positive integer.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1).")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    source = Path(benchmark_path)
    payload = _load_benchmark(source)
    rng = np.random.default_rng(seed)
    records: dict[str, list[dict[str, object]]] = {}
    for split in ("validation", "ood"):
        try:
            points = payload["variants"]["learned_context"][split]["points"]
        except KeyError as error:
            raise ValueError(f"Benchmark JSON has no learned_context/{split} points.") from error
        records[split] = []
        for point_index, point in enumerate(points):
            item: dict[str, object] = {"adaptation_episodes": int(point["adaptation_episodes"]), "metrics": {}}
            for metric in _METRICS:
                differences = _paired_values(payload, split, point_index, metric)
                lower, upper = _bootstrap_interval(differences, bootstrap_samples, confidence, rng)
                p_value, method = _permutation_p_value(differences, monte_carlo_permutations, rng)
                item["metrics"][metric] = {
                    "paired_seeds": int(differences.size),
                    "mean_difference": float(np.mean(differences)),
                    "bootstrap_confidence": confidence,
                    "bootstrap_ci": [lower, upper],
                    "two_sided_p_value": p_value,
                    "permutation_method": method,
                }
            records[split].append(item)
    destination = source.parent / "paired_significance.json" if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
    return MetaPPOSignificanceResult(destination)
