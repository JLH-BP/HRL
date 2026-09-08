"""导出敏感性 CSV、SVG 因子趋势和摘要 JSON"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

__all__ = ["MetaPPOSensitivityAnalysisResult", "analyze_meta_ppo_sensitivity"]


@dataclass(frozen=True, slots=True)
class MetaPPOSensitivityAnalysisResult:
    """Paths to flat tables, SVG factor plots, and bootstrap trend summaries."""

    csv_path: Path
    svg_paths: dict[str, Path]
    summary_path: Path


def _payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Sensitivity JSON was not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("sensitivity_path must contain valid JSON.") from error
    if not isinstance(payload.get("cells"), list) or not payload["cells"]:
        raise ValueError("Sensitivity JSON must contain nonempty cells.")
    return payload


def _csv(cells: list[dict[str, Any]], path: Path) -> None:
    rows = [
        {
            "context_capacity": cell["context_capacity"],
            "context_feature_dim": cell["context_feature_dim"],
            "ood_shift_scale": cell["ood_shift_scale"],
            "seeds": cell["seeds"],
            **point,
        }
        for cell in cells for point in cell.get("points", [])
    ]
    if not rows:
        raise ValueError("Sensitivity JSON contains no aggregate points.")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _terminal_seed_values(cells: list[dict[str, Any]], metric: str) -> dict[tuple[int, int, float], np.ndarray]:
    values: dict[tuple[int, int, float], np.ndarray] = {}
    for cell in cells:
        seed_points = cell.get("seed_points")
        if not isinstance(seed_points, list) or not seed_points:
            raise ValueError("Sensitivity JSON must contain seed_points; re-run the current sensitivity pipeline.")
        terminal = seed_points[-1]
        key = (int(cell["context_capacity"]), int(cell["context_feature_dim"]), float(cell["ood_shift_scale"]))
        values[key] = np.asarray([float(point[metric]) for point in terminal], dtype=np.float64)
    return values


def _bootstrap(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float, float]:
    means = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return float(np.mean(values)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _factor_summary(cells: list[dict[str, Any]], metric: str, samples: int, seed: int) -> dict[str, list[dict[str, float]]]:
    values = _terminal_seed_values(cells, metric)
    rng = np.random.default_rng(seed)
    groups = {
        "context_capacity": lambda key: key[0],
        "context_feature_dim": lambda key: key[1],
        "ood_shift_scale": lambda key: key[2],
    }
    summary: dict[str, list[dict[str, float]]] = {}
    for factor, selector in groups.items():
        grouped: dict[float, list[np.ndarray]] = {}
        for key, result in values.items():
            grouped.setdefault(float(selector(key)), []).append(result)
        summary[factor] = []
        for level, arrays in sorted(grouped.items()):
            mean, lower, upper = _bootstrap(np.concatenate(arrays), samples, rng)
            summary[factor].append({"level": level, "mean": mean, "bootstrap_ci": [lower, upper]})
    return summary


def _svg(factor: str, records: list[dict[str, float]], metric: str) -> str:
    width, height, margin = 640, 400, 64
    xs = [record["level"] for record in records]
    ys = [record["mean"] for record in records]
    lo = [record["bootstrap_ci"][0] for record in records]
    hi = [record["bootstrap_ci"][1] for record in records]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    y_min, y_max = min(lo), max(hi)
    padding = max((y_max - y_min) * 0.1, 1e-9)
    y_min, y_max = y_min - padding, y_max + padding
    px = lambda value: margin + (value - x_min) * (width - 2 * margin) / (x_max - x_min)
    py = lambda value: height - margin - (value - y_min) * (height - 2 * margin) / (y_max - y_min)
    upper = [(px(x), py(y)) for x, y in zip(xs, hi)]
    lower = [(px(x), py(y)) for x, y in zip(xs, lo)][::-1]
    band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper + lower)
    line = " ".join(("M" if index == 0 else "L") + f" {px(x):.2f} {py(y):.2f}" for index, (x, y) in enumerate(zip(xs, ys)))
    ticks = "".join(f'<text x="{px(x):.2f}" y="{height - margin + 22}" text-anchor="middle">{x:g}</text>' for x in xs)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><style>text{{font:14px sans-serif;fill:#202020}}.axis{{stroke:#333}}.line{{stroke:#1769aa;fill:none;stroke-width:3}}</style><rect width="100%" height="100%" fill="white"/><text x="{width / 2}" y="26" text-anchor="middle">terminal adaptation: {metric} by {factor}</text><line class="axis" x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"/><line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}"/><polygon points="{band}" fill="#1769aa" opacity="0.18"/><path class="line" d="{line}"/>{ticks}<text x="{width / 2}" y="{height - 12}" text-anchor="middle">{factor}</text></svg>'''


def analyze_meta_ppo_sensitivity(
    sensitivity_path: Path | str,
    *,
    output_directory: Path | str | None = None,
    metric: str = "mean_reward",
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> MetaPPOSensitivityAnalysisResult:
    """Export a factor-grid table and terminal-adaptation trend CIs for one metric."""
    if not metric or not isinstance(metric, str):
        raise ValueError("metric must be a nonempty metric name.")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer.")
    source = Path(sensitivity_path)
    payload = _payload(source)
    output = source.parent / "analysis" if output_directory is None else Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "meta_ppo_sensitivity.csv"
    _csv(payload["cells"], csv_path)
    summary = _factor_summary(payload["cells"], metric, bootstrap_samples, seed)
    summary_path = output / "terminal_factor_trends.json"
    summary_path.write_text(json.dumps({"metric": metric, "factors": summary}, indent=2, sort_keys=True), encoding="utf-8")
    svg_paths: dict[str, Path] = {}
    for factor, records in summary.items():
        path = output / f"terminal_{metric}_by_{factor}.svg"
        path.write_text(_svg(factor, records, metric), encoding="utf-8")
        svg_paths[factor] = path
    return MetaPPOSensitivityAnalysisResult(csv_path, svg_paths, summary_path)
