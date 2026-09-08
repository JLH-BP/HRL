"""导出基准 CSV、SVG 曲线和配对差异 JSON"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

__all__ = ["MetaPPOAnalysisResult", "analyze_meta_ppo_benchmark"]


@dataclass(frozen=True, slots=True)
class MetaPPOAnalysisResult:
    """Paths to tabular, visual, and paired-difference benchmark artifacts."""

    csv_path: Path
    svg_paths: dict[str, Path]
    paired_summary_path: Path


def _read_payload(benchmark_path: Path) -> dict[str, Any]:
    if not benchmark_path.is_file():
        raise FileNotFoundError(f"Benchmark JSON was not found: {benchmark_path}")
    try:
        payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("benchmark_path must contain valid JSON.") from error
    if not isinstance(payload.get("variants"), dict):
        raise ValueError("Benchmark JSON must contain a variants object.")
    required = {"learned_context", "raw_context"}
    if not required.issubset(payload["variants"]):
        raise ValueError("Benchmark JSON must contain learned_context and raw_context variants.")
    return payload


def _points(payload: dict[str, Any], variant: str, split: str) -> list[dict[str, Any]]:
    try:
        points = payload["variants"][variant][split]["points"]
    except KeyError as error:
        raise ValueError(f"Benchmark JSON has no {variant}/{split} points.") from error
    if not isinstance(points, list) or not points:
        raise ValueError(f"Benchmark JSON has invalid {variant}/{split} points.")
    return points


def _write_csv(payload: dict[str, Any], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for variant, splits in payload["variants"].items():
        if not isinstance(splits, dict):
            continue
        for split, record in splits.items():
            for point in record.get("points", []):
                rows.append({"variant": variant, "split": split, **point})
    if not rows:
        raise ValueError("Benchmark JSON contains no aggregate points.")
    fields = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _line_points(points: list[dict[str, Any]], metric: str) -> tuple[list[float], list[float], list[float]]:
    suffix = f"{metric}_mean"
    std_suffix = f"{metric}_std"
    try:
        x = [float(point["adaptation_episodes"]) for point in points]
        y = [float(point[suffix]) for point in points]
        std = [float(point[std_suffix]) for point in points]
    except KeyError as error:
        raise ValueError(f"Benchmark point is missing metric {metric}.") from error
    return x, y, std


def _svg_curve(split: str, metric: str, learned: list[dict[str, Any]], raw: list[dict[str, Any]]) -> str:
    """Render a compact, portable SVG without making matplotlib a project dependency."""
    width, height, margin = 720, 430, 70
    lx, ly, ls = _line_points(learned, metric)
    rx, ry, rs = _line_points(raw, metric)
    all_x, all_y = lx + rx, [value + spread for value, spread in zip(ly, ls)] + [value + spread for value, spread in zip(ry, rs)]
    lower = [value - spread for value, spread in zip(ly, ls)] + [value - spread for value, spread in zip(ry, rs)]
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max:
        x_min, x_max = x_min - 1.0, x_max + 1.0
    y_min, y_max = min(lower), max(all_y)
    if y_min == y_max:
        y_min, y_max = y_min - 1.0, y_max + 1.0
    padding = max((y_max - y_min) * 0.08, 1e-9)
    y_min -= padding
    y_max += padding

    def px(value: float) -> float:
        return margin + (value - x_min) * (width - 2 * margin) / (x_max - x_min)

    def py(value: float) -> float:
        return height - margin - (value - y_min) * (height - 2 * margin) / (y_max - y_min)

    def path(values: list[float], xs: list[float]) -> str:
        return " ".join(("M" if index == 0 else "L") + f" {px(x):.2f} {py(y):.2f}" for index, (x, y) in enumerate(zip(xs, values)))

    def band(values: list[float], spreads: list[float], xs: list[float]) -> str:
        upper = [(px(x), py(y + spread)) for x, y, spread in zip(xs, values, spreads)]
        lower_points = [(px(x), py(y - spread)) for x, y, spread in zip(xs, values, spreads)][::-1]
        points = upper + lower_points
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    x_ticks = "".join(f'<text x="{px(x):.2f}" y="{height - margin + 24}" text-anchor="middle">{x:g}</text>' for x in lx)
    y_ticks = "".join(f'<text x="{margin - 12}" y="{py(y):.2f}" text-anchor="end">{y:.3g}</text>' for y in np.linspace(y_min, y_max, 5))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>text{{font:14px sans-serif;fill:#202020}} .axis{{stroke:#333;stroke-width:1}} .grid{{stroke:#ddd;stroke-width:1}} .learned{{stroke:#1769aa;fill:none;stroke-width:3}} .raw{{stroke:#d65f00;fill:none;stroke-width:3}}</style>
<rect width="100%" height="100%" fill="white"/><text x="{width / 2}" y="28" text-anchor="middle" font-weight="bold">{split}: {metric}</text>
<line class="axis" x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}"/><line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}"/>
<polygon points="{band(ly, ls, lx)}" fill="#1769aa" opacity="0.15"/><polygon points="{band(ry, rs, rx)}" fill="#d65f00" opacity="0.15"/>
<path class="learned" d="{path(ly, lx)}"/><path class="raw" d="{path(ry, rx)}"/>
<text x="{width / 2}" y="{height - 15}" text-anchor="middle">support episodes</text><text x="16" y="{height / 2}" text-anchor="middle" transform="rotate(-90 16 {height / 2})">{metric}</text>
{x_ticks}{y_ticks}<rect x="{width - 220}" y="44" width="165" height="48" fill="white" stroke="#aaa"/><line class="learned" x1="{width - 208}" y1="61" x2="{width - 178}" y2="61"/><text x="{width - 170}" y="66">learned context</text><line class="raw" x1="{width - 208}" y1="82" x2="{width - 178}" y2="82"/><text x="{width - 170}" y="87">raw context</text>
</svg>'''


def _paired_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute aggregate learned-minus-raw effects for every split and budget."""
    results: dict[str, list[dict[str, float]]] = {}
    for split in ("validation", "ood"):
        learned, raw = _points(payload, "learned_context", split), _points(payload, "raw_context", split)
        if [point["adaptation_episodes"] for point in learned] != [point["adaptation_episodes"] for point in raw]:
            raise ValueError(f"{split} variants use incompatible adaptation budgets.")
        results[split] = []
        for learned_point, raw_point in zip(learned, raw):
            record: dict[str, float] = {"adaptation_episodes": float(learned_point["adaptation_episodes"])}
            for metric in ("mean_reward", "mean_sum_rate", "mean_min_user_rate", "mean_jain_fairness", "mean_total_qos_gap", "qos_satisfaction_rate"):
                record[f"{metric}_difference"] = float(learned_point[f"{metric}_mean"]) - float(raw_point[f"{metric}_mean"])
            results[split].append(record)
    return results


def analyze_meta_ppo_benchmark(
    benchmark_path: Path | str,
    *,
    output_directory: Path | str | None = None,
    metrics: tuple[str, ...] = ("mean_reward", "mean_sum_rate"),
) -> MetaPPOAnalysisResult:
    """Export benchmark tables, uncertainty-band SVG curves, and ablation deltas."""
    source = Path(benchmark_path)
    payload = _read_payload(source)
    if not metrics:
        raise ValueError("metrics must contain at least one aggregate metric.")
    output = source.parent / "analysis" if output_directory is None else Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "meta_ppo_benchmark.csv"
    _write_csv(payload, csv_path)
    svg_paths: dict[str, Path] = {}
    for split in ("validation", "ood"):
        learned, raw = _points(payload, "learned_context", split), _points(payload, "raw_context", split)
        for metric in metrics:
            path = output / f"{split}_{metric}.svg"
            path.write_text(_svg_curve(split, metric, learned, raw), encoding="utf-8")
            svg_paths[f"{split}_{metric}"] = path
    paired_summary_path = output / "paired_learned_minus_raw.json"
    paired_summary_path.write_text(json.dumps(_paired_summary(payload), indent=2, sort_keys=True), encoding="utf-8")
    return MetaPPOAnalysisResult(csv_path, svg_paths, paired_summary_path)
