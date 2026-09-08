"""Create line charts from a numeric CSV table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read_rows(source: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    if not source.is_file():
        raise FileNotFoundError(f"CSV file was not found: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV file has no header row.")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV file contains no data rows.")
    return rows, tuple(reader.fieldnames)


def _is_numeric(rows: list[dict[str, str]], column: str) -> bool:
    try:
        for row in rows:
            float(row[column])
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _columns(rows: list[dict[str, str]], headers: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    numeric = tuple(header for header in headers if _is_numeric(rows, header))
    categorical = tuple(header for header in headers if header not in numeric)
    if not numeric:
        raise ValueError("CSV file has no fully numeric columns to plot.")
    return numeric, categorical


def _plot(
    rows: list[dict[str, str]], *, x_column: str, y_columns: tuple[str, ...], group_column: str | None,
    output_directory: Path, title: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[group_column] if group_column else "data"].append(row)

    paths: list[Path] = []
    for y_column in y_columns:
        figure, axis = plt.subplots(figsize=(8, 5))
        for label, values in sorted(groups.items()):
            values.sort(key=lambda row: float(row[x_column]))
            x_values = [float(row[x_column]) for row in values]
            y_values = [float(row[y_column]) for row in values]
            axis.plot(x_values, y_values, marker="o", linewidth=2, label=label)
        axis.set_title(f"{title}: {y_column}" if title else y_column)
        axis.set_xlabel(x_column)
        axis.set_ylabel(y_column)
        axis.grid(visible=True, alpha=0.3)
        if group_column:
            axis.legend(title=group_column)
        figure.tight_layout()
        stem = output_directory / f"{y_column}_by_{x_column}"
        for suffix in (".png", ".svg"):
            path = stem.with_suffix(suffix)
            figure.savefig(path, dpi=200)
            paths.append(path)
        plt.close(figure)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot numeric CSV columns. Use --list-columns to inspect a file first."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the CSV file")
    parser.add_argument("--x", dest="x_column", help="Numeric horizontal-axis column")
    parser.add_argument("--y", help="Comma-separated numeric columns to plot")
    parser.add_argument("--group-by", help="Optional categorical column for separate lines")
    parser.add_argument("--output-directory", type=Path, help="Directory for PNG and SVG figures")
    parser.add_argument("--title", default="", help="Optional chart title prefix")
    parser.add_argument("--list-columns", action="store_true", help="Print available columns and exit")
    arguments = parser.parse_args()

    rows, headers = _read_rows(arguments.csv_path)
    numeric, categorical = _columns(rows, headers)
    if arguments.list_columns:
        print(f"Numeric columns: {', '.join(numeric)}")
        print(f"Categorical columns: {', '.join(categorical) or '(none)'}")
        return 0

    x_column = arguments.x_column or numeric[0]
    if x_column not in numeric:
        raise ValueError(f"--x must name a numeric column. Available: {', '.join(numeric)}")
    if arguments.group_by and arguments.group_by not in headers:
        raise ValueError(f"Unknown --group-by column: {arguments.group_by}")
    y_columns = tuple(column.strip() for column in arguments.y.split(",")) if arguments.y else tuple(
        column for column in numeric if column != x_column and not column.endswith("_std")
    )
    if not y_columns or any(column not in numeric for column in y_columns):
        raise ValueError(f"--y must name one or more numeric columns. Available: {', '.join(numeric)}")

    output_directory = arguments.output_directory or arguments.csv_path.parent / "plots"
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = _plot(
        rows, x_column=x_column, y_columns=y_columns, group_column=arguments.group_by,
        output_directory=output_directory, title=arguments.title,
    )
    print("Generated figures:")
    for path in paths:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
