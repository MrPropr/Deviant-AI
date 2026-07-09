"""Create public-safe aggregate figures from summary metric tables."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


CONDITION_ORDER = ["direct", "polite", "multi_turn", "polite_multi_turn"]


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Input path is not a file: {path}")
    return path


def output_dir(value: str) -> Path:
    path = Path(value)
    if path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError(f"Output path is not a directory: {path}")
    return path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def ordered(values: set[str], preferred: list[str]) -> list[str]:
    preferred_values = [value for value in preferred if value in values]
    extra_values = sorted(value for value in values if value not in preferred)
    return preferred_values + extra_values


def write_svg_bar_chart(rows: list[dict], metric: str, title: str, path: Path) -> None:
    filtered = [row for row in rows if to_float(row.get(metric, "")) is not None]
    if not filtered:
        return

    conditions = ordered({row["condition"] for row in filtered}, CONDITION_ORDER)
    models = sorted({row["model_id"] for row in filtered})
    values = {
        (row["condition"], row["model_id"]): to_float(row.get(metric, ""))
        for row in filtered
    }

    width = 920
    height = 520
    margin_left = 95
    margin_right = 35
    margin_top = 55
    margin_bottom = 115
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    group_width = chart_width / max(len(conditions), 1)
    bar_width = max(group_width / max(len(models), 1) * 0.68, 8)
    colors = ["#31688e", "#35b779", "#fdae61", "#7b3294", "#80cdc1"]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#333"/>',
    ]

    for tick in range(6):
        value = tick / 5
        y = margin_top + chart_height - value * chart_height
        lines.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>')

    for condition_index, condition in enumerate(conditions):
        group_x = margin_left + condition_index * group_width
        label_x = group_x + group_width / 2
        lines.append(
            f'<text x="{label_x:.1f}" y="{height - 70}" text-anchor="middle" '
            f'font-family="Arial" font-size="12">{html.escape(condition)}</text>'
        )
        for model_index, model_id in enumerate(models):
            value = values.get((condition, model_id))
            if value is None:
                continue
            x = group_x + (model_index + 0.18) * (group_width / max(len(models), 1))
            bar_height = max(min(value, 1.0), 0.0) * chart_height
            y = margin_top + chart_height - bar_height
            color = colors[model_index % len(colors)]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
            )

    legend_x = margin_left
    legend_y = height - 42
    for index, model_id in enumerate(models):
        x = legend_x + index * 180
        color = colors[index % len(colors)]
        lines.append(f'<rect x="{x}" y="{legend_y}" width="12" height="12" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 18}" y="{legend_y + 11}" font-family="Arial" font-size="12">{html.escape(model_id)}</text>'
        )

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=existing_file, required=True, help="Aggregate summary CSV file.")
    parser.add_argument("--output-dir", type=output_dir, default=Path("figures"), help="Figure output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows to plot in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [
        (args.output_dir / "asr_by_condition.svg", "attack_success_rate", "Attack Success Rate By Condition"),
        (
            args.output_dir / "mean_harmfulness_by_condition.svg",
            "mean_harmfulness_score",
            "Mean Harmfulness By Condition",
        ),
    ]

    written = []
    for path, metric, title in outputs:
        write_svg_bar_chart(rows, metric, title, path)
        if path.exists():
            written.append(path)

    print(f"Wrote {len(written)} figures: {', '.join(str(path) for path in written)}")


if __name__ == "__main__":
    main()
