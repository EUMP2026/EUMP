#!/usr/bin/env python3
"""
Summarize planning_exp LST target sweeps into city-by-target tables.

Expected input layout:
    experiments/planning_exp/outputs/lst_delta_base_joint_sweep_5cities/
      lst_delta_1p01/summary.json
      lst_delta_1p00/summary.json
      lst_delta_0p99/summary.json

Outputs:
    lst_delta_base_city_table.csv
    lst_delta_base_city_table.md
    lst_delta_base_long.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "experiments/planning_exp/outputs/lst_delta_base_joint_sweep_5cities"
)
DEFAULT_TARGETS = ("1.01", "1.00", "0.99")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build city x LST target summary tables for planning_exp sweeps."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing lst_delta_* result folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary tables. Default: input root.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGETS),
        help="Comma-separated target order. Default: 1.01,1.00,0.99.",
    )
    parser.add_argument(
        "--cities",
        default=None,
        help="Optional comma-separated city order. Default: order discovered from summaries.",
    )
    return parser.parse_args(argv)


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _target_label(target: str) -> str:
    return str(target).replace(".", "p")


def _target_float(target: str) -> float:
    return float(str(target).strip())


def _find_summary_path(input_root: Path, target: str) -> Path:
    target_value = _target_float(target)
    labels = [
        _target_label(target),
        f"{target_value:.2f}".replace(".", "p"),
        f"{target_value:.1f}".replace(".", "p"),
        str(int(target_value)) if target_value.is_integer() else "",
    ]
    candidates = []
    seen_labels: set[str] = set()
    for label in labels:
        if label and label not in seen_labels:
            seen_labels.add(label)
            candidates.append(input_root / f"lst_delta_{label}" / "summary.json")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing summary.json for target {target!r}; checked: "
        + ", ".join(str(p) for p in candidates)
    )


def _read_summary(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data.get("results"), list):
        raise ValueError(f"{path} does not contain a list field named 'results'")
    return data


def _discover_city_order(rows: list[dict[str, Any]]) -> list[int]:
    order: list[int] = []
    seen: set[int] = set()
    for row in rows:
        city_id = int(row["city_id"])
        if city_id not in seen:
            seen.add(city_id)
            order.append(city_id)
    return order


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _result_signed_lst_delta(result: dict[str, Any]) -> Any:
    return result.get("final_signed_delta_lst", result.get("final_delta_lst"))


def _result_abs_lst_delta(result: dict[str, Any]) -> Any:
    if result.get("final_abs_delta_lst") is not None:
        return result.get("final_abs_delta_lst")
    signed = _result_signed_lst_delta(result)
    return None if signed is None else abs(float(signed))


def _existing_path_text(path: Path) -> str:
    return str(path) if path.exists() else ""


def _summary_metric_result(result: dict[str, Any], mode: str | None) -> dict[str, Any]:
    if isinstance(result.get("joint"), dict):
        return result["joint"]
    if mode == "cooling_only" and isinstance(result.get("cooling_only"), dict):
        return result["cooling_only"]
    if isinstance(result.get("cooling_only"), dict):
        return result["cooling_only"]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    def cell(value: Any) -> str:
        return "" if value is None else str(value)

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(name)) for name in fieldnames) + " |")
    path.write_text("\n".join(lines) + "\n")


def build_tables(
    *,
    input_root: Path,
    output_dir: Path,
    targets: list[str],
    cities: list[int] | None,
) -> tuple[Path, Path, Path]:
    long_rows: list[dict[str, Any]] = []

    for target in targets:
        summary_path = _find_summary_path(input_root, target)
        summary = _read_summary(summary_path)
        cfg = summary.get("config", {})
        target_value = float(
            cfg.get("targets", {}).get("lst_delta_base", _target_float(target))
        )
        mode = summary.get("mode", cfg.get("mode"))
        year = summary.get("year", cfg.get("year"))

        for result in summary["results"]:
            city_id = int(result["city_id"])
            metric_result = _summary_metric_result(result, mode)
            long_rows.append({
                "city_id": city_id,
                "lst_delta_base": f"{target_value:.2f}",
                "mode": mode,
                "year": year,
                "final_delta_lst": _format_float(_result_signed_lst_delta(metric_result)),
                "final_abs_delta_lst": _format_float(_result_abs_lst_delta(metric_result)),
                "final_delta_traffic": _format_float(metric_result.get("final_delta_traffic")),
                "original_traffic": _format_float(metric_result.get("original_traffic")),
                "target_traffic": _format_float(metric_result.get("target_traffic")),
                "adjusted_traffic": _format_float(metric_result.get("adjusted_traffic")),
                "target_traffic_drop_pct": _format_float(
                    metric_result.get("target_traffic_drop_pct")
                ),
                "traffic_drop_pct": _format_float(metric_result.get("traffic_drop_pct")),
                "fig4_png": _existing_path_text(
                    summary_path.parent / "figures" / "fig4.png"
                ),
                "result_dir": str(summary_path.parent),
            })

    city_order = cities if cities is not None else _discover_city_order(long_rows)
    target_order = [f"{_target_float(t):.2f}" for t in targets]
    value_by_city_target = {
        (int(row["city_id"]), str(row["lst_delta_base"])): row["final_delta_lst"]
        for row in long_rows
    }

    wide_fieldnames = ["city_id"] + [f"target_{t}" for t in target_order]
    wide_rows: list[dict[str, Any]] = []
    for city_id in city_order:
        row: dict[str, Any] = {"city_id": city_id}
        for target in target_order:
            row[f"target_{target}"] = value_by_city_target.get((city_id, target), "")
        wide_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = output_dir / "lst_delta_base_city_table.csv"
    wide_md = output_dir / "lst_delta_base_city_table.md"
    long_csv = output_dir / "lst_delta_base_long.csv"

    _write_csv(wide_csv, wide_rows, wide_fieldnames)
    _write_markdown(wide_md, wide_rows, wide_fieldnames)
    _write_csv(
        long_csv,
        long_rows,
        [
            "city_id",
            "lst_delta_base",
            "mode",
            "year",
            "final_delta_lst",
            "final_abs_delta_lst",
            "final_delta_traffic",
            "original_traffic",
            "target_traffic",
            "adjusted_traffic",
            "target_traffic_drop_pct",
            "traffic_drop_pct",
            "fig4_png",
            "result_dir",
        ],
    )

    return wide_csv, wide_md, long_csv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = _resolve_path(args.input_root)
    output_dir = _resolve_path(args.output_dir) if args.output_dir else input_root
    targets = [part.strip() for part in str(args.targets).split(",") if part.strip()]
    cities = (
        [int(part.strip()) for part in str(args.cities).split(",") if part.strip()]
        if args.cities
        else None
    )

    wide_csv, wide_md, long_csv = build_tables(
        input_root=input_root,
        output_dir=output_dir,
        targets=targets,
        cities=cities,
    )
    print(f"Saved city x target table: {wide_csv}")
    print(f"Saved markdown table: {wide_md}")
    print(f"Saved long-format table: {long_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
