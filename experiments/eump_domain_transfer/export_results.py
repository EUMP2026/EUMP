#!/usr/bin/env python3
"""Export both domain-transfer protocols and their supplementary comparisons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.eump_domain_transfer.visualize_transferability import (
    MODEL_ORDER,
    build_rank_tables,
    build_transfer_table,
    read_summary,
    save_table,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = EXPERIMENT_DIR / "outputs"
GLOBAL_RUN = OUTPUT_ROOT / "eump_domain_transfer_mae_v3_city_classes_multi_transfer"
SOURCE_RUN = OUTPUT_ROOT / "eump_domain_transfer_mae_v3_source_pretrain_climate_tropic_gdp_poor"
DELIVERY_ROOT = OUTPUT_ROOT / "delivery"
VARIANT_LABELS = {
    "source_pretrain_swin": "Swin (source MAE pretrain)",
    "global_pretrain_swin": "Swin (global MAE pretrain)",
    "scalar_mlp": "Scalar MLP",
    "aef_mlp": "AEF MLP",
}
SCENARIO_LABELS = {
    "climate_bcd_to_a": "Climate B+C+D → A (Tropical)",
    "gdp_nonpoor_to_poor": "GDP non-poor → Poor",
}
TASK_LABELS = {"lst": "LST", "precip": "Precipitation", "pm25": "PM2.5", "traffic": "Traffic"}


def _path(value: Path) -> str:
    value = value.resolve()
    try:
        return str(value.relative_to(REPO_ROOT))
    except ValueError:
        return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _index(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        if key in result:
            raise ValueError(f"Duplicate metric row: {key}")
        result[key] = row
    return result


def _export_protocol(run_dir: Path, output_dir: Path, *, source_pretrain: bool):
    import yaml

    runtime = yaml.safe_load((run_dir / "runtime_config.yaml").read_text(encoding="utf-8"))
    config = runtime["config"]
    if bool(config["pretrain"]["enabled"]) != source_pretrain:
        raise ValueError(f"Run does not match the requested source-pretraining protocol: {run_dir}")
    raw, comparable = read_summary(run_dir / "summary.csv", selected_models=MODEL_ORDER)
    rows = raw.to_dict("records")
    _index(rows, ("model", "scenario", "eval_domain", "task", "scope"))
    checkpoints = sorted({str(row["model_path"]) for row in rows})
    for checkpoint in checkpoints:
        path = Path(checkpoint)
        if not (path if path.is_absolute() else REPO_ROOT / path).is_file():
            raise FileNotFoundError(checkpoint)
    transfer = build_transfer_table(comparable)
    rank_rows, rank_summary = build_rank_tables(transfer)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "summary.csv", output_dir / "summary.csv")
    tables = {"comparable_metrics.csv": comparable, "transferability_table.csv": transfer}
    if not rank_rows.empty:
        tables["transferability_rank_rows.csv"] = rank_rows
        tables["transferability_rank_summary.csv"] = rank_summary
    for filename, table in tables.items():
        save_table(table, output_dir / filename)
    manifest = {
        "source_run": _path(run_dir),
        "source_summary_sha256": _sha256(run_dir / "summary.csv"),
        "source_pretrain_enabled": source_pretrain,
        "swin_encoder_initialization": "source-domain MAE final checkpoint" if source_pretrain else "global MAE v3 best checkpoint",
        "downstream_checkpoint_selection": "final; no validation or target-domain selection",
        "models": sorted(raw["model"].unique().tolist()),
        "scenarios": sorted(raw["scenario"].unique().tolist()),
        "summary_rows": len(raw),
        "table_rows": {filename: len(table) for filename, table in tables.items()},
        "checkpoint_files_verified": len(checkpoints),
        "metric_scopes": "Swin LST city_mean; MLP LST global; other tasks global. Full LST global_pixels remains in summary.csv.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "# Domain-transfer protocol results\n\n"
        f"Source run: `{_path(run_dir)}`.\n\n"
        f"Source-domain MAE pretraining enabled: `{str(source_pretrain).lower()}`. "
        "The disabled setting still loads the global MAE v3 encoder; it does not train Swin from random initialization.\n\n"
        "`summary.csv` preserves all original metric scopes. `comparable_metrics.csv` applies the shared city-level LST comparison rule. "
        "`transferability_table.csv` contains target metrics and source-to-target changes. Source-domain evaluation uses the training domain. "
        "Sample coverage is retained per model and may differ between model families.\n",
        encoding="utf-8",
    )
    return raw, comparable, manifest


def _comparison_rows(global_comparable, source_comparable) -> list[dict[str, Any]]:
    global_targets = global_comparable[~global_comparable["is_source_eval"]].to_dict("records")
    source_targets = source_comparable[~source_comparable["is_source_eval"]].to_dict("records")
    global_index = _index(global_targets, ("model", "scenario", "task", "eval_domain"))
    rows = []
    for source in source_targets:
        scenario, task, domain = (str(source[key]) for key in ("scenario", "task", "eval_domain"))
        variants = {"source_pretrain_swin": source}
        for model, variant in (("swin", "global_pretrain_swin"), ("scalar_mlp", "scalar_mlp"), ("aef_mlp", "aef_mlp")):
            variants[variant] = global_index[model, scenario, task, domain]
        for variant, row in variants.items():
            reference_variant = "global_pretrain_swin" if variant == "source_pretrain_swin" else "source_pretrain_swin"
            reference = variants[reference_variant]
            comparison = "source_vs_global_pretrain" if variant.endswith("pretrain_swin") else "model_baselines_vs_source_pretrain_swin"
            rmse, reference_rmse = float(row["rmse"]), float(reference["rmse"])
            r2, reference_r2 = float(row["r2"]), float(reference["r2"])
            rows.append({
                "comparison": comparison,
                "comparison_label": "Source-domain MAE pretrain vs global MAE pretrain" if comparison == "source_vs_global_pretrain" else "Downstream model comparison (reference: source-pretrained Swin)",
                "model": variant,
                "model_label": VARIANT_LABELS[variant],
                "reference_model": reference_variant,
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                "train_domain": "Rich+Medium" if scenario == "gdp_nonpoor_to_poor" else str(source["train_domain"]),
                "eval_domain": domain,
                "task": task,
                "task_label": TASK_LABELS[task],
                "scope": str(row["scope"]),
                "comparison_scope": "city_mean" if task == "lst" else "global",
                "rmse": rmse,
                "mae": float("nan"),
                "r2": r2,
                "reference_rmse": reference_rmse,
                "reference_r2": reference_r2,
                "rmse_delta_vs_reference": rmse - reference_rmse,
                "rmse_ratio_vs_reference": rmse / reference_rmse,
                "r2_delta_vs_reference": r2 - reference_r2,
                "count": int(row["count"]),
                "city_count": int(row["city_count"]),
                "row_count": int(row["row_count"]),
                "year_range": str(row["year_range"]),
                "status": "ok",
            })
    return rows


def _pretrain_protocol_rows(global_raw, source_raw) -> list[dict[str, Any]]:
    fields = ("model", "scenario", "task", "scope", "eval_domain")
    global_rows = _index(global_raw.to_dict("records"), fields)
    source_rows = _index(source_raw.to_dict("records"), fields)
    output = []
    for row in source_raw[~source_raw["is_source_eval"]].to_dict("records"):
        key = tuple(str(row[field]) for field in fields)
        reference = global_rows[key]
        source_domain = "source:" + str(row["train_domain"])
        source = source_rows[str(row["model"]), str(row["scenario"]), str(row["task"]), str(row["scope"]), source_domain]
        for field in ("count", "city_count", "row_count"):
            if int(row[field]) != int(reference[field]):
                raise ValueError(f"Source/global pretrain evaluation coverage differs: {key}, {field}")
        rmse, r2 = float(row["rmse"]), float(row["r2"])
        reference_rmse, reference_r2 = float(reference["rmse"]), float(reference["r2"])
        output.append({
            "scenario": row["scenario"], "task": row["task"], "scope": row["scope"],
            "source_domain": source_domain, "target_domain": row["eval_domain"],
            "source_pretrain_target_rmse": rmse, "source_pretrain_target_r2": r2,
            "global_pretrain_target_rmse": reference_rmse, "global_pretrain_target_r2": reference_r2,
            "rmse_delta_source_minus_global": rmse - reference_rmse,
            "rmse_ratio_source_over_global": rmse / reference_rmse,
            "r2_delta_source_minus_global": r2 - reference_r2,
            "target_count": int(row["count"]), "target_city_count": int(row["city_count"]), "target_row_count": int(row["row_count"]),
            "source_count": int(source["count"]), "source_city_count": int(source["city_count"]), "source_row_count": int(source["row_count"]),
            "year_range": row["year_range"],
        })
    return output


def _reconstruction_rows(source_run_dir: Path) -> list[dict[str, Any]]:
    selected_metrics = {"loss_total", "loss_group_reconstruct", "loss_group_aux"}
    rows = [row for row in _read_csv(source_run_dir / "pretrain_summary.csv") if row["metric"] in selected_metrics and row["status"] == "ok"]
    source = {(row["scenario"], row["metric"]): float(row["value"]) for row in rows if row["eval_domain"].startswith("source:")}
    return [{
        "scenario": row["scenario"],
        "domain_role": "source" if row["eval_domain"].startswith("source:") else "target",
        "eval_domain": row["eval_domain"], "metric": row["metric"], "value": float(row["value"]),
        "target_source_ratio": "" if row["eval_domain"].startswith("source:") else float(row["value"]) / source[row["scenario"], row["metric"]],
        "row_count": int(row["row_count"]), "city_count": int(row["city_count"]), "year_range": row["year_range"],
    } for row in rows]


def export_results(*, global_run_dir: Path = GLOBAL_RUN, source_run_dir: Path = SOURCE_RUN, output_dir: Path = DELIVERY_ROOT) -> dict[str, Any]:
    global_run_dir, source_run_dir, output_dir = (path.resolve() for path in (global_run_dir, source_run_dir, output_dir))
    global_raw, global_comparable, global_manifest = _export_protocol(global_run_dir, output_dir / "without_source_pretrain", source_pretrain=False)
    source_raw, source_comparable, source_manifest = _export_protocol(source_run_dir, output_dir / "with_source_pretrain", source_pretrain=True)
    comparison_dir = output_dir / "comparisons"
    comparisons = _comparison_rows(global_comparable, source_comparable)
    protocol_rows = _pretrain_protocol_rows(global_raw, source_raw)
    reconstruction = _reconstruction_rows(source_run_dir)
    _write_csv(comparison_dir / "comparable_metrics.csv", comparisons)
    _write_csv(comparison_dir / "pretrain_protocol_comparison.csv", protocol_rows)
    _write_csv(comparison_dir / "pretrain_reconstruction_transfer.csv", reconstruction)
    manifest = {
        "experiment": "leave-domain-out supplementary comparisons",
        "source_run": _path(source_run_dir), "global_run": _path(global_run_dir),
        "comparisons": [{"id": name, "rows": sum(row["comparison"] == name for row in comparisons)} for name in ("source_vs_global_pretrain", "model_baselines_vs_source_pretrain_swin")],
        "pretrain_protocol_rows": len(protocol_rows), "reconstruction_rows": len(reconstruction),
        "notes": [
            "pretrain.enabled=false skips scenario MAE training and still loads the global MAE v3 encoder.",
            "Source-domain MAE uses the standard pretrained swin_L initialization; it is not random-initialized backbone training.",
            "Physical downstream metrics and processed-space reconstruction losses are exported separately.",
            "Swin LST city_mean is compared with city-level MLP predictions recorded as global. Full global_pixels remains in protocol summary.csv.",
            "MLP cohorts can differ from Swin cohorts; each row retains sample coverage. Source/global Swin target counts, city counts, and row counts are checked for equality.",
        ],
    }
    _write_json(comparison_dir / "manifest.json", manifest)
    (comparison_dir / "README.md").write_text(
        "# Leave-domain-out supplementary comparisons\n\n"
        "`comparable_metrics.csv` contains target metrics for source-pretrained Swin, global-pretrained Swin, Scalar MLP, and AEF MLP. "
        "`pretrain_protocol_comparison.csv` additionally preserves LST global-pixel results. "
        "`pretrain_reconstruction_transfer.csv` contains processed-space reconstruction losses, separate from physical downstream metrics.\n\n"
        "Rebuild with `python experiments/eump_domain_transfer/export_results.py`.\n",
        encoding="utf-8",
    )
    report = {"passed": True, "without_source_pretrain": global_manifest, "with_source_pretrain": source_manifest, "comparisons": manifest}
    _write_json(output_dir / "export_manifest.json", report)
    (output_dir / "README.md").write_text(
        "# Domain-transfer results\n\n"
        "- `without_source_pretrain/`: six scenarios with the global MAE v3 encoder and two MLP baselines.\n"
        "- `with_source_pretrain/`: two scenarios with source-domain MAE pretraining and frozen-encoder downstream training.\n"
        "- `comparisons/`: source/global Swin pretraining and MLP comparisons on the two supplementary scenarios.\n\n"
        "Rebuild with `python experiments/eump_domain_transfer/export_results.py`. Generated files are ignored by Git.\n",
        encoding="utf-8",
    )
    return {"manifest": report, "comparison_rows": comparisons, "comparison_directory": comparison_dir}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-run-dir", type=Path, default=GLOBAL_RUN)
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DELIVERY_ROOT)
    args = parser.parse_args(argv)
    result = export_results(global_run_dir=args.global_run_dir, source_run_dir=args.source_run_dir, output_dir=args.output_dir)
    for name in ("without_source_pretrain", "with_source_pretrain"):
        print(f"{name}: {result['manifest'][name]['table_rows']['comparable_metrics.csv']} comparable rows")
    print(f"Supplementary comparisons: {len(result['comparison_rows'])} rows")
    print(f"Saved: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
