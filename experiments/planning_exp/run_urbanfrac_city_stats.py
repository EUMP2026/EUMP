#!/usr/bin/env python3
"""
Run planning optimization statistics for large urban-fraction cities.

The script reads a city CSV, splits cities evenly across GPUs, launches one
run.py subprocess per GPU, and merges each worker's planning_data.pkl into one
standard planning data package. Worker runs disable per-city NPZ export.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CITY_CSV = PROJECT_ROOT / "experiments/planning_exp/urbanfrac_cities_ge0.1.csv"
# The retained planning line uses the joint-gradient v2 policy: the summed
# LST and traffic objective updates both intervention channels.
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs/joint_gradient_v2_292cities"
DEFAULT_CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
WORKER_DATA_FILE = "worker_planning_data.pkl"


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run planning statistics for urbanfrac city CSV with multi-GPU workers."
    )
    parser.add_argument("--city-csv", type=Path, default=DEFAULT_CITY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-file", default="planning_data.pkl")
    parser.add_argument(
        "--mode",
        choices=("joint", "cooling_only", "both"),
        default="both",
        help="Optimization mode passed to run.py. Default: both.",
    )
    parser.add_argument(
        "--gradient-mode",
        choices=("source_separated", "joint"),
        default="joint",
        help="Gradient policy passed to run.py. Default: joint (v2).",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device seen inside each worker after CUDA_VISIBLE_DEVICES is set.",
    )
    parser.add_argument(
        "--data-override",
        action="append",
        default=None,
        help=(
            "Hydra override passed to run.py. Can be repeated. "
            "Default: data=eump_full/server_v2"
        ),
    )
    parser.add_argument(
        "--python-bin",
        default=os.environ.get("PYTHON_BIN", sys.executable),
        help="Python executable used for worker subprocesses.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-worker-dirs", action="store_true")
    args, extra_overrides = parser.parse_known_args(argv)
    return args, extra_overrides


def _resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _parse_gpus(value: str) -> list[str]:
    gpus = [part.strip() for part in str(value).split(",") if part.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    return gpus


def _to_float_or_text(value: str | None) -> float | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return ""
    try:
        return float(text)
    except ValueError:
        return text


def _read_city_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain an 'id' column")
        rows: list[dict[str, Any]] = []
        for row in reader:
            city_id = int(str(row["id"]).strip())
            rows.append({
                "id": city_id,
                "name": row.get("name", ""),
                "area": _to_float_or_text(row.get("area")),
                "urban_fraction": _to_float_or_text(row.get("urban_fraction")),
            })
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _split_even(items: list[dict[str, Any]], n_parts: int) -> list[list[dict[str, Any]]]:
    q, r = divmod(len(items), n_parts)
    chunks: list[list[dict[str, Any]]] = []
    start = 0
    for idx in range(n_parts):
        size = q + (1 if idx < r else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _worker_command(
    *,
    args: argparse.Namespace,
    extra_overrides: list[str],
    worker_dir: Path,
    city_ids: list[int],
) -> list[str]:
    run_py = EXPERIMENT_DIR / "run.py"
    cmd = [
        str(args.python_bin),
        str(run_py),
        "--config-path",
        str(_resolve_path(args.config_path)),
        "--mode",
        str(args.mode),
        "--gradient-mode",
        str(args.gradient_mode),
        "--cities",
        ",".join(str(city_id) for city_id in city_ids),
        "--year",
        str(int(args.year)),
        "--device",
        str(args.device),
        "--num-workers",
        str(int(args.num_workers)),
        "--no-city-npz",
        "--data-file",
        WORKER_DATA_FILE,
        "--output-dir",
        str(worker_dir),
    ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(int(args.epochs))])
    if args.lr is not None:
        cmd.extend(["--lr", str(float(args.lr))])

    overrides = args.data_override if args.data_override else ["data=eump_full/server_v2"]
    cmd.extend(overrides)
    cmd.extend(extra_overrides)
    return cmd


def _load_worker_package(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Worker package not found: {path}")
    with path.open("rb") as f:
        package = pickle.load(f)
    if not isinstance(package, dict) or package.get("schema_version") != 1:
        raise ValueError(f"Invalid worker planning package: {path}")
    if not isinstance(package.get("results"), list) or not isinstance(package.get("city_data"), list):
        raise ValueError(f"Worker package missing results/city_data lists: {path}")
    return package


def _merge_packages(
    *,
    packages: list[dict[str, Any]],
    city_rows: list[dict[str, Any]],
    workers: list[dict[str, Any]],
    args: argparse.Namespace,
    city_csv: Path,
) -> dict[str, Any]:
    order = {int(row["id"]): idx for idx, row in enumerate(city_rows)}
    results_by_city: dict[int, dict[str, Any]] = {}
    data_by_city: dict[int, dict[str, Any]] = {}
    config = deepcopy(packages[0].get("config") if packages else {})

    for package in packages:
        if package.get("mode") != args.mode:
            raise ValueError(
                f"Worker mode mismatch: expected {args.mode!r}, got {package.get('mode')!r}"
            )
        for result in package["results"]:
            city_id = int(result["city_id"])
            if city_id in results_by_city:
                raise ValueError(f"Duplicate result for city {city_id}")
            results_by_city[city_id] = result
        for item in package["city_data"]:
            city_id = int(item["city_id"])
            if city_id in data_by_city:
                raise ValueError(f"Duplicate city_data for city {city_id}")
            data_by_city[city_id] = item

    expected_ids = [int(row["id"]) for row in city_rows]
    missing = [city_id for city_id in expected_ids if city_id not in data_by_city]
    if missing:
        raise ValueError(f"Missing worker results for city IDs: {missing[:20]}")

    sorted_ids = sorted(expected_ids, key=lambda city_id: order[city_id])
    if isinstance(config, dict):
        config["mode"] = args.mode
        config["year"] = int(args.year)
        config["cities"] = sorted_ids
        config["output_dir"] = str(_resolve_path(args.output_dir))
        config.setdefault("data_export", {})["enabled"] = True
        config.setdefault("data_export", {})["data_file"] = str(args.output_file)
        config.setdefault("outputs", {})["save_city_npz"] = False

    return {
        "schema_version": 1,
        "mode": args.mode,
        "year": int(args.year),
        "cities": sorted_ids,
        "config": config,
        "results": [results_by_city[city_id] for city_id in sorted_ids],
        "city_data": [data_by_city[city_id] for city_id in sorted_ids],
        "source_city_csv": str(city_csv),
        "city_metadata": city_rows,
        "workers": workers,
        "run_args": {
            "mode": args.mode,
            "gradient_mode": args.gradient_mode,
            "year": int(args.year),
            "gpus": _parse_gpus(args.gpus),
            "config_path": str(_resolve_path(args.config_path)),
            "data_override": args.data_override or ["data=eump_full/server_v2"],
            "epochs": args.epochs,
            "lr": args.lr,
            "num_workers": int(args.num_workers),
            "device": args.device,
            "limit": args.limit,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args, extra_overrides = parse_args(argv)
    city_csv = _resolve_path(args.city_csv)
    output_dir = _resolve_path(args.output_dir)
    gpus = _parse_gpus(args.gpus)
    city_rows = _read_city_rows(city_csv, args.limit)
    if not city_rows:
        raise ValueError(f"No cities found in {city_csv}")

    chunks = _split_even(city_rows, len(gpus))
    assignments = [
        {
            "worker_index": idx,
            "gpu": gpu,
            "cities": [int(row["id"]) for row in chunk],
        }
        for idx, (gpu, chunk) in enumerate(zip(gpus, chunks))
        if chunk
    ]

    print(
        f"Loaded {len(city_rows)} cities from {city_csv}; "
        f"{len(assignments)} workers across GPUs {','.join(gpus)}."
    )
    for assignment in assignments:
        cities = assignment["cities"]
        print(
            f"worker {assignment['worker_index']} gpu={assignment['gpu']} "
            f"n={len(cities)} first={cities[0]} last={cities[-1]}"
        )

    if args.dry_run:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    workers_dir = output_dir / "_workers"
    workers_dir.mkdir(parents=True, exist_ok=True)

    processes: list[dict[str, Any]] = []
    try:
        for assignment in assignments:
            worker_index = int(assignment["worker_index"])
            gpu = str(assignment["gpu"])
            worker_dir = workers_dir / f"worker_{worker_index:02d}_gpu_{gpu}"
            if worker_dir.exists():
                shutil.rmtree(worker_dir)
            worker_dir.mkdir(parents=True, exist_ok=True)
            log_path = worker_dir / "worker.log"
            city_ids = [int(city_id) for city_id in assignment["cities"]]
            cmd = _worker_command(
                args=args,
                extra_overrides=extra_overrides,
                worker_dir=worker_dir,
                city_ids=city_ids,
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            log_f = log_path.open("w")
            print(f"Launching worker {worker_index} on GPU {gpu}: {len(city_ids)} cities")
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            assignment["worker_dir"] = str(worker_dir)
            assignment["log_file"] = str(log_path)
            processes.append({
                "assignment": assignment,
                "process": proc,
                "log_file": log_f,
                "data_path": worker_dir / WORKER_DATA_FILE,
            })

        failed: list[dict[str, Any]] = []
        for item in processes:
            proc = item["process"]
            rc = proc.wait()
            item["log_file"].close()
            if rc != 0:
                failed.append(item)

        if failed:
            for item in failed:
                assignment = item["assignment"]
                log_path = Path(assignment["log_file"])
                print(
                    f"\nWorker {assignment['worker_index']} failed "
                    f"(gpu={assignment['gpu']}, returncode={item['process'].returncode})."
                )
                print(_tail_text(log_path))
            raise RuntimeError(f"{len(failed)} worker(s) failed; keeping worker dirs for debugging.")

        packages = [_load_worker_package(item["data_path"]) for item in processes]
        package = _merge_packages(
            packages=packages,
            city_rows=city_rows,
            workers=assignments,
            args=args,
            city_csv=city_csv,
        )
        final_path = output_dir / str(args.output_file)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with final_path.open("wb") as f:
            pickle.dump(package, f)
        print(f"Saved merged planning data package: {final_path}")

        if not args.keep_worker_dirs:
            shutil.rmtree(workers_dir)
        return 0
    except Exception:
        for item in processes:
            log_f = item.get("log_file")
            if log_f and not log_f.closed:
                log_f.close()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
