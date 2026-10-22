from __future__ import annotations

import argparse
import sys
from pathlib import Path

from urban_embedding.eump_launch import resolve_launch_jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve an EUMP launch config into runnable jobs.")
    parser.add_argument("--phase", choices=("train", "eval", "verify"), required=True)
    parser.add_argument("--launch", required=True, help="Launch config name, e.g. eump_full/mae_v3")
    parser.add_argument("--mode", default=None, help="Launcher mode. Defaults to launch.default_mode.")
    parser.add_argument("--config-dir", default=None, help="Hydra config root. Defaults to <repo>/configs.")
    args = parser.parse_args(argv)

    try:
        jobs = resolve_launch_jobs(
            args.launch,
            mode=args.mode,
            phase=args.phase,
            config_dir=None if args.config_dir is None else Path(args.config_dir),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for job in jobs:
        fields = [job.kind, job.name, job.pipeline, *job.extra_overrides]
        print("\t".join(fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
