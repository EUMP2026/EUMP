#!/usr/bin/env python3
"""Render the planning Fig4 visualization from a planning_data.pkl package."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_figs_v2.fig4.fig4 import main as render_fig4_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--input-dir" not in args:
        args = ["--input-dir", str(DEFAULT_OUTPUT_DIR), *args]
    return render_fig4_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
