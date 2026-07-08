# Script Layout

The `script/` directory is grouped by the kind of work each entry point supports:

- `data/`: raw-data sampling, EUMP scene-store export, preprocessing, split manifests, validation, and the full data-processing launcher.
- `training/`: reconstruction/downstream training and evaluation launchers.
- `analysis/`: raw-data inspections and distribution plotting utilities.
- `experiments/`: CV launchers (`run_current_traffic_cv.sh`, `run_current_pm25_cv.sh`,
  `run_city_scalar_cv.py`), embedding export (`export_eump_embeddings.py`), and the
  reference input-perturbation script (`run_eump_input_counterfactual.py`).
- `share/`: bundle packaging helpers for Docker-based source sharing, including
  runtime/model bundles and filtered processed-EUMP data subsets.
- `local/`: ignored local debugging helpers.

Examples usually assume the repository root. Path-aware shell wrappers cd back
to the repository root before invoking `main/` entry points.
