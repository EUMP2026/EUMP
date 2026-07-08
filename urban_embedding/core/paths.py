from __future__ import annotations

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_data")
CSV_DATA_DIR = os.path.join(BASE_DIR, "data", "csv_data")
EUMP_DATA_DIR = os.path.join(BASE_DIR, "data", "eump_v1")
EUMP_MANIFEST_DIR = os.path.join(EUMP_DATA_DIR, "manifest")
EUMP_SCENES_DIR = os.path.join(EUMP_DATA_DIR, "scenes")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "log")


def resolve_model_artifact_path(model_name: str, file_name: str | None) -> str | None:
    if file_name is None:
        return None
    if os.path.isabs(file_name):
        return file_name
    return os.path.join(MODEL_DIR, model_name, file_name)


__all__ = [
    "BASE_DIR",
    "CSV_DATA_DIR",
    "LOG_DIR",
    "MODEL_DIR",
    "EUMP_DATA_DIR",
    "EUMP_MANIFEST_DIR",
    "EUMP_SCENES_DIR",
    "RAW_DATA_DIR",
    "resolve_model_artifact_path",
]
