from __future__ import annotations

import math
from dataclasses import dataclass
from numcodecs import Blosc
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import zarr
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .schema import DEFAULT_RASTER_CHUNK, EUMP_STORE_VERSION


def import_numcodecs():
    return Blosc


def import_rasterio():
    return rasterio, Resampling, reproject


def import_zarr():
    return zarr


@dataclass(frozen=True)
class ReferenceGrid:
    scene_id: int
    height: int
    width: int
    crs: Any
    crs_wkt: str | None
    transform: Any
    transform_gdal: list[float]
    path: Path

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass
class SourcePayload:
    source_name: str
    data: np.ndarray
    valid_mask: np.ndarray
    attrs: dict[str, Any]


def scalar_column(base_name: str, year: int, start_year: int) -> str:
    offset = year - start_year
    return base_name if offset == 0 else f"{base_name}_{offset}"


def bh8_band_name(base_name: str, year: int, start_year: int = 2003) -> str:
    offset = year - start_year
    return base_name if offset == 0 else f"{base_name}_{offset}"


def single_channel_names(source_name: str) -> list[str]:
    return [source_name]


def normalize_nodata_rule(value: Any) -> Any:
    if value in ("/", "", None):
        return None
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return "nan"
    if isinstance(value, str) and value.lower() == "nan":
        return "nan"
    return value


def _nodata_match_mask(data: np.ndarray, nodata_rule: Any) -> np.ndarray:
    if isinstance(nodata_rule, (int, float, np.integer, np.floating)):
        nodata_value = float(nodata_rule)
        if math.isfinite(nodata_value) and abs(nodata_value) >= 1e30:
            # Some GeoTIFFs document float32 nodata as -3.4e38 while pixels store
            # the nearby GDAL sentinel around -3.402823e38; exact comparison misses it.
            return data <= nodata_value if nodata_value < 0 else data >= nodata_value
    return data == nodata_rule


def valid_mask_from_rule(data: np.ndarray, nodata_rule: Any) -> np.ndarray:
    valid_mask = np.isfinite(data)
    nodata_rule = normalize_nodata_rule(nodata_rule)
    if nodata_rule is None:
        return valid_mask
    if nodata_rule == "nan":
        return np.isfinite(data)
    return valid_mask & ~_nodata_match_mask(data, nodata_rule)


def compressor():
    blosc = import_numcodecs()
    return blosc(cname="zstd", clevel=5, shuffle=blosc.SHUFFLE)


def raster_chunks(height: int, width: int) -> tuple[int, int, int]:
    return DEFAULT_RASTER_CHUNK[0], min(DEFAULT_RASTER_CHUNK[1], height), min(DEFAULT_RASTER_CHUNK[2], width)


def scalar_chunks(length: int) -> tuple[int]:
    return (max(1, int(length)),)


def as_raster_payload(
    source_name: str,
    data_2d: np.ndarray,
    valid_mask_2d: np.ndarray,
    attrs: dict[str, Any],
) -> SourcePayload:
    data = np.asarray(data_2d)[None, ...]
    valid_mask = np.asarray(valid_mask_2d, dtype=bool)[None, ...]
    return SourcePayload(source_name=source_name, data=data, valid_mask=valid_mask, attrs=attrs)


def as_scalar_payload(
    source_name: str,
    value: float,
    valid: bool,
    attrs: dict[str, Any],
) -> SourcePayload:
    return SourcePayload(
        source_name=source_name,
        data=np.asarray([value], dtype=np.float32),
        valid_mask=np.asarray([valid], dtype=bool),
        attrs=attrs,
    )


def build_root_attrs(
    reference: ReferenceGrid,
    scene_id: int,
    available_sources: list[str],
    sample_id: int | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    payload = {
        "scene_id": int(scene_id),
        "sample_id": None if sample_id is None else int(sample_id),
        "spatial_shape": list(reference.spatial_shape),
        "crs_wkt": reference.crs_wkt,
        "transform_gdal": list(reference.transform_gdal),
        "reference_grid": "BH8_250m",
        "export_version": EUMP_STORE_VERSION,
        "available_sources": list(sorted(available_sources)),
    }
    if year is not None:
        payload["year"] = int(year)
    return payload


def build_source_attrs(
    spec: dict[str, Any],
    source_name: str,
    data: np.ndarray,
    *,
    source_years: list[int] | None,
    source_path: str | None,
    channel_names: list[str] | None = None,
    derived_from: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "display_name": spec["display_name"],
        "storage_level": spec["storage_level"],
        "modality": spec["modality"],
        "channel_names": list(channel_names or single_channel_names(source_name)),
        "dtype": str(data.dtype),
        "nodata_rule": spec["nodata"],
        "raw_source_dir": spec["raw_source_dir"],
        "raw_resolution": spec["raw_resolution"],
        "reference_grid": spec["reference_grid"],
        "source_years": [] if source_years is None else [int(year) for year in source_years],
        "source_path": source_path,
        "is_derived": bool(spec["is_derived"]),
        "raw_source_name": spec.get("raw_source_name"),
        "derived_from": list(derived_from or spec.get("derived_from", [])),
        "resampling": None if spec["modality"] == "scalar" else "nearest",
        "resampled_to": None if spec["modality"] == "scalar" else spec["reference_grid"],
    }


__all__ = [
    "ReferenceGrid",
    "SourcePayload",
    "as_raster_payload",
    "as_scalar_payload",
    "bh8_band_name",
    "build_root_attrs",
    "build_source_attrs",
    "compressor",
    "import_rasterio",
    "import_zarr",
    "normalize_nodata_rule",
    "raster_chunks",
    "scalar_chunks",
    "scalar_column",
    "single_channel_names",
    "valid_mask_from_rule",
]
