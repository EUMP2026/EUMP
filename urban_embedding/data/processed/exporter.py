from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from urban_embedding.core.paths import CSV_DATA_DIR, EUMP_DATA_DIR, RAW_DATA_DIR

from .alignment import (
    ReferenceGrid,
    SourcePayload,
    as_raster_payload,
    as_scalar_payload,
    bh8_band_name,
    build_root_attrs,
    build_source_attrs,
    compressor,
    import_rasterio,
    import_zarr,
    raster_chunks,
    scalar_chunks,
    scalar_column,
    single_channel_names,
    valid_mask_from_rule,
)
from .manifest import (
    normalization_stats_path,
    read_jsonl,
    scene_years_manifest_path,
    scenes_manifest_path,
    schema_path,
    write_json,
    write_jsonl,
)
from .schema import (
    CITY_MASK_SOURCE,
    CLIMATE_PRECIP_SOURCE,
    PRECIP_TARGET_SOURCE,
    build_eump_store_schema,
)
from .validation import validate_bh8_padding_facts


WORKER_EXPORTER = None
GHG_SCALAR_CSV_NAME = "ghg_area_weighted_sum_mean_2003_2022_wide.csv"
GHG_SCALAR_SOURCES = {
    "GHG_emission": "GHG_sum",
    "GHG_mean": "GHG_mean",
}
PM25_MEAN_SCALAR_CSV_NAME = "pm25_mean.csv"
PM25_MEAN_SOURCE = "PM25_mean"


def init_scene_export_worker(raw_root: str, csv_root: str, output_root: str) -> None:
    global WORKER_EXPORTER
    WORKER_EXPORTER = ProcessedSceneStoreExporter(
        raw_root=raw_root,
        csv_root=csv_root,
        output_root=output_root,
        overwrite=False,
        compute_stats=False,
        num_workers=1,
        validate_padding=False,
    )


def export_scene_worker(scene_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if WORKER_EXPORTER is None:
        raise RuntimeError("Scene export worker was not initialized")
    return WORKER_EXPORTER._export_scene(int(scene_id))


class ProcessedSceneStoreExporter:
    def __init__(
        self,
        raw_root: str | Path = RAW_DATA_DIR,
        csv_root: str | Path = CSV_DATA_DIR,
        output_root: str | Path = EUMP_DATA_DIR,
        *,
        overwrite: bool = False,
        scene_ids: list[int] | None = None,
        compute_stats: bool = True,
        num_workers: int = 0,
        validate_padding: bool = False,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.csv_root = Path(csv_root)
        self.output_root = Path(output_root)
        self.overwrite = overwrite
        self.scene_ids = None if scene_ids is None else sorted(int(scene_id) for scene_id in scene_ids)
        self.compute_stats = compute_stats
        self.num_workers = int(num_workers)
        self.export_warnings: list[str] = []
        self.schema = build_eump_store_schema()
        self.source_specs = self.schema["sources"]
        self.padding_validation = (
            validate_bh8_padding_facts(self.raw_root / "BH8_v1")
            if validate_padding
            else {"skipped": True, "reason": "validate_padding was not requested"}
        )
        self.climate_by_scene = self._load_climate_table()
        self.traffic_by_scene = self._load_traffic_table()
        self.pm25_mean_by_scene = self._load_pm25_mean_table()
        self.ghg_by_scene = self._load_ghg_table()

    def export(self) -> dict[str, Any]:
        self._prepare_output_root()
        scene_ids = self.scene_ids or self._discover_scene_ids()

        self.schema["bh8_padding_validation"] = self.padding_validation
        write_json(schema_path(self.output_root), self.schema)

        scene_rows: list[dict[str, Any]] = []
        scene_year_rows: list[dict[str, Any]] = []

        requested_worker_count = self._resolve_worker_count(len(scene_ids))
        scene_exports, actual_worker_count = self._export_scenes(scene_ids, requested_worker_count)
        for scene_row, yearly_rows in scene_exports:
            scene_rows.append(scene_row)
            scene_year_rows.extend(yearly_rows)

        write_jsonl(scenes_manifest_path(self.output_root), scene_rows)
        write_jsonl(scene_years_manifest_path(self.output_root), scene_year_rows)

        stats = {}
        if self.compute_stats:
            stats = self.compute_normalization_stats()

        return {
            "eump_root": str(self.output_root),
            "scene_count": len(scene_rows),
            "scene_year_count": len(scene_year_rows),
            "stats_written": bool(stats),
            "worker_count": actual_worker_count,
            "warnings": list(self.export_warnings),
            "bh8_padding_validation": self.padding_validation,
        }

    def _resolve_worker_count(self, scene_count: int) -> int:
        if scene_count <= 1:
            return 1
        if self.num_workers > 0:
            return max(1, min(int(self.num_workers), scene_count))
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, scene_count))

    def _export_scenes(
        self,
        scene_ids: list[int],
        worker_count: int,
    ) -> tuple[list[tuple[dict[str, Any], list[dict[str, Any]]]], int]:
        if worker_count <= 1:
            return [
                self._export_scene(scene_id)
                for scene_id in tqdm(scene_ids, desc="Export scenes", unit="scene")
            ], 1

        exported: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
                initializer=init_scene_export_worker,
                initargs=(str(self.raw_root), str(self.csv_root), str(self.output_root)),
            ) as executor:
                future_map = {
                    executor.submit(export_scene_worker, int(scene_id)): int(scene_id)
                    for scene_id in scene_ids
                }
                for future in tqdm(
                    as_completed(future_map),
                    total=len(future_map),
                    desc=f"Export scenes ({worker_count} workers)",
                    unit="scene",
                ):
                    exported.append(future.result())
        except (OSError, PermissionError) as exc:
            self.export_warnings.append(
                "Multiprocess scene export was unavailable in the current runtime; "
                f"fell back to serial export. Original error: {exc}"
            )
            return [
                self._export_scene(scene_id)
                for scene_id in tqdm(scene_ids, desc="Export scenes", unit="scene")
            ], 1

        exported.sort(key=lambda item: int(item[0]["scene_id"]))
        return exported, worker_count

    def _prepare_output_root(self) -> None:
        if self.output_root.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output root already exists: {self.output_root}. Use overwrite=True to replace it."
                )
            shutil.rmtree(self.output_root)

        (self.output_root / "manifest" / "stats").mkdir(parents=True, exist_ok=True)
        (self.output_root / "scenes").mkdir(parents=True, exist_ok=True)

    def _discover_scene_ids(self) -> list[int]:
        bh8_root = self.raw_root / "BH8_v1"
        scene_ids = []
        for path in sorted(bh8_root.glob("BH8_250_*.tif")):
            stem = path.stem
            scene_ids.append(int(stem.split("_")[-1]))
        return scene_ids

    def _load_climate_table(self) -> dict[int, dict[str, Any]]:
        table = pd.read_csv(self.csv_root / "clim_values.csv")
        table["ID"] = table["ID"].astype(int)
        return {int(row["ID"]): row.to_dict() for _, row in table.iterrows()}

    def _load_traffic_table(self) -> dict[int, dict[str, Any]]:
        table = pd.read_csv(self.csv_root / "traffic.csv")
        table["ID"] = table["ID"].astype(int)
        return {int(row["ID"]): row.to_dict() for _, row in table.iterrows()}

    def _load_ghg_table(self) -> dict[int, dict[str, Any]]:
        path = self.csv_root / GHG_SCALAR_CSV_NAME
        table = pd.read_csv(path, encoding="utf-8-sig")
        if "ID" not in table.columns:
            raise ValueError(f"{path} must contain an ID column")
        table["ID"] = table["ID"].astype(int)

        duplicate_ids = sorted(int(scene_id) for scene_id in table.loc[table["ID"].duplicated(), "ID"].unique())
        if duplicate_ids:
            preview = ", ".join(str(scene_id) for scene_id in duplicate_ids[:10])
            suffix = "" if len(duplicate_ids) <= 10 else f", ... ({len(duplicate_ids)} total)"
            raise ValueError(f"{path} contains duplicate ID values: {preview}{suffix}")

        required_columns: list[str] = []
        for source_name, raw_prefix in GHG_SCALAR_SOURCES.items():
            spec = self.source_specs[source_name]
            required_columns.extend(f"{raw_prefix}_{year}" for year in spec["canonical_years"])
        missing_columns = sorted(column for column in required_columns if column not in table.columns)
        if missing_columns:
            preview = ", ".join(missing_columns[:10])
            suffix = "" if len(missing_columns) <= 10 else f", ... ({len(missing_columns)} total)"
            raise ValueError(f"{path} is missing required GHG columns: {preview}{suffix}")

        return {int(row["ID"]): row.to_dict() for _, row in table.iterrows()}

    def _load_pm25_mean_table(self) -> dict[int, dict[str, Any]]:
        path = self.csv_root / PM25_MEAN_SCALAR_CSV_NAME
        table = pd.read_csv(path, encoding="utf-8-sig")
        if "ID" not in table.columns:
            raise ValueError(f"{path} must contain an ID column")
        table["ID"] = table["ID"].astype(int)

        duplicate_ids = sorted(int(scene_id) for scene_id in table.loc[table["ID"].duplicated(), "ID"].unique())
        if duplicate_ids:
            preview = ", ".join(str(scene_id) for scene_id in duplicate_ids[:10])
            suffix = "" if len(duplicate_ids) <= 10 else f", ... ({len(duplicate_ids)} total)"
            raise ValueError(f"{path} contains duplicate ID values: {preview}{suffix}")

        spec = self.source_specs[PM25_MEAN_SOURCE]
        required_columns = [f"{PM25_MEAN_SOURCE}_{year}" for year in spec["canonical_years"]]
        missing_columns = sorted(column for column in required_columns if column not in table.columns)
        if missing_columns:
            preview = ", ".join(missing_columns[:10])
            suffix = "" if len(missing_columns) <= 10 else f", ... ({len(missing_columns)} total)"
            raise ValueError(f"{path} is missing required PM2.5 mean columns: {preview}{suffix}")

        return {int(row["ID"]): row.to_dict() for _, row in table.iterrows()}

    def _load_reference_grid(self, scene_id: int) -> tuple[ReferenceGrid, np.ndarray, tuple[str, ...]]:
        rasterio, _, _ = import_rasterio()
        path = self.raw_root / "BH8_v1" / f"BH8_250_{scene_id}.tif"
        if not path.exists():
            raise FileNotFoundError(f"Missing BH8 reference file for scene {scene_id}: {path}")

        with rasterio.open(path) as src:
            data = src.read()
            descriptions = tuple(src.descriptions)
            reference = ReferenceGrid(
                scene_id=scene_id,
                height=src.height,
                width=src.width,
                crs=src.crs,
                crs_wkt=None if src.crs is None else src.crs.to_wkt(),
                transform=src.transform,
                transform_gdal=list(src.transform.to_gdal()),
                path=path,
            )
        return reference, data, descriptions

    def _align_band_to_reference(
        self,
        src_path: Path,
        reference: ReferenceGrid,
        *,
        band_index: int = 1,
    ) -> np.ndarray:
        rasterio, resampling_enum, reproject = import_rasterio()
        with rasterio.open(src_path) as src:
            source = src.read(band_index)
            destination = np.empty(reference.spatial_shape, dtype=source.dtype)
            reproject(
                source=source,
                destination=destination,
                src_transform=src.transform,
                src_crs=reference.crs if src.crs is None else src.crs,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                resampling=resampling_enum.nearest,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,
            )
        return destination

    def _write_store(self, path: Path, root_attrs: dict[str, Any], payloads: dict[str, SourcePayload]) -> None:
        zarr = import_zarr()
        group = zarr.open_group(str(path), mode="w", zarr_version=2)
        group.attrs.update(root_attrs)
        zstd = compressor()

        for source_name, payload in sorted(payloads.items()):
            source_group = group.require_group(source_name)
            data = np.asarray(payload.data)
            valid_mask = np.asarray(payload.valid_mask, dtype=bool)
            if data.shape != valid_mask.shape:
                raise ValueError(f"{source_name} data/valid_mask shape mismatch: {data.shape} vs {valid_mask.shape}")

            if data.ndim == 3:
                chunks = raster_chunks(int(data.shape[-2]), int(data.shape[-1]))
            elif data.ndim == 1:
                chunks = scalar_chunks(int(data.shape[0]))
            else:
                raise ValueError(f"{source_name} must be 1D or 3D, got {data.shape}")

            source_group.create_dataset(
                "data",
                data=data,
                shape=data.shape,
                chunks=chunks,
                compressor=zstd,
                overwrite=True,
            )
            source_group.create_dataset(
                "valid_mask",
                data=valid_mask,
                shape=valid_mask.shape,
                chunks=chunks,
                compressor=zstd,
                overwrite=True,
            )
            source_group.attrs.update(payload.attrs)

    def _bh8_band_index(self, descriptions: tuple[str, ...], band_name: str) -> int:
        try:
            return descriptions.index(band_name)
        except ValueError as exc:
            raise KeyError(f"Missing BH8 band {band_name!r}") from exc

    def _collect_static_payloads(
        self,
        scene_id: int,
        reference: ReferenceGrid,
        bh8_data: np.ndarray,
        bh8_descriptions: tuple[str, ...],
    ) -> dict[str, SourcePayload]:
        payloads: dict[str, SourcePayload] = {}
        static_band_names = {
            "water": "water",
            "BH": "BH",
            "BF": "BF",
            "DEM": "DEM",
            CITY_MASK_SOURCE: "mask",
        }

        for source_name, raw_band_name in static_band_names.items():
            spec = self.source_specs[source_name]
            band_index = self._bh8_band_index(bh8_descriptions, raw_band_name)
            band = bh8_data[band_index]
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            payloads[source_name] = as_raster_payload(
                source_name,
                band,
                valid_mask,
                build_source_attrs(
                    spec,
                    source_name,
                    band[None, ...],
                    source_years=None,
                    source_path=str(reference.path.relative_to(self.raw_root.parent)),
                    channel_names=single_channel_names(source_name),
                ),
            )

        lcz_path = self.raw_root / "LCZ" / f"LCZ_250_{scene_id}.tif"
        if lcz_path.exists():
            spec = self.source_specs["LCZ"]
            band = self._align_band_to_reference(lcz_path, reference)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            payloads["LCZ"] = as_raster_payload(
                "LCZ",
                band,
                valid_mask,
                build_source_attrs(
                    spec,
                    "LCZ",
                    band[None, ...],
                    source_years=None,
                    source_path=str(lcz_path.relative_to(self.raw_root.parent)),
                    channel_names=["LCZ"],
                ),
            )

        return payloads

    def _collect_yearly_payloads(
        self,
        scene_id: int,
        reference: ReferenceGrid,
        bh8_data: np.ndarray,
        bh8_descriptions: tuple[str, ...],
    ) -> dict[int, dict[str, SourcePayload]]:
        yearly: dict[int, dict[str, SourcePayload]] = {}

        def add(year: int, payload: SourcePayload) -> None:
            yearly.setdefault(int(year), {})[payload.source_name] = payload

        self._collect_bh8_yearly(scene_id, bh8_data, bh8_descriptions, add)
        self._collect_vi3_yearly(scene_id, reference, add)
        self._collect_nl_yearly(scene_id, reference, add)
        self._collect_land_cover_yearly(scene_id, reference, add)
        self._collect_pm25_yearly(scene_id, reference, add)
        self._collect_road_density_yearly(scene_id, reference, add)
        self._collect_electricity_yearly(scene_id, reference, add)
        self._collect_gdp_yearly(scene_id, reference, add)
        self._collect_lst_yearly(scene_id, reference, add)
        self._collect_precipitation_yearly(scene_id, reference, add)
        self._collect_climate_scalars(scene_id, add)
        self._collect_traffic_scalars(scene_id, add)
        self._collect_pm25_mean_scalars(scene_id, add)
        self._collect_ghg_scalars(scene_id, add)

        return yearly

    def _collect_bh8_yearly(
        self,
        scene_id: int,
        bh8_data: np.ndarray,
        bh8_descriptions: tuple[str, ...],
        add,
    ) -> None:
        for source_name in ("GAIA", "POP"):
            spec = self.source_specs[source_name]
            start_year = min(spec["canonical_years"])
            end_year = max(spec["canonical_years"])
            base_band_name = "GAIA" if source_name == "GAIA" else "population"
            for year in range(start_year, end_year + 1):
                band_name = bh8_band_name(base_band_name, year, start_year=2003)
                band_index = self._bh8_band_index(bh8_descriptions, band_name)
                band = bh8_data[band_index]
                valid_mask = valid_mask_from_rule(band, spec["nodata"])
                payload = as_raster_payload(
                    source_name,
                    band,
                    valid_mask,
                    build_source_attrs(
                        spec,
                        source_name,
                        band[None, ...],
                        source_years=[year],
                        source_path=str(
                            (self.raw_root / "BH8_v1" / f"BH8_250_{scene_id}.tif").relative_to(self.raw_root.parent)
                        ),
                        channel_names=[source_name],
                    ),
                )
                add(year, payload)

    def _collect_vi3_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        path = self.raw_root / "VI3_v1" / f"VI3_500_{scene_id}.tif"
        if not path.exists():
            return
        rasterio, _, _ = import_rasterio()
        with rasterio.open(path) as src:
            descriptions = tuple(src.descriptions)
        spec = self.source_specs["NDVI"]
        for year in spec["canonical_years"]:
            band_name = bh8_band_name("NDVI", year, start_year=2003)
            if band_name not in descriptions:
                continue
            band_index = descriptions.index(band_name) + 1
            band = self._align_band_to_reference(path, reference, band_index=band_index)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "NDVI",
                    band,
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "NDVI",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["NDVI"],
                    ),
                ),
            )

    def _collect_nl_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        spec = self.source_specs["NL"]
        for year in spec["canonical_years"]:
            path = self.raw_root / "NL_v1" / f"NL_{year}_{scene_id}.tif"
            if not path.exists():
                continue
            band = self._align_band_to_reference(path, reference)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "NL",
                    band,
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "NL",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["NL"],
                    ),
                ),
            )

    def _collect_land_cover_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        path = self.raw_root / "land_cover" / f"LC_500_{scene_id}.tif"
        if not path.exists():
            return
        rasterio, _, _ = import_rasterio()
        with rasterio.open(path) as src:
            descriptions = tuple(src.descriptions)
        spec = self.source_specs["LC"]
        for year in spec["canonical_years"]:
            band_name = f"LC_{year}"
            if band_name not in descriptions:
                continue
            band_index = descriptions.index(band_name) + 1
            band = self._align_band_to_reference(path, reference, band_index=band_index)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "LC",
                    band,
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "LC",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["LC"],
                    ),
                ),
            )

    def _collect_pm25_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        path = self.raw_root / "PM25" / f"PM25_1000_{scene_id}.tif"
        if not path.exists():
            return
        rasterio, _, _ = import_rasterio()
        with rasterio.open(path) as src:
            descriptions = tuple(src.descriptions)
        spec = self.source_specs["PM25"]
        for year in spec["canonical_years"]:
            band_name = f"PM25_{year}"
            if band_name not in descriptions:
                continue
            band_index = descriptions.index(band_name) + 1
            band = self._align_band_to_reference(path, reference, band_index=band_index)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "PM25",
                    band.astype(np.float32, copy=False),
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "PM25",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["PM25"],
                    ),
                ),
            )

    def _collect_road_density_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        spec = self.source_specs["road_density"]
        for year in spec["canonical_years"]:
            matches = sorted((self.raw_root / "road_density" / str(year)).glob(f"{year}_{scene_id}_*_roads_250m.tif"))
            if not matches:
                continue
            path = matches[0]
            band = self._align_band_to_reference(path, reference)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "road_density",
                    band.astype(np.float32, copy=False),
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "road_density",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["road_density"],
                    ),
                ),
            )

    def _collect_electricity_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        spec = self.source_specs["Electricity"]
        for year in spec["canonical_years"]:
            matches = sorted((self.raw_root / "electricity").glob(f"{year}_{scene_id}_*_masked.tif"))
            if not matches:
                continue
            path = matches[0]
            band = self._align_band_to_reference(path, reference)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "Electricity",
                    band.astype(np.float32, copy=False),
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "Electricity",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["Electricity"],
                    ),
                ),
            )

    def _collect_gdp_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        path = self.raw_root / "GDP" / f"gdpTot_6bands_{scene_id}.tif"
        if not path.exists():
            return
        rasterio, _, _ = import_rasterio()
        with rasterio.open(path) as src:
            descriptions = tuple(src.descriptions)
        spec = self.source_specs["GDP"]
        for year in spec["canonical_years"]:
            band_name = f"gdp_{year}"
            if band_name not in descriptions:
                continue
            band_index = descriptions.index(band_name) + 1
            band = self._align_band_to_reference(path, reference, band_index=band_index)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "GDP",
                    band.astype(np.float32, copy=False),
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "GDP",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["GDP"],
                    ),
                ),
            )

    def _collect_lst_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        path = self.raw_root / "lst_v1" / f"lst_1000_{scene_id}.tif"
        if not path.exists():
            return
        rasterio, _, _ = import_rasterio()
        with rasterio.open(path) as src:
            descriptions = tuple(src.descriptions)
        spec = self.source_specs["LST"]
        for year in spec["canonical_years"]:
            band_name = bh8_band_name("LST_day", year, start_year=2003)
            if band_name not in descriptions:
                continue
            band_index = descriptions.index(band_name) + 1
            band = self._align_band_to_reference(path, reference, band_index=band_index)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    "LST",
                    band.astype(np.float32, copy=False),
                    valid_mask,
                    build_source_attrs(
                        spec,
                        "LST",
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=["LST"],
                    ),
                ),
            )

    def _collect_precipitation_yearly(self, scene_id: int, reference: ReferenceGrid, add) -> None:
        spec = self.source_specs[PRECIP_TARGET_SOURCE]
        for year in spec["canonical_years"]:
            path = self.raw_root / "precip_v1" / f"Precip_{year}_{scene_id}.tif"
            if not path.exists():
                continue
            band = self._align_band_to_reference(path, reference)
            valid_mask = valid_mask_from_rule(band, spec["nodata"])
            add(
                year,
                as_raster_payload(
                    PRECIP_TARGET_SOURCE,
                    band.astype(np.float32, copy=False),
                    valid_mask,
                    build_source_attrs(
                        spec,
                        PRECIP_TARGET_SOURCE,
                        band[None, ...],
                        source_years=[year],
                        source_path=str(path.relative_to(self.raw_root.parent)),
                        channel_names=[PRECIP_TARGET_SOURCE],
                    ),
                ),
            )

    def _collect_climate_scalars(self, scene_id: int, add) -> None:
        row = self.climate_by_scene.get(scene_id)
        if row is None:
            return
        source_map = {"Ta": "TLML", CLIMATE_PRECIP_SOURCE: "PRECTOTCORR", "wind": "SPEED"}
        for source_name, raw_name in source_map.items():
            spec = self.source_specs[source_name]
            for year in spec["canonical_years"]:
                column = scalar_column(raw_name, year, start_year=2003)
                if column not in row:
                    continue
                value = row[column]
                valid = pd.notna(value)
                add(
                    year,
                    as_scalar_payload(
                        source_name,
                        0.0 if not valid else float(value),
                        bool(valid),
                        build_source_attrs(
                            spec,
                            source_name,
                            np.asarray([0.0 if not valid else float(value)], dtype=np.float32),
                            source_years=[year],
                            source_path=str((self.csv_root / "clim_values.csv").relative_to(self.raw_root.parent)),
                            channel_names=[source_name],
                        ),
                    ),
                )

    def _collect_traffic_scalars(self, scene_id: int, add) -> None:
        row = self.traffic_by_scene.get(scene_id)
        if row is None:
            return
        spec = self.source_specs["Traffic"]
        for year in spec["canonical_years"]:
            column = f"Congestion level %（{year}）"
            if column not in row:
                continue
            value = row[column]
            valid = pd.notna(value)
            add(
                year,
                as_scalar_payload(
                    "Traffic",
                    0.0 if not valid else float(value),
                    bool(valid),
                    build_source_attrs(
                        spec,
                        "Traffic",
                        np.asarray([0.0 if not valid else float(value)], dtype=np.float32),
                        source_years=[year],
                        source_path=str((self.csv_root / "traffic.csv").relative_to(self.raw_root.parent)),
                        channel_names=["Traffic"],
                    ),
                ),
            )

    def _collect_pm25_mean_scalars(self, scene_id: int, add) -> None:
        row = self.pm25_mean_by_scene.get(scene_id)
        if row is None:
            return
        spec = self.source_specs[PM25_MEAN_SOURCE]
        source_path = str((self.csv_root / PM25_MEAN_SCALAR_CSV_NAME).relative_to(self.raw_root.parent))
        for year in spec["canonical_years"]:
            column = f"{PM25_MEAN_SOURCE}_{year}"
            if column not in row:
                continue
            value = row[column]
            valid = pd.notna(value)
            scalar_value = 0.0 if not valid else float(value)
            add(
                year,
                as_scalar_payload(
                    PM25_MEAN_SOURCE,
                    scalar_value,
                    bool(valid),
                    build_source_attrs(
                        spec,
                        PM25_MEAN_SOURCE,
                        np.asarray([scalar_value], dtype=np.float32),
                        source_years=[year],
                        source_path=source_path,
                        channel_names=[PM25_MEAN_SOURCE],
                    ),
                ),
            )

    def _collect_ghg_scalars(self, scene_id: int, add) -> None:
        row = self.ghg_by_scene.get(scene_id)
        if row is None:
            return
        source_path = str((self.csv_root / GHG_SCALAR_CSV_NAME).relative_to(self.raw_root.parent))
        for source_name, raw_prefix in GHG_SCALAR_SOURCES.items():
            spec = self.source_specs[source_name]
            for year in spec["canonical_years"]:
                column = f"{raw_prefix}_{year}"
                if column not in row:
                    continue
                value = row[column]
                valid = pd.notna(value)
                scalar_value = 0.0 if not valid else float(value)
                add(
                    year,
                    as_scalar_payload(
                        source_name,
                        scalar_value,
                        bool(valid),
                        build_source_attrs(
                            spec,
                            source_name,
                            np.asarray([scalar_value], dtype=np.float32),
                            source_years=[year],
                            source_path=source_path,
                            channel_names=[source_name],
                        ),
                    ),
                )

    def _export_scene(self, scene_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        reference, bh8_data, bh8_descriptions = self._load_reference_grid(scene_id)
        static_payloads = self._collect_static_payloads(scene_id, reference, bh8_data, bh8_descriptions)
        yearly_payloads = self._collect_yearly_payloads(scene_id, reference, bh8_data, bh8_descriptions)

        static_rel_path = Path("scenes") / str(scene_id) / "static.zarr"
        static_path = self.output_root / static_rel_path
        static_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_store(
            static_path,
            build_root_attrs(reference, scene_id, list(static_payloads)),
            static_payloads,
        )

        scene_row = {
            "scene_id": int(scene_id),
            "sample_id": int(scene_id),
            "static_path": str(static_rel_path),
            "present_sources": sorted(static_payloads),
            "missing_sources": sorted(
                source_name
                for source_name, spec in self.source_specs.items()
                if spec["storage_level"] == "static" and source_name not in static_payloads
            ),
            "years": sorted(int(year) for year in yearly_payloads),
            "spatial_shape": list(reference.spatial_shape),
            "crs_wkt": reference.crs_wkt,
            "transform_gdal": list(reference.transform_gdal),
            "reference_grid": "BH8_250m",
        }

        yearly_rows: list[dict[str, Any]] = []
        for year, payloads in sorted(yearly_payloads.items()):
            relative_path = Path("scenes") / str(scene_id) / "years" / f"{year}.zarr"
            year_path = self.output_root / relative_path
            year_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_store(
                year_path,
                build_root_attrs(reference, scene_id, list(payloads), sample_id=scene_id, year=year),
                payloads,
            )

            expected_yearly_sources = sorted(
                source_name
                for source_name, spec in self.source_specs.items()
                if spec["storage_level"] == "yearly" and year in spec["canonical_years"]
            )
            yearly_rows.append(
                {
                    "scene_id": int(scene_id),
                    "year": int(year),
                    "year_path": str(relative_path),
                    "present_sources": sorted(payloads),
                    "missing_sources": sorted(source_name for source_name in expected_yearly_sources if source_name not in payloads),
                    "has_derived_sources": any(self.source_specs[source_name]["is_derived"] for source_name in payloads),
                }
            )

        return scene_row, yearly_rows

    def compute_normalization_stats(self) -> dict[str, Any]:
        zarr = import_zarr()
        scene_rows = read_jsonl(scenes_manifest_path(self.output_root))
        scene_year_rows = read_jsonl(scene_years_manifest_path(self.output_root))
        accumulators: dict[str, dict[str, Any]] = {}

        def update(source_name: str, data: np.ndarray, valid_mask: np.ndarray) -> None:
            data_array = np.asarray(data)
            source_valid_mask = valid_mask_from_rule(data_array, self.source_specs[source_name]["nodata"])
            valid = data_array[np.asarray(valid_mask, dtype=bool) & source_valid_mask]
            if valid.size == 0:
                return
            valid = valid.astype(np.float64, copy=False)
            stats = accumulators.setdefault(
                source_name,
                {"count": 0, "sum": 0.0, "sum_sq": 0.0, "min": None, "max": None},
            )
            stats["count"] += int(valid.size)
            stats["sum"] += float(valid.sum())
            stats["sum_sq"] += float(np.square(valid).sum())
            current_min = float(valid.min())
            current_max = float(valid.max())
            stats["min"] = current_min if stats["min"] is None else min(stats["min"], current_min)
            stats["max"] = current_max if stats["max"] is None else max(stats["max"], current_max)

        for row in tqdm(scene_rows, desc="Compute normalization stats: static stores", unit="scene"):
            group = zarr.open_group(str(self.output_root / row["static_path"]), mode="r")
            for source_name in group.group_keys():
                source_group = group[source_name]
                update(source_name, np.asarray(source_group["data"]), np.asarray(source_group["valid_mask"]))

        for row in tqdm(scene_year_rows, desc="Compute normalization stats: yearly stores", unit="scene-year"):
            group = zarr.open_group(str(self.output_root / row["year_path"]), mode="r")
            for source_name in group.group_keys():
                source_group = group[source_name]
                update(source_name, np.asarray(source_group["data"]), np.asarray(source_group["valid_mask"]))

        normalization_stats: dict[str, Any] = {
            "schema_version": self.schema["schema_version"],
            "stats": {},
        }
        for source_name, stats in sorted(accumulators.items()):
            count = max(1, int(stats["count"]))
            mean = stats["sum"] / count
            variance = max(stats["sum_sq"] / count - mean ** 2, 0.0)
            normalization_stats["stats"][source_name] = {
                "count": int(stats["count"]),
                "mean": mean,
                "std": float(math.sqrt(variance)),
                "min": stats["min"],
                "max": stats["max"],
            }

        write_json(normalization_stats_path(self.output_root), normalization_stats)
        return normalization_stats


__all__ = [
    "GHG_SCALAR_CSV_NAME",
    "PM25_MEAN_SCALAR_CSV_NAME",
    "PM25_MEAN_SOURCE",
    "ProcessedSceneStoreExporter",
]
