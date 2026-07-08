from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urban_embedding.core.paths import EUMP_DATA_DIR


EUMP_STORE_VERSION = "eump_scene_store_v1"
SCHEMA_FILENAME = "schema.json"
SCENES_MANIFEST_FILENAME = "scenes.jsonl"
SCENE_YEARS_MANIFEST_FILENAME = "scene_years.jsonl"
NORMALIZATION_FILENAME = "normalization_v1.json"
PREPROCESSING_FILENAME = "preprocessing_v1.json"
DEFAULT_RASTER_CHUNK = (1, 256, 256)
CITY_MASK_SOURCE = "city_mask"
AREA_SOURCE = "Area"
POP_SUM_SOURCE = "pop_sum"
PRECIP_TARGET_SOURCE = "precip"
CLIMATE_PRECIP_SOURCE = "precip_bg"


def years(start: int, end: int) -> tuple[int, ...]:
    return tuple(range(start, end + 1))


def as_jsonable_year_map(year_map: dict[int, int]) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in sorted(year_map.items())}


def as_int_year_map(year_map: dict[str, int] | dict[int, int]) -> dict[int, int]:
    return {int(key): int(value) for key, value in year_map.items()}


@dataclass(frozen=True)
class PaddingRule:
    kind: str
    year_map: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "year_map": as_jsonable_year_map(self.year_map),
        }


@dataclass(frozen=True)
class SourceSpec:
    source_name: str
    display_name: str
    raw_resolution: int | str
    raw_source_dir: str | None
    modality: str
    storage_level: str
    nodata: int | float | str | None
    is_derived: bool
    reference_grid: str
    raw_materialized_years: tuple[int, ...] = ()
    canonical_years: tuple[int, ...] = ()
    padding_rule: PaddingRule | None = None
    raw_source_name: str | None = None
    derived_from: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "display_name": self.display_name,
            "raw_resolution": self.raw_resolution,
            "raw_source_dir": self.raw_source_dir,
            "modality": self.modality,
            "storage_level": self.storage_level,
            "nodata": self.nodata,
            "is_derived": self.is_derived,
            "reference_grid": self.reference_grid,
            "raw_materialized_years": list(self.raw_materialized_years),
            "canonical_years": list(self.canonical_years),
            "padding_rule": None if self.padding_rule is None else self.padding_rule.to_dict(),
            "raw_source_name": self.raw_source_name,
            "derived_from": list(self.derived_from),
        }


def build_source_specs() -> dict[str, SourceSpec]:
    gaia_padding = PaddingRule(
        kind="repeat_last_real_year_materialized_in_raw",
        year_map={2019: 2018, 2020: 2018, 2021: 2018, 2022: 2018},
    )
    pop_padding = PaddingRule(
        kind="repeat_last_real_year_materialized_in_raw",
        year_map={2021: 2020, 2022: 2020},
    )

    specs = [
        SourceSpec("BH", "BH", 250, "raw_data/BH8_v1", "raster", "static", None, False, "BH8_250m"),
        SourceSpec("BF", "BF", 250, "raw_data/BH8_v1", "raster", "static", None, False, "BH8_250m"),
        SourceSpec("water", "water", 250, "raw_data/BH8_v1", "raster", "static", None, False, "BH8_250m"),
        SourceSpec("DEM", "DEM", 250, "raw_data/BH8_v1", "raster", "static", None, False, "BH8_250m"),
        SourceSpec("LCZ", "LCZ", 250, "raw_data/LCZ", "raster", "static", 0, False, "BH8_250m"),
        SourceSpec(
            CITY_MASK_SOURCE,
            "mask",
            250,
            "raw_data/BH8_v1",
            "raster",
            "static",
            None,
            False,
            "BH8_250m",
            raw_source_name="mask",
        ),
        SourceSpec(
            "NDVI",
            "NDVI",
            500,
            "raw_data/VI3_v1",
            "raster",
            "yearly",
            0,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
        ),
        SourceSpec(
            "NL",
            "NL",
            1000,
            "raw_data/NL_v1",
            "raster",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2000, 2021),
            canonical_years=years(2000, 2021),
        ),
        SourceSpec(
            "POP",
            "POP",
            250,
            "raw_data/BH8_v1",
            "raster",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2020),
            padding_rule=pop_padding,
            raw_source_name="population",
        ),
        SourceSpec(
            "LC",
            "LC",
            500,
            "raw_data/land_cover",
            "raster",
            "yearly",
            0,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
        ),
        SourceSpec(
            "GAIA",
            "GAIA",
            250,
            "raw_data/BH8_v1",
            "raster",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2018),
            padding_rule=gaia_padding,
        ),
        SourceSpec(
            "PM25",
            "PM2.5",
            1000,
            "raw_data/PM25",
            "raster",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2017, 2022),
            canonical_years=years(2017, 2022),
        ),
        SourceSpec(
            "road_density",
            "road_density",
            250,
            "raw_data/road_density",
            "raster",
            "yearly",
            -9999,
            False,
            "BH8_250m",
            raw_materialized_years=years(2008, 2022),
            canonical_years=years(2008, 2022),
        ),
        SourceSpec(
            "Electricity",
            "Electricity",
            1000,
            "raw_data/electricity",
            "raster",
            "yearly",
            -3.4e38,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2019),
            canonical_years=years(2003, 2019),
        ),
        SourceSpec(
            "GDP",
            "GDP",
            1000,
            "raw_data/GDP",
            "raster",
            "yearly",
            "nan",
            False,
            "BH8_250m",
            raw_materialized_years=(2000, 2005, 2010, 2015, 2020, 2024),
            canonical_years=(2000, 2005, 2010, 2015, 2020, 2024),
        ),
        SourceSpec(
            "LST",
            "LST",
            1000,
            "raw_data/lst_v1",
            "raster",
            "yearly",
            0,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
        ),
        SourceSpec(
            PRECIP_TARGET_SOURCE,
            "precip",
            4000,
            "raw_data/precip_v1",
            "raster",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2024),
            canonical_years=years(2003, 2024),
        ),
        SourceSpec(
            "Ta",
            "Ta",
            "scalar",
            "csv_data/clim_values.csv",
            "scalar",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
            raw_source_name="TLML",
        ),
        SourceSpec(
            CLIMATE_PRECIP_SOURCE,
            "precip_bg",
            "scalar",
            "csv_data/clim_values.csv",
            "scalar",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
            raw_source_name="PRECTOTCORR",
        ),
        SourceSpec(
            "wind",
            "wind",
            "scalar",
            "csv_data/clim_values.csv",
            "scalar",
            "yearly",
            None,
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
            raw_source_name="SPEED",
        ),
        SourceSpec(
            "Traffic",
            "Traffic",
            "scalar",
            "csv_data/traffic.csv",
            "scalar",
            "yearly",
            "nan",
            False,
            "BH8_250m",
            raw_materialized_years=years(2017, 2021),
            canonical_years=years(2017, 2021),
        ),
        SourceSpec(
            "PM25_mean",
            "PM2.5 mean",
            "scalar",
            "csv_data/pm25_mean.csv",
            "scalar",
            "yearly",
            "nan",
            False,
            "BH8_250m",
            raw_materialized_years=years(2015, 2019),
            canonical_years=years(2015, 2019),
        ),
        SourceSpec(
            "GHG_emission",
            "GHG emission sum",
            "scalar",
            "csv_data/ghg_area_weighted_sum_mean_2003_2022_wide.csv",
            "scalar",
            "yearly",
            "nan",
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
            raw_source_name="GHG_sum",
        ),
        SourceSpec(
            "GHG_mean",
            "GHG emission mean",
            "scalar",
            "csv_data/ghg_area_weighted_sum_mean_2003_2022_wide.csv",
            "scalar",
            "yearly",
            "nan",
            False,
            "BH8_250m",
            raw_materialized_years=years(2003, 2022),
            canonical_years=years(2003, 2022),
            raw_source_name="GHG_mean",
        ),
    ]
    return {spec.source_name: spec for spec in specs}


def build_runtime_derived_source_specs() -> dict[str, SourceSpec]:
    specs = [
        SourceSpec(
            AREA_SOURCE,
            AREA_SOURCE,
            "scalar",
            None,
            "scalar",
            "static",
            None,
            True,
            "BH8_250m",
            derived_from=(CITY_MASK_SOURCE,),
        ),
        SourceSpec(
            POP_SUM_SOURCE,
            POP_SUM_SOURCE,
            "scalar",
            None,
            "scalar",
            "yearly",
            None,
            True,
            "BH8_250m",
            raw_materialized_years=years(2003, 2020),
            canonical_years=years(2003, 2020),
            derived_from=("POP", CITY_MASK_SOURCE),
        ),
    ]
    return {spec.source_name: spec for spec in specs}


def build_eump_store_schema() -> dict[str, Any]:
    source_specs = build_source_specs()
    return {
        "schema_version": EUMP_STORE_VERSION,
        "reference_grid_source": "BH8_v1",
        "eump_root": str(Path(EUMP_DATA_DIR).name),
        "raster_chunk": list(DEFAULT_RASTER_CHUNK),
        "scalar_chunk": "full",
        "sources": {name: spec.to_dict() for name, spec in source_specs.items()},
    }


__all__ = [
    "AREA_SOURCE",
    "CITY_MASK_SOURCE",
    "CLIMATE_PRECIP_SOURCE",
    "DEFAULT_RASTER_CHUNK",
    "NORMALIZATION_FILENAME",
    "POP_SUM_SOURCE",
    "PRECIP_TARGET_SOURCE",
    "PREPROCESSING_FILENAME",
    "EUMP_STORE_VERSION",
    "PaddingRule",
    "SCENES_MANIFEST_FILENAME",
    "SCENE_YEARS_MANIFEST_FILENAME",
    "SCHEMA_FILENAME",
    "SourceSpec",
    "as_int_year_map",
    "as_jsonable_year_map",
    "build_eump_store_schema",
    "build_runtime_derived_source_specs",
    "build_source_specs",
    "years",
]
