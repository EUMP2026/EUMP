from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import CITY_MASK_SOURCE


@dataclass(frozen=True)
class ProcessedTaskConfig:
    input_sources: tuple[str, ...] = ()
    optional_input_sources: tuple[str, ...] = ()
    reconstruct_target_source: str | None = None
    aux_target_sources: tuple[str, ...] = ()
    optional_aux_target_sources: tuple[str, ...] = ()
    prediction_sources: tuple[str, ...] = ()
    city_mask_source: str | None = CITY_MASK_SOURCE

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ProcessedTaskConfig":
        return cls(
            input_sources=tuple(payload.get("input_sources", ())),
            optional_input_sources=tuple(payload.get("optional_input_sources", ())),
            reconstruct_target_source=payload.get("reconstruct_target_source"),
            aux_target_sources=tuple(payload.get("aux_target_sources", ())),
            optional_aux_target_sources=tuple(payload.get("optional_aux_target_sources", ())),
            prediction_sources=tuple(payload.get("prediction_sources", ())),
            city_mask_source=payload.get("city_mask_source", CITY_MASK_SOURCE),
        )

    def referenced_sources(
        self,
        include_city_mask: bool = False,
        include_optional: bool = True,
    ) -> tuple[str, ...]:
        optional_inputs = set(self.optional_input_sources)
        ordered = [
            source_name
            for source_name in self.input_sources
            if source_name not in optional_inputs
        ]
        if self.reconstruct_target_source is not None:
            ordered.append(self.reconstruct_target_source)
        ordered.extend(self.aux_target_sources)
        if include_optional:
            ordered.extend(self.optional_input_sources)
            ordered.extend(self.optional_aux_target_sources)
        ordered.extend(self.prediction_sources)
        if include_city_mask and self.city_mask_source is not None:
            ordered.append(self.city_mask_source)

        deduped: list[str] = []
        seen: set[str] = set()
        for source_name in ordered:
            if source_name in seen:
                continue
            seen.add(source_name)
            deduped.append(source_name)
        return tuple(deduped)


def load_task_config(path: str | Path) -> ProcessedTaskConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in task config, got {type(payload)!r}")
    return ProcessedTaskConfig.from_mapping(payload)


__all__ = ["ProcessedTaskConfig", "load_task_config"]
