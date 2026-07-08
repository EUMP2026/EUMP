from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from urban_embedding.data.processed.pipeline_spec import PipelineLossConfig, TargetSchema, is_hurdle_loss

from .losses import _align_mask, build_loss_mask, compute_weighted_loss, reduce_weighted_loss


LOSS_GROUPS = ("reconstruct", "aux")


@dataclass(slots=True)
class LossReport:
    total: torch.Tensor
    group_raw: dict[str, torch.Tensor]
    group_weighted: dict[str, torch.Tensor]
    target_raw: dict[str, torch.Tensor]
    target_weighted: dict[str, torch.Tensor]
    mask_stats: dict[str, torch.Tensor]
    grad_debug_terms: dict[str, torch.Tensor] = field(default_factory=dict)

    def metric_tensors(self) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        for name, value in self.target_raw.items():
            metrics[f"loss/raw/{name}"] = value
        for name, value in self.target_weighted.items():
            metrics[f"loss/weighted_target/{name}"] = value
        for group, value in self.group_raw.items():
            metrics[f"loss/group_raw/{group}"] = value
        for group, value in self.group_weighted.items():
            metrics[f"loss/group_weighted/{group}"] = value
        metrics["loss/total"] = self.total
        metrics.update(self.mask_stats)
        return metrics


@dataclass(slots=True)
class SliceLossEntry:
    name: str
    source: str
    raw_loss: torch.Tensor
    support_weight: torch.Tensor
    legacy_weight: float
    source_weight: float
    inner_weight: float
    debug_terms: dict[str, torch.Tensor]


def coverage(mask: torch.Tensor | None, *, like: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if mask is None or mask.numel() == 0:
        return like.new_tensor(float(default))
    return mask.float().mean()


def tensor_to_float(value: torch.Tensor | float | int) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _prediction_indices(indices: tuple[int, ...], fallback: tuple[int, ...]) -> tuple[int, ...]:
    return indices if indices else fallback


def _threshold_tensor(
    thresholds: tuple[float, ...],
    *,
    target_data: torch.Tensor,
) -> torch.Tensor:
    if len(thresholds) != int(target_data.shape[1]):
        raise ValueError(
            f"Hurdle threshold count {len(thresholds)} does not match target channels {target_data.shape[1]}"
        )
    values = target_data.new_tensor(tuple(float(value) for value in thresholds))
    return values.view(1, int(target_data.shape[1]), *([1] * (target_data.dim() - 2)))


def _hurdle_regression_loss(kind: str, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kind = kind.lower()
    if kind == "hurdle_mse":
        return (prediction - target) ** 2
    if kind == "hurdle_mae":
        return torch.abs(prediction - target)
    if kind == "hurdle_huber":
        return F.huber_loss(prediction, target, delta=1.0, reduction="none")
    raise ValueError(f"Unsupported hurdle loss kind {kind!r}")


def _compute_hurdle_loss(
    *,
    name: str,
    loss_kind: str,
    loss_config: PipelineLossConfig,
    value_prediction: torch.Tensor,
    logit_prediction: torch.Tensor,
    target_data: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    positive_thresholds: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    threshold = _threshold_tensor(positive_thresholds, target_data=target_data)
    positive_mask = target_data > threshold
    cls_target = positive_mask.to(dtype=target_data.dtype)

    cls_unreduced = F.binary_cross_entropy_with_logits(logit_prediction, cls_target, reduction="none")
    cls_loss = reduce_weighted_loss(cls_unreduced, mask=loss_mask)

    if loss_mask is None:
        reg_mask = valid_mask.bool() & positive_mask
    else:
        reg_mask = loss_mask.bool() & positive_mask
    reg_unreduced = _hurdle_regression_loss(loss_kind, value_prediction, target_data)
    reg_loss = reduce_weighted_loss(reg_unreduced, mask=reg_mask.float())
    if loss_mask is None:
        has_support = valid_mask.bool().any()
        positive_support = (valid_mask.bool() & positive_mask).any()
        valid_count = valid_mask.float().sum().clamp_min(1e-8)
        positive_ratio = (valid_mask.bool() & positive_mask).float().sum() / valid_count
    else:
        has_support = (loss_mask > 0).any()
        positive_support = (loss_mask.bool() & positive_mask).any()
        valid_count = loss_mask.float().sum().clamp_min(1e-8)
        positive_ratio = (loss_mask.bool() & positive_mask).float().sum() / valid_count

    support_weight = has_support.to(dtype=cls_loss.dtype, device=cls_loss.device)
    positive_support_weight = positive_support.to(dtype=cls_loss.dtype, device=cls_loss.device)
    cls_branch_weight = cls_loss.new_tensor(float(loss_config.hurdle_cls_weight)) * support_weight
    reg_branch_weight = cls_loss.new_tensor(float(loss_config.hurdle_reg_weight)) * positive_support_weight
    cls_component_raw = cls_loss * support_weight
    reg_component_raw = reg_loss * positive_support_weight
    if loss_config.hurdle_reduction == "weighted_mean":
        branch_weight_sum = (cls_branch_weight + reg_branch_weight).clamp_min(1e-8)
        cls_component = cls_component_raw * cls_branch_weight / branch_weight_sum
        reg_component = reg_component_raw * reg_branch_weight / branch_weight_sum
    else:
        cls_component = cls_component_raw * cls_branch_weight
        reg_component = reg_component_raw * reg_branch_weight
    raw_loss = cls_component + reg_component
    stats = {
        f"loss/hurdle_cls/{name}": cls_component,
        f"loss/hurdle_reg/{name}": reg_component,
        f"mask/hurdle_positive_ratio/{name}": positive_ratio * support_weight,
        f"mask/hurdle_has_positive/{name}": positive_support_weight,
    }
    debug_terms = {
        f"hurdle_cls/{name}": cls_component,
        f"hurdle_reg/{name}": reg_component,
    }
    return raw_loss, stats, support_weight, debug_terms


def _aggregate_reconstruction_slices_legacy(
    *,
    target_name: str,
    entries: list[SliceLossEntry],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    weighted_terms: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    debug_weighted: dict[str, torch.Tensor] = {}
    for entry in entries:
        effective_weight = entry.raw_loss.new_tensor(float(entry.legacy_weight)) * entry.support_weight
        weighted_terms.append(entry.raw_loss * effective_weight)
        weights.append(effective_weight)
        debug_weighted[f"slice/{target_name}/{entry.name}"] = entry.raw_loss * effective_weight
        for debug_name, debug_value in entry.debug_terms.items():
            debug_weighted[debug_name] = debug_value * effective_weight

    weight_sum = torch.stack(weights).sum() if weights else zero
    if weighted_terms:
        raw_loss = torch.stack(weighted_terms).sum() / weight_sum.clamp_min(1e-8)
        target_debug_terms = {
            name: value / weight_sum.clamp_min(1e-8)
            for name, value in debug_weighted.items()
        }
    else:
        raw_loss = zero
        target_debug_terms = {
            name: zero
            for name in debug_weighted
        }
    return raw_loss, target_debug_terms, weight_sum, {}


def _aggregate_reconstruction_slices_by_source(
    *,
    target_name: str,
    entries: list[SliceLossEntry],
    zero: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    by_source: dict[str, list[SliceLossEntry]] = defaultdict(list)
    for entry in entries:
        by_source[entry.source].append(entry)

    source_weighted_terms: list[torch.Tensor] = []
    source_weights: list[torch.Tensor] = []
    target_debug_terms: dict[str, torch.Tensor] = {}
    source_stats: dict[str, torch.Tensor] = {}
    source_payloads: list[tuple[str, torch.Tensor, torch.Tensor, list[tuple[SliceLossEntry, torch.Tensor]], dict[str, torch.Tensor]]] = []

    for source_name, source_entries in by_source.items():
        inner_weighted_terms: list[torch.Tensor] = []
        inner_weights: list[torch.Tensor] = []
        inner_debug_weighted: dict[str, torch.Tensor] = {}
        slice_payloads: list[tuple[SliceLossEntry, torch.Tensor]] = []
        for entry in source_entries:
            inner_weight = entry.raw_loss.new_tensor(float(entry.inner_weight)) * entry.support_weight
            inner_weighted_terms.append(entry.raw_loss * inner_weight)
            inner_weights.append(inner_weight)
            slice_payloads.append((entry, inner_weight))
            inner_debug_weighted[f"slice/{target_name}/{entry.name}"] = entry.raw_loss * inner_weight
            for debug_name, debug_value in entry.debug_terms.items():
                inner_debug_weighted[debug_name] = debug_value * inner_weight

        inner_weight_sum = torch.stack(inner_weights).sum() if inner_weights else zero
        if inner_weighted_terms:
            source_loss = torch.stack(inner_weighted_terms).sum() / inner_weight_sum.clamp_min(1e-8)
        else:
            source_loss = zero
        source_support = (inner_weight_sum > 0).to(dtype=source_loss.dtype, device=source_loss.device)
        source_weight = source_loss.new_tensor(float(source_entries[0].source_weight)) * source_support
        source_weighted_terms.append(source_loss * source_weight)
        source_weights.append(source_weight)
        source_stats[f"loss/raw_source/{target_name}/{source_name}"] = source_loss
        source_stats[f"mask/source_has_loss/{target_name}/{source_name}"] = source_support
        source_payloads.append((source_name, source_loss, inner_weight_sum, slice_payloads, inner_debug_weighted))

    source_weight_sum = torch.stack(source_weights).sum() if source_weights else zero
    if source_weighted_terms:
        raw_loss = torch.stack(source_weighted_terms).sum() / source_weight_sum.clamp_min(1e-8)
        for source_name, source_loss, inner_weight_sum, slice_payloads, inner_debug_weighted in source_payloads:
            source_weight_value = zero.new_tensor(float(by_source[source_name][0].source_weight))
            source_support = (inner_weight_sum > 0).to(dtype=raw_loss.dtype, device=raw_loss.device)
            source_factor = source_weight_value * source_support / source_weight_sum.clamp_min(1e-8)
            target_debug_terms[f"source/{target_name}/{source_name}"] = source_loss * source_factor
            for entry, inner_weight in slice_payloads:
                slice_factor = inner_weight / inner_weight_sum.clamp_min(1e-8) * source_factor
                target_debug_terms[f"slice/{target_name}/{entry.name}"] = entry.raw_loss * slice_factor
            for debug_name, debug_value in inner_debug_weighted.items():
                target_debug_terms[debug_name] = (
                    debug_value / inner_weight_sum.clamp_min(1e-8) * source_factor
                )
    else:
        raw_loss = zero
        for source_name, _, _, slice_payloads, inner_debug_weighted in source_payloads:
            target_debug_terms[f"source/{target_name}/{source_name}"] = zero
            for entry, _ in slice_payloads:
                target_debug_terms[f"slice/{target_name}/{entry.name}"] = zero
            for debug_name in inner_debug_weighted:
                target_debug_terms[debug_name] = zero

    return raw_loss, target_debug_terms, source_weight_sum, source_stats


def _apply_city_weight(
    loss_mask: torch.Tensor | None,
    city_mask: torch.Tensor | None,
    city_weight: float,
) -> torch.Tensor | None:
    if loss_mask is None or city_mask is None or city_weight <= 1.0:
        return loss_mask
    city_boost = city_mask.float().clamp(0, 1) * (city_weight - 1.0) + 1.0
    return loss_mask * _align_mask(city_boost, loss_mask)


def compute_target_loss(
    *,
    target: TargetSchema,
    prediction: torch.Tensor,
    target_data: torch.Tensor,
    valid_mask: torch.Tensor,
    city_mask: torch.Tensor | None = None,
    masked_patch: torch.Tensor | None = None,
    loss_config: PipelineLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
    loss_config = PipelineLossConfig() if loss_config is None else loss_config
    if target.loss_slices:
        entries: list[SliceLossEntry] = []
        stats: dict[str, torch.Tensor] = {}
        zero = prediction.sum() * 0.0

        for loss_slice in target.loss_slices:
            indices = list(loss_slice.channel_indices)
            prediction_indices = list(_prediction_indices(loss_slice.prediction_channel_indices, loss_slice.channel_indices))
            slice_prediction = prediction[:, prediction_indices, ...]
            slice_target_data = target_data[:, indices, ...]
            slice_valid_mask = valid_mask[:, indices, ...]
            finite_mask = torch.isfinite(slice_target_data)
            effective_valid_mask = slice_valid_mask.bool() & finite_mask
            safe_target_data = torch.nan_to_num(slice_target_data, nan=0.0, posinf=0.0, neginf=0.0)
            loss_mask = build_loss_mask(
                target.spec.mask,
                effective_valid_mask,
                city_mask=city_mask,
                masked_patch=masked_patch,
            )
            loss_mask = _apply_city_weight(loss_mask, city_mask, target.spec.city_weight)
            if is_hurdle_loss(loss_slice.loss):
                logit_indices = list(loss_slice.hurdle_logit_channel_indices)
                if not logit_indices:
                    raise ValueError(f"Hurdle loss slice {target.name}/{loss_slice.name} is missing logit channels")
                logit_prediction = prediction[:, logit_indices, ...]
                raw_loss, hurdle_stats, support_weight, hurdle_debug_terms = _compute_hurdle_loss(
                    name=f"{target.name}/{loss_slice.name}",
                    loss_kind=loss_slice.loss,
                    loss_config=loss_config,
                    value_prediction=slice_prediction,
                    logit_prediction=logit_prediction,
                    target_data=safe_target_data,
                    valid_mask=effective_valid_mask,
                    loss_mask=loss_mask,
                    positive_thresholds=loss_slice.positive_thresholds,
                )
                stats.update(hurdle_stats)
            else:
                raw_loss = compute_weighted_loss(loss_slice.loss, slice_prediction, safe_target_data, mask=loss_mask)
                if loss_mask is None:
                    has_support = finite_mask.any()
                else:
                    has_support = (loss_mask > 0).any()
                support_weight = has_support.to(dtype=raw_loss.dtype, device=raw_loss.device)
                raw_loss = raw_loss * support_weight
                hurdle_debug_terms = {}
            entries.append(
                SliceLossEntry(
                    name=loss_slice.name,
                    source=loss_slice.source,
                    raw_loss=raw_loss,
                    support_weight=support_weight,
                    legacy_weight=float(loss_slice.weight),
                    source_weight=float(loss_slice.source_weight),
                    inner_weight=float(loss_slice.inner_weight),
                    debug_terms=hurdle_debug_terms,
                )
            )
            stats.update(
                {
                    f"loss/raw_slice/{target.name}/{loss_slice.name}": raw_loss,
                    f"mask/slice_valid_ratio/{target.name}/{loss_slice.name}": coverage(
                        effective_valid_mask,
                        like=raw_loss,
                    ),
                    f"mask/slice_loss_mask_coverage/{target.name}/{loss_slice.name}": coverage(
                        loss_mask,
                        like=raw_loss,
                        default=1.0,
                    ),
                    f"mask/slice_has_loss/{target.name}/{loss_slice.name}": support_weight,
                }
            )

        if loss_config.reconstruct_reduction == "source_weighted_mean":
            raw_loss, target_debug_terms, support_weight, source_stats = _aggregate_reconstruction_slices_by_source(
                target_name=target.name,
                entries=entries,
                zero=zero,
            )
            stats.update(source_stats)
        else:
            raw_loss, target_debug_terms, support_weight, _ = _aggregate_reconstruction_slices_legacy(
                target_name=target.name,
                entries=entries,
                zero=zero,
            )
        effective_valid_mask = valid_mask.bool() & torch.isfinite(target_data)
        loss_mask = build_loss_mask(
            target.spec.mask,
            effective_valid_mask,
            city_mask=city_mask,
            masked_patch=masked_patch,
        )
        target_support = (support_weight > 0).to(dtype=raw_loss.dtype, device=raw_loss.device)
        stats.update(
            {
                f"mask/target_valid_ratio/{target.name}": coverage(effective_valid_mask, like=raw_loss),
                f"mask/loss_mask_coverage/{target.name}": coverage(loss_mask, like=raw_loss, default=1.0),
                f"mask/target_has_loss/{target.name}": target_support,
                f"loss/target_effective_weight/{target.name}": support_weight,
            }
        )
        return raw_loss, stats, support_weight, target_debug_terms

    finite_mask = torch.isfinite(target_data)
    effective_valid_mask = valid_mask.bool() & finite_mask
    safe_target_data = torch.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)
    loss_mask = build_loss_mask(
        target.spec.mask,
        effective_valid_mask,
        city_mask=city_mask,
        masked_patch=masked_patch,
    )
    loss_mask = _apply_city_weight(loss_mask, city_mask, target.spec.city_weight)
    if is_hurdle_loss(target.spec.loss):
        channels = int(target_data.shape[1])
        value_prediction = prediction[:, :channels, ...]
        logit_prediction = prediction[:, channels : channels * 2, ...]
        raw_loss, stats, support_weight, target_debug_terms = _compute_hurdle_loss(
            name=target.name,
            loss_kind=target.spec.loss,
            loss_config=loss_config,
            value_prediction=value_prediction,
            logit_prediction=logit_prediction,
            target_data=safe_target_data,
            valid_mask=effective_valid_mask,
            loss_mask=loss_mask,
            positive_thresholds=target.positive_thresholds,
        )
    else:
        raw_loss = compute_weighted_loss(target.spec.loss, prediction, safe_target_data, mask=loss_mask)
        if loss_mask is None:
            has_support = finite_mask.any()
        else:
            has_support = (loss_mask > 0).any()
        support_weight = has_support.to(dtype=raw_loss.dtype, device=raw_loss.device)
        raw_loss = raw_loss * support_weight
        stats = {}
        target_debug_terms = {}
    stats.update(
        {
            f"mask/target_valid_ratio/{target.name}": coverage(effective_valid_mask, like=raw_loss),
            f"mask/loss_mask_coverage/{target.name}": coverage(loss_mask, like=raw_loss, default=1.0),
            f"mask/target_has_loss/{target.name}": support_weight,
        }
    )
    return raw_loss, stats, support_weight, target_debug_terms


def _as_weight_tensor(value: torch.Tensor | float, *, zero: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=zero.device, dtype=zero.dtype)
    return zero.new_tensor(float(value))


def aggregate_loss_report(
    *,
    loss_config: PipelineLossConfig,
    target_raw: dict[str, torch.Tensor],
    target_weighted: dict[str, torch.Tensor],
    target_groups: dict[str, str],
    target_weights: dict[str, torch.Tensor | float],
    target_sources: dict[str, str] | None = None,
    target_debug_terms: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    mask_stats: dict[str, torch.Tensor],
    zero: torch.Tensor,
) -> LossReport:
    if not target_raw:
        raise ValueError("No configured losses were produced for the EUMP batch")

    group_raw: dict[str, torch.Tensor] = {}
    group_weighted: dict[str, torch.Tensor] = {}
    group_weight_sums: dict[str, torch.Tensor] = {}
    target_factors: dict[str, torch.Tensor] = {}
    source_debug_terms: dict[str, torch.Tensor] = {}
    total = zero
    target_sources = {name: name for name in target_raw} if target_sources is None else dict(target_sources)

    for group in LOSS_GROUPS:
        names = [name for name, target_group in target_groups.items() if target_group == group]
        if group == "aux" and loss_config.aux_reduction == "source_weighted_mean":
            source_names = sorted({target_sources.get(name, name) for name in names})
            source_weighted_terms: list[torch.Tensor] = []
            source_weights: list[torch.Tensor] = []
            source_payloads: list[tuple[str, torch.Tensor, torch.Tensor, list[str]]] = []
            for source_name in source_names:
                source_target_names = [name for name in names if target_sources.get(name, name) == source_name]
                target_effective_weights = [
                    _as_weight_tensor(target_weights[name], zero=zero)
                    for name in source_target_names
                ]
                target_weight_sum = torch.stack(target_effective_weights).sum() if target_effective_weights else zero
                if source_target_names:
                    source_numerator = torch.stack(
                        [target_weighted[name] for name in source_target_names]
                    ).sum()
                    source_raw = source_numerator / target_weight_sum.clamp_min(1e-8)
                else:
                    source_raw = zero
                configured_source_weight = loss_config.aux_source_weights.get(source_name, 1.0)
                source_support = (target_weight_sum > 0).to(dtype=source_raw.dtype, device=source_raw.device)
                source_weight = source_raw.new_tensor(float(configured_source_weight)) * source_support
                source_weighted_terms.append(source_raw * source_weight)
                source_weights.append(source_weight)
                source_payloads.append((source_name, source_raw, target_weight_sum, source_target_names))

            weight_sum = torch.stack(source_weights).sum() if source_weights else zero
            group_weight_sums[group] = weight_sum
            if names:
                raw_value = torch.stack(source_weighted_terms).sum() / weight_sum.clamp_min(1e-8)
                group_factor = zero.new_tensor(float(loss_config.group_weight(group)))
                for source_name, source_raw, target_weight_sum, source_target_names in source_payloads:
                    source_support = (target_weight_sum > 0).to(dtype=source_raw.dtype, device=source_raw.device)
                    source_weight = source_raw.new_tensor(
                        float(loss_config.aux_source_weights.get(source_name, 1.0))
                    ) * source_support
                    source_factor = source_weight / weight_sum.clamp_min(1e-8)
                    source_debug_terms[f"source/{group}/{source_name}"] = source_raw * group_factor * source_factor
                    for name in source_target_names:
                        target_weight = _as_weight_tensor(target_weights[name], zero=zero)
                        target_factors[name] = group_factor * source_factor * target_weight / target_weight_sum.clamp_min(1e-8)
            else:
                raw_value = zero
                for source_name, _, _, source_target_names in source_payloads:
                    source_debug_terms[f"source/{group}/{source_name}"] = zero
                    for name in source_target_names:
                        target_factors[name] = zero
        else:
            weights = [_as_weight_tensor(target_weights[name], zero=zero) for name in names]
            weight_sum = torch.stack(weights).sum() if weights else zero
            group_weight_sums[group] = weight_sum
            if names:
                numerator = torch.stack([target_weighted[name] for name in names]).sum()
                raw_value = numerator / weight_sum.clamp_min(1e-8)
                group_factor = zero.new_tensor(float(loss_config.group_weight(group)))
                for name in names:
                    target_weight = _as_weight_tensor(target_weights[name], zero=zero)
                    target_factors[name] = group_factor * target_weight / weight_sum.clamp_min(1e-8)
            else:
                raw_value = zero
                for name in names:
                    target_factors[name] = zero
        weighted_value = raw_value * float(loss_config.group_weight(group))
        group_raw[group] = raw_value
        group_weighted[group] = weighted_value
        total = total + weighted_value

    # Zero-weight targets may still have dedicated heads. Keep a zero-valued
    # graph edge to their raw losses so DDP does not flag those head parameters
    # as unused while preserving exactly the same numeric objective.
    unused_target_anchor = zero
    for raw_value in target_raw.values():
        unused_target_anchor = unused_target_anchor + raw_value * raw_value.new_tensor(0.0)
    total = total + unused_target_anchor

    target_debug_terms = {} if target_debug_terms is None else target_debug_terms
    grad_debug_terms: dict[str, torch.Tensor] = {"total": total}
    for group, value in group_weighted.items():
        grad_debug_terms[f"group/{group}"] = value
    grad_debug_terms.update(source_debug_terms)
    for name, raw_value in target_raw.items():
        factor = target_factors.get(name, zero)
        target_contribution = raw_value * factor
        grad_debug_terms[f"target/{name}"] = target_contribution
        for debug_name, debug_value in target_debug_terms.get(name, {}).items():
            grad_debug_terms[debug_name] = debug_value * factor

    return LossReport(
        total=total,
        group_raw=group_raw,
        group_weighted=group_weighted,
        target_raw=target_raw,
        target_weighted=target_weighted,
        mask_stats=mask_stats,
        grad_debug_terms=grad_debug_terms,
    )


def accumulate_metrics(
    sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    metrics: Mapping[str, torch.Tensor],
) -> None:
    for name, value in metrics.items():
        if torch.is_tensor(value):
            metric = value.detach().float()
        else:
            metric = torch.tensor(float(value))
        if name not in sums:
            sums[name] = metric
        else:
            current = sums[name]
            if metric.is_cuda and not current.is_cuda:
                current = current.to(device=metric.device)
            elif current.device != metric.device:
                metric = metric.to(device=current.device)
            sums[name] = current + metric
        counts[name] = counts.get(name, 0) + 1


def average_metrics(
    sums: dict[str, torch.Tensor],
    counts: dict[str, int],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: (sums[name].to(device=device) / max(1, counts[name]))
        for name in sums
    }


__all__ = [
    "LOSS_GROUPS",
    "LossReport",
    "accumulate_metrics",
    "aggregate_loss_report",
    "average_metrics",
    "compute_target_loss",
    "coverage",
    "tensor_to_float",
]
