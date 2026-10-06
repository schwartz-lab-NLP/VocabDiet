import json
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


def _aggregate_subset(
    summary: Dict[str, Any], group_names: List[str], subset: str
) -> Dict[str, Any]:
    totals = {
        "total": 0,
        "correct": 0,
        "total_base_correct": 0,
        "correct_base": 0,
        "total_non_default": 0,
        "correct_non_default": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
    }
    for group in group_names:
        group_data = summary["groups"].get(group, {})
        totals["total"] += int(group_data.get("total", 0))
        totals["correct"] += int(group_data.get("correct", 0))
        totals["total_base_correct"] += int(group_data.get("total_base_correct", 0))
        totals["correct_base"] += int(group_data.get("correct_base", 0))
        totals["total_non_default"] += int(group_data.get("total_non_default", 0))
        totals["correct_non_default"] += int(group_data.get("correct_non_default", 0))
        totals["tp"] += int(group_data.get("tp", 0))
        totals["fp"] += int(group_data.get("fp", 0))
        totals["tn"] += int(group_data.get("tn", 0))
        totals["fn"] += int(group_data.get("fn", 0))

    pred_pos = totals["tp"] + totals["fp"]
    pred_neg = totals["tn"] + totals["fn"]
    actual_pos = totals["tp"] + totals["fn"]

    def _safe_div(num: int, den: int) -> float:
        return (num / den) if den else 0.0

    precision = _safe_div(totals["tp"], pred_pos)
    recall = _safe_div(totals["tp"], actual_pos)
    pred_pos_tp_share = _safe_div(totals["tp"], pred_pos)
    pred_pos_fp_share = _safe_div(totals["fp"], pred_pos)
    pred_neg_tn_share = _safe_div(totals["tn"], pred_neg)
    pred_neg_fn_share = _safe_div(totals["fn"], pred_neg)

    total_positive = totals["total_non_default"]
    total_negative = totals["tn"] + totals["fp"]
    accuracy_positive = _safe_div(totals["correct_non_default"], total_positive)
    accuracy_negative = _safe_div(totals["tn"], total_negative)

    if subset == "positives":
        subset_total = total_positive
        subset_correct = totals["correct_non_default"]
        subset_accuracy = accuracy_positive
    elif subset == "negatives":
        subset_total = total_negative
        subset_correct = totals["tn"]
        subset_accuracy = accuracy_negative
    else:
        subset_total = totals["total"]
        subset_correct = totals["correct"]
        subset_accuracy = _safe_div(totals["correct"], totals["total"])

    return {
        "accuracy": subset_accuracy,
        "accuracy_base_correct": _safe_div(totals["correct_base"], totals["total_base_correct"]),
        "accuracy_non_default": accuracy_positive,
        "accuracy_positive": accuracy_positive,
        "accuracy_negative": accuracy_negative,
        "total": subset_total,
        "correct": subset_correct,
        "total_base_correct": totals["total_base_correct"],
        "total_non_default": total_positive,
        "total_negative": total_negative,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "tn": totals["tn"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "pred_pos_tp_share": pred_pos_tp_share,
        "pred_pos_fp_share": pred_pos_fp_share,
        "pred_neg_tn_share": pred_neg_tn_share,
        "pred_neg_fn_share": pred_neg_fn_share,
    }


class TransformationAccuracyAggregator:
    def __init__(
        self,
        type_config,
        group_value_names: Dict[str, List[str]],
        default_indices: Dict[str, List[int]],
        na_indices: Dict[str, List[int]],
        ignore_na_types: bool = True,
        collapse_na_types: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.type_config = type_config
        self.group_names = list(type_config.type_groups.keys())
        self.group_value_names = group_value_names
        self.default_indices = {g: set(v) for g, v in default_indices.items()}
        self.na_indices = {g: set(v) for g, v in na_indices.items()}
        self.ignore_na_types = ignore_na_types
        self.collapse_na_types = collapse_na_types

        self.device = device or torch.device("cpu")
        num_groups = len(self.group_names)

        self.group_total = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_correct = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_total_base = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_correct_base = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_total_non_default = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_correct_non_default = torch.zeros(
            num_groups, dtype=torch.long, device=self.device
        )
        self.group_tp = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_fp = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_tn = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.group_fn = torch.zeros(num_groups, dtype=torch.long, device=self.device)

        self.value_total = {}
        self.value_correct = {}
        self.group_sizes = {}
        self.group_confusion = {}
        self.group_allowed = {}
        self.group_collapse_map = {}
        for group in self.group_names:
            group_slice = self.type_config.get_slice(group)
            group_size = group_slice.stop - group_slice.start
            self.value_total[group] = torch.zeros(group_size, dtype=torch.long, device=self.device)
            self.value_correct[group] = torch.zeros(
                group_size, dtype=torch.long, device=self.device
            )
            self.group_sizes[group] = group_size
            self.group_confusion[group] = {}

            names = self.group_value_names.get(group, [])
            name_to_idx = {name: idx for idx, name in enumerate(names)}
            if self.collapse_na_types and group_size > 1:
                collapse_map = torch.arange(group_size, dtype=torch.long, device=self.device)
                if names and names[-1].lower().startswith("na_"):
                    collapse_map[-1] = 0
                self.group_collapse_map[group] = collapse_map
            allowed = torch.eye(group_size, dtype=torch.bool, device=self.device)
            prefix = f"{group}_"
            for idx, name in enumerate(names):
                if not name.startswith(prefix):
                    continue
                value = name[len(prefix) :]
                if "+" not in value:
                    continue
                for part in value.split("+"):
                    comp_name = prefix + part
                    comp_idx = name_to_idx.get(comp_name)
                    if comp_idx is not None and 0 <= comp_idx < group_size:
                        allowed[idx, comp_idx] = True
            self.group_allowed[group] = allowed

        self.combo_total = torch.tensor(0, dtype=torch.long, device=self.device)
        self.combo_correct = torch.tensor(0, dtype=torch.long, device=self.device)
        self.combo_total_base = torch.tensor(0, dtype=torch.long, device=self.device)
        self.combo_correct_base = torch.tensor(0, dtype=torch.long, device=self.device)

    def _build_mask(
        self, labels: torch.Tensor, group_name: str, base_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = base_mask
        if self.ignore_na_types and self.na_indices.get(group_name):
            na_mask = torch.zeros_like(labels, dtype=torch.bool)
            for idx in self.na_indices[group_name]:
                na_mask |= labels == idx
            mask = mask & (~na_mask)
        return mask

    def update(
        self,
        type_logits: torch.Tensor,
        type_labels: torch.Tensor,
        base_preds: Optional[torch.Tensor],
        base_targets: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if valid_mask is None:
            valid_mask = torch.ones(
                type_logits.shape[0], dtype=torch.bool, device=type_logits.device
            )

        base_correct = None
        if base_preds is not None and base_targets is not None:
            base_correct = base_preds == base_targets

        combo_mask = valid_mask.clone()
        combo_correct = torch.ones_like(valid_mask, dtype=torch.bool)
        combo_positive = torch.zeros_like(valid_mask, dtype=torch.bool)
        combo_has_default_groups = False

        for group_idx, group_name in enumerate(self.group_names):
            group_slice = self.type_config.get_slice(group_name)
            labels = type_labels[..., group_slice].argmax(-1)
            preds = type_logits[..., group_slice].argmax(-1)
            if self.collapse_na_types:
                collapse_map = self.group_collapse_map.get(group_name)
                if collapse_map is not None:
                    labels = collapse_map[labels]
                    preds = collapse_map[preds]

            mask = self._build_mask(labels, group_name, valid_mask)
            labels_masked = labels[mask]
            preds_masked = preds[mask]

            value_mask = valid_mask
            labels_value = labels[value_mask]
            preds_value = preds[value_mask]

            total = labels_masked.numel()
            if total > 0:
                allowed = self.group_allowed.get(group_name)
                if allowed is not None:
                    correct_mask = allowed[labels_masked, preds_masked]
                    correct = correct_mask.sum()
                else:
                    correct = (preds_masked == labels_masked).sum()
            else:
                correct = torch.tensor(0, dtype=torch.long, device=self.device)

            self.group_total[group_idx] += total
            self.group_correct[group_idx] += correct

            # Base-correct conditional
            if base_correct is not None:
                base_mask = mask & base_correct
                labels_base = labels[base_mask]
                preds_base = preds[base_mask]
                total_base = labels_base.numel()
                if total_base > 0:
                    allowed = self.group_allowed.get(group_name)
                    if allowed is not None:
                        correct_base = allowed[labels_base, preds_base].sum()
                    else:
                        correct_base = (preds_base == labels_base).sum()
                else:
                    correct_base = torch.tensor(0, dtype=torch.long, device=self.device)
                self.group_total_base[group_idx] += total_base
                self.group_correct_base[group_idx] += correct_base

            # Non-default accuracy
            if self.default_indices.get(group_name):
                default_mask = torch.zeros_like(labels, dtype=torch.bool)
                for idx in self.default_indices[group_name]:
                    default_mask |= labels == idx
                non_default_mask = mask & (~default_mask)
                combo_positive |= non_default_mask
                combo_has_default_groups = True
                labels_non_default = labels[non_default_mask]
                preds_non_default = preds[non_default_mask]
                total_non_default = labels_non_default.numel()
                if total_non_default > 0:
                    allowed = self.group_allowed.get(group_name)
                    if allowed is not None:
                        correct_non_default = allowed[labels_non_default, preds_non_default].sum()
                    else:
                        correct_non_default = (preds_non_default == labels_non_default).sum()
                else:
                    correct_non_default = torch.tensor(0, dtype=torch.long, device=self.device)
                self.group_total_non_default[group_idx] += total_non_default
                self.group_correct_non_default[group_idx] += correct_non_default

                # Confusion counts at group level (positive = non-default)
                pred_default_mask = torch.zeros_like(preds, dtype=torch.bool)
                for idx in self.default_indices[group_name]:
                    pred_default_mask |= preds == idx
                pred_positive = mask & (~pred_default_mask)
                label_positive = mask & (~default_mask)
                tp = (pred_positive & label_positive).sum()
                fp = (pred_positive & (~label_positive)).sum()
                fn = ((~pred_positive) & label_positive).sum()
                tn = ((~pred_positive) & (~label_positive)).sum()
                self.group_tp[group_idx] += tp
                self.group_fp[group_idx] += fp
                self.group_fn[group_idx] += fn
                self.group_tn[group_idx] += tn

            # Per-value counts (include NA values even when ignored for accuracy)
            if labels_value.numel() > 0:
                group_size = group_slice.stop - group_slice.start
                self.value_total[group_name] += torch.bincount(labels_value, minlength=group_size)
                allowed = self.group_allowed.get(group_name)
                if allowed is not None:
                    matched = labels_value[allowed[labels_value, preds_value]]
                else:
                    matched = labels_value[preds_value == labels_value]
                if matched.numel() > 0:
                    self.value_correct[group_name] += torch.bincount(matched, minlength=group_size)

            # Positive-only confusion matrix (non-default labels)
            if self.default_indices.get(group_name):
                pos_labels = labels[non_default_mask]
                pos_preds = preds[non_default_mask]
                if pos_labels.numel() > 0:
                    group_size = group_slice.stop - group_slice.start
                    pair_index = pos_labels * group_size + pos_preds
                    uniq, counts = pair_index.unique(return_counts=True)
                    pair_counts = self.group_confusion[group_name]
                    for pair, count in zip(uniq.tolist(), counts.tolist()):
                        pair_counts[pair] = pair_counts.get(pair, 0) + int(count)

            # Combo accuracy tracking
            na_mask = torch.zeros_like(labels, dtype=torch.bool)
            if self.ignore_na_types and self.na_indices.get(group_name):
                for idx in self.na_indices[group_name]:
                    na_mask |= labels == idx
                combo_mask = combo_mask & (~na_mask)
                group_correct = (preds == labels) | na_mask
            else:
                group_correct = preds == labels
            combo_correct = combo_correct & group_correct

        if combo_has_default_groups:
            combo_mask = combo_mask & combo_positive

        combo_total = combo_mask.sum()
        combo_correct_count = (combo_correct & combo_mask).sum()
        self.combo_total += combo_total
        self.combo_correct += combo_correct_count

        if base_correct is not None:
            combo_base_mask = combo_mask & base_correct
            combo_total_base = combo_base_mask.sum()
            combo_correct_base = (combo_correct & combo_base_mask).sum()
            self.combo_total_base += combo_total_base
            self.combo_correct_base += combo_correct_base

    def all_reduce(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            return
        tensors = [
            self.group_total,
            self.group_correct,
            self.group_total_base,
            self.group_correct_base,
            self.group_total_non_default,
            self.group_correct_non_default,
            self.group_tp,
            self.group_fp,
            self.group_tn,
            self.group_fn,
            self.combo_total,
            self.combo_correct,
            self.combo_total_base,
            self.combo_correct_base,
        ]
        for tensor in tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        for group in self.group_names:
            dist.all_reduce(self.value_total[group], op=dist.ReduceOp.SUM)
            dist.all_reduce(self.value_correct[group], op=dist.ReduceOp.SUM)

    def to_summary(self) -> Dict[str, Any]:
        summary = {
            "combo_accuracy": (
                (self.combo_correct.float() / self.combo_total.float()).item()
                if self.combo_total.item() > 0
                else 0.0
            ),
            "combo_accuracy_base_correct": (
                (self.combo_correct_base.float() / self.combo_total_base.float()).item()
                if self.combo_total_base.item() > 0
                else 0.0
            ),
            "combo_total": int(self.combo_total.item()),
            "combo_total_base_correct": int(self.combo_total_base.item()),
            "groups": {},
        }

        for group_idx, group_name in enumerate(self.group_names):
            total = int(self.group_total[group_idx].item())
            correct = int(self.group_correct[group_idx].item())
            total_base = int(self.group_total_base[group_idx].item())
            correct_base = int(self.group_correct_base[group_idx].item())
            total_non_default = int(self.group_total_non_default[group_idx].item())
            correct_non_default = int(self.group_correct_non_default[group_idx].item())
            tp = int(self.group_tp[group_idx].item())
            fp = int(self.group_fp[group_idx].item())
            tn = int(self.group_tn[group_idx].item())
            fn = int(self.group_fn[group_idx].item())
            pred_pos = tp + fp
            pred_neg = tn + fn
            actual_pos = tp + fn
            precision = (tp / pred_pos) if pred_pos else 0.0
            recall = (tp / actual_pos) if actual_pos else 0.0
            pred_pos_tp_share = (tp / pred_pos) if pred_pos else 0.0
            pred_pos_fp_share = (fp / pred_pos) if pred_pos else 0.0
            pred_neg_tn_share = (tn / pred_neg) if pred_neg else 0.0
            pred_neg_fn_share = (fn / pred_neg) if pred_neg else 0.0
            total_negative = tn + fp
            accuracy_negative = (tn / total_negative) if total_negative else 0.0
            accuracy_positive = (
                (correct_non_default / total_non_default) if total_non_default else 0.0
            )

            group_summary = {
                "accuracy": (correct / total) if total else 0.0,
                "accuracy_base_correct": (correct_base / total_base) if total_base else 0.0,
                "accuracy_non_default": (correct_non_default / total_non_default)
                if total_non_default
                else 0.0,
                "accuracy_positive": accuracy_positive,
                "accuracy_negative": accuracy_negative,
                "total": total,
                "correct": correct,
                "total_base_correct": total_base,
                "correct_base": correct_base,
                "total_non_default": total_non_default,
                "correct_non_default": correct_non_default,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "pred_pos_tp_share": pred_pos_tp_share,
                "pred_pos_fp_share": pred_pos_fp_share,
                "pred_neg_tn_share": pred_neg_tn_share,
                "pred_neg_fn_share": pred_neg_fn_share,
            }

            names = self.group_value_names.get(group_name, [])
            value_totals = self.value_total[group_name].detach().cpu().tolist()
            value_corrects = self.value_correct[group_name].detach().cpu().tolist()
            values = []
            for idx, count in enumerate(value_totals):
                if count == 0:
                    continue
                name = names[idx] if idx < len(names) else f"idx_{idx}"
                acc = value_corrects[idx] / count if count else 0.0
                values.append({"name": name, "count": count, "accuracy": acc})
            values.sort(key=lambda x: x["count"], reverse=True)
            selected = values[:20]
            existing = {item["name"] for item in selected}
            for item in values:
                if item["name"].lower().startswith("na_") and item["name"] not in existing:
                    selected.append(item)
                    existing.add(item["name"])
            group_summary["values"] = selected
            summary["groups"][group_name] = group_summary

        summary["subsets"] = {
            "overall": _aggregate_subset(summary, self.group_names, "overall"),
            "positives": _aggregate_subset(summary, self.group_names, "positives"),
            "negatives": _aggregate_subset(summary, self.group_names, "negatives"),
            "group_sets": {
                "overall": self.group_names,
                "positives": self.group_names,
                "negatives": self.group_names,
            },
        }

        summary["group_value_names"] = self.group_value_names
        summary["group_confusion"] = self.get_confusion_payload()

        return summary

    def get_confusion_payload(self) -> Dict[str, Dict[str, Any]]:
        payload = {}
        for group_name in self.group_names:
            pairs = self.group_confusion.get(group_name, {})
            payload[group_name] = {
                "size": self.group_sizes.get(group_name, 0),
                "pairs": {str(k): int(v) for k, v in pairs.items()},
            }
        return payload


def merge_confusion_payloads(
    payloads: List[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for payload in payloads:
        if not payload:
            continue
        for group_name, data in payload.items():
            if group_name not in merged:
                merged[group_name] = {"size": data.get("size", 0), "pairs": {}}
            merged[group_name]["size"] = max(merged[group_name]["size"], data.get("size", 0))
            pairs = data.get("pairs", {})
            for key, value in pairs.items():
                merged[group_name]["pairs"][key] = merged[group_name]["pairs"].get(key, 0) + int(
                    value
                )
    return merged
