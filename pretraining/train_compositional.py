import os
import sys

from pathlib import Path

code = Path(__file__).read_text()
import uuid
import time
import math
from dataclasses import dataclass
from functools import lru_cache
import json
import argparse

try:
    import wandb
except ImportError:  # Optional for local runs and command-line help.
    wandb = None
import copy
import torch
import numpy as np

from torch import Tensor, nn
import torch.distributed as dist

from compositional_tokenizer_bundle import get_or_create_tokenizer_bundle, get_tokenizer_cache_path
from modeling_compositional import CompositionalGPT, TypeConfig, next_multiple_of_n
from optimizer import DistAdam, Muon
from data_loader import distributed_data_generator
from eval_utils import (
    compute_validation_tokenization_stats,
    compute_factorized_bytes_per_token_metrics,
)
from transformers import AutoTokenizer
from train_gpt import (
    Hyperparameters as BaseHyperparameters,
    get_base_parser,
    save_checkpoint,
    load_checkpoint,
    nvidia_smi,
)
from gpu_utils import get_combined_gpu_stats

pass
from logging_utils import LossLogger, RunTracker, LogitsStatsCollector
from transformation_accuracy import merge_confusion_payloads


BASE_EMBED_LR = 0.0006
BASE_LM_HEAD_LR = 0.002
BASE_HIDDEN_LR = 0.005
MODEL_SIZE_LR_MUL = 1.0
TRAIN_BATCH_MODIFIER = 1
VAL_BATCH_MODIFIER = 1
TRAIN_ITERS_MODIFIER = 1

MUON_MOMENTUM = 0.95
ADAM_BETA1 = 0.8
ADAM_BETA2 = 0.95
MOMENTUM_WARMUP_GAP = 0.1
CLIP_GRADIENT = 1.0


def compute_top1_accuracies(
    model,
    model_output,
    extended_targets,
    base_targets,
    target_modifier_ids=None,
    collapse_na_types_metrics: bool = False,
):
    """
    Compute top-1 accuracies for extended vocab, base vocab, and type predictions.

    Args:
        model: CompositionalGPT model (for optional metric-only recomposition)
        model_output: CompositionalOutputs from model forward pass
        extended_targets: [seq_len] ground truth token IDs (extended vocab in single-stream; base IDs in dual-stream)
        base_targets: [seq_len] ground truth base vocab token IDs
        target_modifier_ids: Optional modifier IDs for targets (dual-stream mode)

    Returns:
        dict with accuracy metrics
    """
    accuracies = {}

    # Extended vocabulary accuracy (composed base + types)
    if model_output.top1_extended_idx is not None:
        extended_preds = model_output.top1_extended_idx
        if (
            collapse_na_types_metrics
            and model_output.type_logits is not None
            and model_output.base_logits is not None
        ):
            base_preds = model_output.base_logits.argmax(dim=-1)
            extended_preds = model.conditioned_transformation_head.compose_extended_token(
                base_preds,
                model_output.type_logits,
                collapse_na_types_override=True,
            )
        if (
            target_modifier_ids is not None
            and model_output.type_logits is not None
            and model_output.base_logits is not None
        ):
            target_old_type_ids = model.token_to_type_ids[extended_targets]
            target_new_type_ids = model.map_modifier_ids_to_type_ids(target_modifier_ids)
            target_type_ids = model.merge_type_ids(target_old_type_ids, target_new_type_ids)

            base_preds = model_output.base_logits.argmax(dim=-1)
            base_correct = base_preds == base_targets

            group_correct = torch.ones_like(base_correct, dtype=torch.bool)
            allowed_groups = getattr(
                model.conditioned_transformation_head, "composition_group_allowed", None
            )

            for group_idx, group_name in enumerate(model.type_config.type_groups.keys()):
                sl = model.type_config.get_slice(group_name)
                pred_rel_raw = model_output.type_logits[:, sl].argmax(dim=-1)
                target_rel_raw = target_type_ids[:, sl].argmax(dim=-1)
                group_size = sl.stop - sl.start
                na_rel = group_size - 1

                if collapse_na_types_metrics and group_size > 1:
                    pred_rel = torch.where(
                        pred_rel_raw == na_rel, torch.zeros_like(pred_rel_raw), pred_rel_raw
                    )
                    target_rel = torch.where(
                        target_rel_raw == na_rel, torch.zeros_like(target_rel_raw), target_rel_raw
                    )
                    na_mask = torch.zeros_like(target_rel, dtype=torch.bool)
                else:
                    pred_rel = pred_rel_raw
                    target_rel = target_rel_raw
                    na_mask = target_rel_raw == na_rel

                if allowed_groups:
                    allowed = allowed_groups[group_idx]
                    if allowed.device != pred_rel.device:
                        allowed = allowed.to(pred_rel.device)
                    match = allowed[target_rel, pred_rel]
                else:
                    match = pred_rel == target_rel

                if not collapse_na_types_metrics:
                    match = match | na_mask
                group_correct &= match

            extended_correct = base_correct & group_correct
        else:
            if target_modifier_ids is not None:
                target_old_type_ids = model.token_to_type_ids[extended_targets]
                target_new_type_ids = model.map_modifier_ids_to_type_ids(target_modifier_ids)
                target_type_ids = model.merge_type_ids(target_old_type_ids, target_new_type_ids)
                extended_targets = (
                    model.conditioned_transformation_head.compose_extended_token_from_type_ids(
                        base_targets,
                        target_type_ids,
                        collapse_na_types_override=collapse_na_types_metrics,
                    )
                )
            extended_correct = extended_preds == extended_targets

        accuracies["extended_accuracy"] = extended_correct.float().mean().item()

    # Base vocabulary accuracy
    if model_output.base_logits is not None:
        base_preds = model_output.base_logits.argmax(dim=-1)
        base_correct = (base_preds == base_targets).float()
        accuracies["base_accuracy"] = base_correct.mean().item()

    # Auxiliary head accuracy (if available)
    if model_output.aux_logits is not None:
        aux_preds = model_output.aux_logits.argmax(dim=-1)
        aux_correct = (aux_preds == extended_targets).float()
        accuracies["aux_accuracy"] = aux_correct.mean().item()

    return accuracies


def init_combo_resolution_stats(group_names):
    return {
        "total": 0,
        "eligible": 0,
        "exact": 0,
        "drop": 0,
        "no_match": 0,
        "base_no_combo": 0,
        "drop_counts": {},
        "drop_groups": {name: 0 for name in group_names},
        "invalid_transform_counts": {name: {} for name in group_names},
        "invalid_transform_totals": {name: 0 for name in group_names},
    }


def merge_combo_resolution_stats(total_stats, batch_stats):
    for key in ("total", "eligible", "exact", "drop", "no_match", "base_no_combo"):
        total_stats[key] += int(batch_stats.get(key, 0))
    for key, value in batch_stats.get("drop_counts", {}).items():
        total_stats["drop_counts"][key] = total_stats["drop_counts"].get(key, 0) + int(value)
    for key, value in batch_stats.get("drop_groups", {}).items():
        total_stats["drop_groups"][key] = total_stats["drop_groups"].get(key, 0) + int(value)
    for group_name, counts in batch_stats.get("invalid_transform_counts", {}).items():
        group_bucket = total_stats["invalid_transform_counts"].setdefault(group_name, {})
        for name, value in counts.items():
            group_bucket[name] = group_bucket.get(name, 0) + int(value)
    for group_name, value in batch_stats.get("invalid_transform_totals", {}).items():
        total_stats["invalid_transform_totals"][group_name] = total_stats[
            "invalid_transform_totals"
        ].get(group_name, 0) + int(value)
    return total_stats


def finalize_combo_resolution_stats(stats):
    eligible = stats.get("eligible", 0)
    invalid = stats.get("drop", 0) + stats.get("no_match", 0)
    stats["exact_rate"] = (stats.get("exact", 0) / eligible) if eligible else 0.0
    stats["drop_rate"] = (stats.get("drop", 0) / eligible) if eligible else 0.0
    stats["invalid_rate"] = (invalid / eligible) if eligible else 0.0
    stats["drop_given_invalid_rate"] = (stats.get("drop", 0) / invalid) if invalid else 0.0
    return stats


@dataclass
class CompositionalHyperparameters(BaseHyperparameters):
    """Extended hyperparameters for compositional training."""

    dataset = "HuggingFaceFW/fineweb"  # HuggingFace dataset name
    dataset_config = "sample-10BT"  # Dataset configuration
    output_dir = "vocab_diet_runs"  # data
    base_data_dir = None  # Will be constructed from dataset/config if not provided
    train_seq_len = int(TRAIN_BATCH_MODIFIER * 48) * 1024
    val_seq_len = int(VAL_BATCH_MODIFIER * 48) * 1024
    gradient_accumulation_steps = 1
    # optimization
    num_iterations = int(TRAIN_ITERS_MODIFIER * 1750)
    base_loss_alpha: float = 1.0
    type_loss_alpha: float = None
    aux_loss_alpha: float = 0.0
    aux_loss_until_step: int = None
    aux_loss_probe_only: bool = False
    use_shift: bool = False
    softcap: float = 30.0
    types_addition_alpha: float = 1.0
    use_mixing_combination: bool = False
    use_orthogonal_projections: bool = False
    base_proj_dim: int = None
    type_proj_dim: int = None
    ignore_na_types: bool = False
    collapse_na_types: bool = False
    collapse_na_types_metrics: bool = True
    combo_stats_max_samples: int = 4096
    combo_stats_entry_chunk: int = 65536
    base_correct_samples_per_group: int = 10
    ambiguous_type_loss_mode: str = "equal"
    ambiguous_type_loss_lambda: float = 1.0
    # Transformation head conditioning
    transformation_conditioning_mode: str = "concat"  # concat, add, or cross_attention
    transformation_base_rep_source: str = (
        "unembedding"  # paper conditions on the chosen base's output vector
    )
    transformation_normalize_base_rep: bool = (
        True  # Normalize base representation before conditioning
    )
    transformation_project_base_rep: bool = False  # Project base representation before conditioning
    transformation_base_rep_proj_dim: int = None  # Projection dimension (None = model_dim)
    # compositional vocab
    language: str = "en"
    morphology_mode: str = "atomic"  # "atomic" or "compositional"
    clitic_mode: str = "surface2"
    unimorph_root: str = None
    unimorph_inflections_path: str = None
    unimorph_derivations_path: str = None
    no_diacritics: bool = False
    skip_multi_token_words: bool = False
    skip_three_token_words: bool = False
    skip_four_token_words: bool = False
    skip_multi_token_bases: bool = False
    dont_decompose_spaces: bool = False
    skip_if_no_space_prefix: bool = True
    allow_no_space_prefix: bool = False
    dont_decompose_capitalized: bool = False
    use_relative_space_cap_transforms: bool = False
    canonicalize_token_strings: bool = False
    transform_collision_mode: str = "off"
    filter_transform_collisions: bool = False
    word_boundary_safety: bool = False
    multi_token_modifier_position: str = "last"
    include_derivations: bool = False
    prefer_popular_base_conflicts: bool = True
    drop_morphology_transforms: bool = False
    allow_chained_derivations: bool = False
    include_non_resource_latin_tokens: bool = False
    include_non_latin_tokens: bool = False
    use_type_str_as_type: bool = False
    # New transformation groups (disabled by default)
    decompose_punctuation: bool = False
    decompose_articles: bool = False
    decompose_prepositions: bool = False
    decompose_article_prep_space_prefix: bool = False
    preposition_list: list = None
    modifier_generation_mode: str = "auto"
    multitoken_allowed_groups: list = (
        None  # Whitelist of transformation groups that can be applied to multi-token words
    )
    base_embed_lr: float = BASE_EMBED_LR
    base_lm_head_lr: float = BASE_LM_HEAD_LR
    base_hidden_lr: float = BASE_HIDDEN_LR
    model_size_lr: float = MODEL_SIZE_LR_MUL
    muon_momentum: float = MUON_MOMENTUM
    adam_beta1: float = ADAM_BETA1
    adam_beta2: float = ADAM_BETA2
    momentum_warmup_gap: float = MOMENTUM_WARMUP_GAP
    clip_grad_norm: float = CLIP_GRADIENT


def parse_compositional_args():
    """Parse arguments with compositional extensions."""
    parser = get_base_parser()

    # Add compositional arguments
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="openai-community/gpt2",
        help="Base HuggingFace tokenizer to extend",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for UniMorph selection (default: en)",
    )
    parser.set_defaults(morphology_mode="atomic")
    parser.set_defaults(clitic_mode="surface2")
    parser.add_argument(
        "--unimorph_root",
        type=str,
        default=os.environ.get(
            "UNIMORPH_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "unimorph"),
        ),
        help="Base path to UniMorph language directories (default: UNIMORPH_ROOT or resources/unimorph)",
    )
    parser.add_argument(
        "--unimorph_inflections_path",
        type=str,
        default=None,
        help="Explicit UniMorph inflections path (overrides --language/--unimorph_root)",
    )
    parser.add_argument(
        "--unimorph_derivations_path",
        type=str,
        default=None,
        help="Explicit UniMorph derivations path",
    )
    parser.set_defaults(no_diacritics=False)
    parser.set_defaults(base_loss_alpha=1.0)
    parser.add_argument("--base_data_dir", type=str, default=None)
    parser.set_defaults(type_loss_alpha=None)
    parser.set_defaults(aux_loss_alpha=None)
    parser.set_defaults(aux_loss_until_step=None)
    parser.set_defaults(aux_loss_probe_only=False)
    parser.add_argument("--use_shift", action="store_true", default=False)
    parser.add_argument("--softcap", type=float, default=30.0)
    parser.add_argument("--skip_multi_token_words", action="store_true", default=False)
    parser.add_argument("--skip_three_token_words", action="store_true", default=False)
    parser.add_argument("--skip_four_token_words", action="store_true", default=False)
    parser.add_argument(
        "--skip_multi_token_bases",
        action="store_true",
        default=False,
        help="Skip multi-token base words from decomposition map construction "
        "(required for single_id mode)",
    )
    parser.add_argument("--dont_decompose_spaces", action="store_true", default=False)
    parser.add_argument("--skip_if_no_space_prefix", action="store_true", default=True)
    parser.add_argument(
        "--allow_no_space_prefix",
        action="store_true",
        default=False,
        help="Allow base tokens without space prefix for decomposition (sets skip_if_no_space_prefix=False).",
    )
    parser.add_argument("--dont_decompose_capitalized", action="store_true", default=False)
    parser.set_defaults(use_relative_space_cap_transforms=False)
    parser.set_defaults(canonicalize_token_strings=False)
    parser.set_defaults(transform_collision_mode="off")
    parser.set_defaults(filter_transform_collisions=False)
    parser.set_defaults(word_boundary_safety=False)
    parser.set_defaults(multi_token_modifier_position="last")
    parser.add_argument("--include_derivations", action="store_true", default=True)
    parser.add_argument(
        "--min_derivation_count",
        type=int,
        default=50,
        help="Minimum UniMorph derivation type frequency required to include derivation transforms.",
    )
    parser.set_defaults(prefer_popular_base_conflicts=True)
    parser.set_defaults(drop_morphology_transforms=False)
    parser.set_defaults(allow_chained_derivations=False)
    parser.add_argument("--use_type_str_as_type", action="store_true", default=False)
    parser.add_argument("--skip_stop_words", action="store_true", default=False)
    parser.add_argument("--filter_propernouns_and_aux", action="store_true", default=False)
    parser.add_argument("--merge_plural_and_present_singular", action="store_true", default=False)
    parser.set_defaults(include_non_resource_latin_tokens=True)
    parser.set_defaults(include_non_latin_tokens=True)
    # New transformation groups (opt-in)
    parser.set_defaults(decompose_punctuation=False)
    parser.set_defaults(decompose_articles=False)
    parser.set_defaults(decompose_prepositions=False)
    parser.set_defaults(decompose_article_prep_space_prefix=False)
    parser.set_defaults(preposition_list=None)
    parser.set_defaults(modifier_generation_mode="auto")
    parser.set_defaults(multitoken_allowed_groups=[])
    parser.set_defaults(tokenization_mode="single_id")
    parser.set_defaults(prune_punct_tokens=False)
    parser.set_defaults(possessive_separate_group=False)
    parser.set_defaults(enable_build_optimizations=False)
    parser.add_argument("--overwrite_bundle_cache", action="store_true", default=False)
    parser.set_defaults(types_addition_alpha=1.0)
    parser.set_defaults(use_mixing_combination=False)
    parser.set_defaults(use_orthogonal_projections=False)
    parser.set_defaults(base_proj_dim=None)
    parser.set_defaults(type_proj_dim=None)
    parser.add_argument(
        "--ignore_na_types",
        action="store_true",
        default=False,
        help="Ignore NA types in loss computation",
    )
    parser.set_defaults(collapse_na_types=False)
    parser.set_defaults(collapse_na_types_metrics=True)
    parser.set_defaults(combo_stats_max_samples=0)
    parser.set_defaults(combo_stats_entry_chunk=65536)
    parser.set_defaults(ambiguous_type_loss_mode="equal")
    parser.set_defaults(ambiguous_type_loss_lambda=1.0)
    parser.set_defaults(transformation_conditioning_mode="concat")
    parser.add_argument(
        "--transformation_base_rep_source", choices=["unembedding"], default="unembedding"
    )
    parser.set_defaults(transformation_normalize_base_rep=True)
    parser.set_defaults(no_transformation_normalize_base_rep=False)
    parser.set_defaults(transformation_project_base_rep=False)
    parser.set_defaults(transformation_base_rep_proj_dim=None)
    parser.set_defaults(log_prediction_analysis=False)
    parser.set_defaults(log_base_correct_extended_wrong=False)
    parser.set_defaults(no_log_base_correct_extended_wrong=True)
    parser.set_defaults(base_correct_sample_size=50)
    parser.set_defaults(base_correct_samples_per_group=10)
    parser.set_defaults(log_analysis_all_steps=False)
    parser.set_defaults(analysis_sample_size=10000)
    parser.set_defaults(analysis_sample_all=False)
    parser.set_defaults(combo_error_sample_size=50)
    parser.set_defaults(disable_transform_analysis=True)

    parsed_args = parser.parse_args()
    args = CompositionalHyperparameters()
    for key, value in vars(parsed_args).items():
        if value is not None or not hasattr(args, key):
            setattr(args, key, value)

    if getattr(args, "allow_no_space_prefix", False):
        args.skip_if_no_space_prefix = False
    if not (0.0 < args.throughput_window_frac <= 1.0):
        raise ValueError(
            f"throughput_window_frac must be in (0, 1], got {args.throughput_window_frac}"
        )
    if args.benchmark_regular_ce_memory and args.use_linear_cross_entropy:
        print("benchmark_regular_ce_memory enabled: forcing --no-use_linear_cross_entropy")
        args.use_linear_cross_entropy = False

    return args


def get_data_files(data_dir, tokenization_mode="single_id"):
    """Construct train and val file patterns from data directory.

    Args:
        data_dir: Data directory path
        tokenization_mode: "single_id" or "dual_stream"

    Returns:
        train_files, val_files: File patterns for glob
    """
    train_files = os.path.join(data_dir, "data_train_*.bin")
    val_files = os.path.join(data_dir, "data_val_*.bin")
    return train_files, val_files


def extract_dataset_name(dataset_path):
    """Extract dataset name from HuggingFace format (org/dataset) or simple name."""
    if "/" in dataset_path:
        # HuggingFace Hub format (e.g., "HuggingFaceFW/fineweb" -> "fineweb")
        name = dataset_path.split("/")[-1]
    else:
        # Simple name (e.g., "fineweb")
        name = dataset_path
    return name


def get_tokenizer_name(tokenizer_path):
    """Extract a short name from tokenizer path for run naming."""
    if "/" in tokenizer_path:
        # HuggingFace Hub format (e.g., "openai-community/gpt2-large")
        name = tokenizer_path.split("/")[-1]
    else:
        # Local path (e.g., "tokenizers/custom_bpe_10k")
        name = os.path.basename(tokenizer_path)
    # Replace hyphens with underscores for consistency
    name = name.replace("-", "_")
    return name


def get_run_name(args, model_type="compositional"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dataset_name = extract_dataset_name(args.dataset)
    tokenizer_name = get_tokenizer_name(args.base_tokenizer)

    # Build feature list for multi-token words
    features = []
    if not args.skip_multi_token_words:
        features.append("multi_token")
    if args.skip_three_token_words and not args.skip_multi_token_words:
        features.append("upto_two_tokens")
    elif getattr(args, "skip_four_token_words", False) and not args.skip_multi_token_words:
        features.append("upto_three_tokens")

    # Start with dataset name, tokenizer name, and model type
    run_name = f"{dataset_name}_{tokenizer_name}_{model_type}"

    # Add multi-token features if present
    if features:
        run_name += f"_{'_'.join(features)}"

    # Continue with existing naming pattern
    run_name += f"_lr{MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR:.4f}_cooldown{args.cooldown_frac:.2f}{args.lr_decay_type}_bs{world_size * args.train_seq_len}_d{args.model_dim}_ff{args.intermediate_dim}_l{args.num_layers}_h{args.num_heads}"
    run_name = run_name + "_bf16" if args.bf16 else run_name
    run_name = (
        run_name + f"_accum{args.gradient_accumulation_steps}steps"
        if args.gradient_accumulation_steps > 1
        else run_name
    )
    run_name = run_name + "_torchce" if not args.use_linear_cross_entropy else run_name
    run_name = run_name + "_membench" if args.benchmark_regular_ce_memory else run_name
    run_name = run_name + f"_loss_base{args.base_loss_alpha}_types{args.type_loss_alpha}"
    run_name = (
        run_name + f"_types_add_alpha{args.types_addition_alpha}"
        if args.types_addition_alpha != 1.0
        else run_name
    )
    run_name = run_name + "_mixing" if args.use_mixing_combination else run_name
    run_name = run_name + "_orthogonal_proj" if args.use_orthogonal_projections else run_name
    # Add transformation conditioning info
    if args.transformation_base_rep_source != "embedding":
        run_name += f"_rep{args.transformation_base_rep_source}"
    return run_name


def get_wandb_run_name(args):
    if args.wandb_run_name:
        return args.wandb_run_name

    return get_run_name(args)


def _normalize_group_ranges(types_loss_indices_map):
    normalized = {}
    for group_name, span in (types_loss_indices_map or {}).items():
        if span is None or len(span) != 2:
            continue
        normalized[group_name] = (int(span[0]), int(span[1]))
    return normalized


def _extract_type_indices(type_ids, total_types):
    if type_ids is None or isinstance(type_ids, str):
        return []
    if hasattr(type_ids, "tolist"):
        type_ids = type_ids.tolist()
    try:
        values = list(type_ids)
    except TypeError:
        return []
    if not values:
        return []

    # Backward compatibility for old one-hot bundle caches.
    is_one_hot = len(values) == total_types and all(
        (isinstance(val, (bool, int, np.integer)) and int(val) in (0, 1))
        or (isinstance(val, (float, np.floating)) and -1e-6 <= float(val) <= 1.0 + 1e-6)
        for val in values
    )
    if is_one_hot:
        return [idx for idx, val in enumerate(values) if float(val) > 0.5]

    indices = []
    for val in values:
        try:
            idx = int(val)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < total_types:
            indices.append(idx)
    return indices


def build_tokenizer_report(
    args,
    bundle,
    tokenizer,
    model_init_data,
    run_name,
    run_id=None,
    run_dir=None,
    transform_group_indices=None,
    unified_modifier_groups=None,
):
    transformation_names_to_int = model_init_data.get("transformation_names_to_int", {})
    type_groups = model_init_data.get("type_groups", {})
    types_loss_indices_map = _normalize_group_ranges(
        model_init_data.get("types_loss_indices_map", {})
    )
    final_decomposition_map = model_init_data.get("final_decomposition_map", {})

    idx_to_name = {}
    for name, idx in transformation_names_to_int.items():
        try:
            idx_to_name[int(idx)] = name
        except (TypeError, ValueError):
            continue

    group_details = []
    grouped_ids = set()
    for group_name, (start, end) in sorted(types_loss_indices_map.items(), key=lambda kv: kv[1][0]):
        transforms = []
        for idx in range(start, end):
            grouped_ids.add(idx)
            transforms.append(
                {
                    "id": idx,
                    "name": idx_to_name.get(idx, f"<missing:{idx}>"),
                }
            )
        group_details.append(
            {
                "group": group_name,
                "start_idx": start,
                "end_idx": end,
                "size": end - start,
                "transformations": transforms,
            }
        )

    ungrouped_transforms = []
    for idx, name in sorted(idx_to_name.items()):
        if idx not in grouped_ids:
            ungrouped_transforms.append({"id": idx, "name": name})

    total_types = len(transformation_names_to_int)
    used_transform_counts = {}
    for _, (_, type_ids) in final_decomposition_map.items():
        for idx in _extract_type_indices(type_ids, total_types):
            used_transform_counts[idx] = used_transform_counts.get(idx, 0) + 1

    used_transformations = [
        {"id": idx, "name": idx_to_name.get(idx, f"<missing:{idx}>"), "count": count}
        for idx, count in sorted(used_transform_counts.items(), key=lambda kv: kv[0])
    ]

    all_inflections = {int(x) for x in model_init_data.get("all_inflections", [])}
    in_vocab_inflections = {int(x) for x in model_init_data.get("inflections_single_token", [])}
    oov_inflections_by_id = sorted(all_inflections - in_vocab_inflections)

    decomposition_map = model_init_data.get("decomposition_map", {}) or {}
    syntactic_decomposition_map = model_init_data.get("syntactic_decomposition_map", {}) or {}
    n_variants_total = 0
    n_syntactic_variants_total = 0
    unique_variant_forms = set()
    for variants in decomposition_map.values():
        if isinstance(variants, dict):
            n_variants_total += len(variants)
            unique_variant_forms.update(str(form) for form in variants.keys())
    for variants in syntactic_decomposition_map.values():
        if isinstance(variants, dict):
            n_syntactic_variants_total += len(variants)

    tokenizer_vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    tokenizer_vocab_forms = set(tokenizer_vocab.keys())
    oov_variant_forms_count = 0
    oov_variant_examples = []
    for form in unique_variant_forms:
        if form not in tokenizer_vocab_forms:
            oov_variant_forms_count += 1
            if len(oov_variant_examples) < 100:
                oov_variant_examples.append(form)
    oov_variant_examples.sort()

    base_vocab_size = len(model_init_data.get("non_inflection_indices", []))
    extended_vocab_size = len(tokenizer)
    base_tokenizer_size = int(model_init_data.get("base_tokenizer_size", extended_vocab_size))
    saved_embeddings = extended_vocab_size - base_vocab_size
    saved_embeddings_pct = (saved_embeddings / extended_vocab_size) if extended_vocab_size else 0.0
    compression_ratio = (base_vocab_size / extended_vocab_size) if extended_vocab_size else None
    new_tokens_added_to_extended_vocab = max(0, extended_vocab_size - base_tokenizer_size)

    group_indices_payload = {}
    if transform_group_indices:
        for group_name, span in transform_group_indices.items():
            if span is None or len(span) != 2:
                continue
            group_indices_payload[group_name] = [int(span[0]), int(span[1])]

    report = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_name": run_name,
        "run_id": str(run_id) if run_id is not None else None,
        "run_dir": run_dir,
        "dataset": {
            "name": args.dataset,
            "config": args.dataset_config,
            "data_dir": args.data_dir,
        },
        "tokenization": {
            "mode": bundle.tokenization_mode,
            "base_tokenizer": args.base_tokenizer,
            "base_tokenizer_size": base_tokenizer_size,
            "extended_tokenizer_size": extended_vocab_size,
            "base_model_vocab_size": base_vocab_size,
            "new_tokens_added_to_extended_vocab": new_tokens_added_to_extended_vocab,
        },
        "vocab_diet": {
            "vocab_reduction_absolute": saved_embeddings,
            "vocab_reduction_relative": saved_embeddings_pct,
            "vocab_reduction_percent": 100.0 * saved_embeddings_pct,
            "saved_embeddings_count_vs_extended_vocab": saved_embeddings,
            "saved_embeddings_fraction_vs_extended_vocab": saved_embeddings_pct,
            "base_to_extended_vocab_ratio": compression_ratio,
        },
        "coverage": {
            "decomposition_base_words": len(decomposition_map),
            "decomposition_variants_total": n_variants_total,
            "decomposition_variants_unique": len(unique_variant_forms),
            "syntactic_variants_total": n_syntactic_variants_total,
            "single_token_decompositions": len(final_decomposition_map),
            "all_inflections_total": len(all_inflections),
            "in_vocab_inflections_total": len(in_vocab_inflections),
            "oov_inflections_expressible_by_id_total": len(oov_inflections_by_id),
            "oov_variant_forms_expressible_vs_tokenizer_vocab_total": oov_variant_forms_count,
            "oov_variant_form_examples": oov_variant_examples,
        },
        "transformations": {
            "num_transform_groups": len(types_loss_indices_map),
            "num_transformations_defined": len(transformation_names_to_int),
            "num_transformations_used_in_final_decomposition": len(used_transform_counts),
            "type_groups": {k: int(v) for k, v in type_groups.items()},
            "types_loss_indices_map": {
                group_name: [start, end]
                for group_name, (start, end) in types_loss_indices_map.items()
            },
            "groups": group_details,
            "used_transformations": used_transformations,
            "ungrouped_transformations": ungrouped_transforms,
            "unified_modifier_groups": list(unified_modifier_groups)
            if unified_modifier_groups
            else [],
            "dual_stream_transform_group_indices": group_indices_payload,
        },
        "bundle": {
            "cache_key": get_tokenizer_cache_path(args),
            "has_sequence_map": bundle.sequence_map is not None,
            "sequence_map_size": len(bundle.sequence_map) if bundle.sequence_map is not None else 0,
            "has_unified_modifier_array": bundle.unified_modifier_array is not None,
            "has_modifier_map": bundle.modifier_map is not None,
            "active_morphological_groups": list(
                model_init_data.get("active_morphological_groups") or []
            ),
        },
        "config_flags": {
            "morphology_mode": args.morphology_mode,
            "tokenization_mode": args.tokenization_mode,
            "drop_morphology_transforms": args.drop_morphology_transforms,
            "decompose_articles": args.decompose_articles,
            "decompose_prepositions": args.decompose_prepositions,
            "decompose_punctuation": args.decompose_punctuation,
            "decompose_article_prep_space_prefix": args.decompose_article_prep_space_prefix,
            "prune_punct_tokens": args.prune_punct_tokens,
            "possessive_separate_group": args.possessive_separate_group,
            "preposition_list": list(args.preposition_list) if args.preposition_list else [],
            "multitoken_allowed_groups": list(args.multitoken_allowed_groups)
            if args.multitoken_allowed_groups
            else [],
        },
    }
    return report


if __name__ == "__main__":
    args = parse_compositional_args()
    if args.wandb_project and wandb is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. Install wandb or pass --wandb_project ''."
        )

    # Construct base_data_dir if not provided
    if args.base_data_dir is None:
        dataset_name = extract_dataset_name(args.dataset)
        args.base_data_dir = f"data/vocab_diet/{dataset_name}_{args.dataset_config}/"

    # Set output dir and data dir
    setattr(args, "output_dir", os.path.join(args.output_dir, get_tokenizer_cache_path(args)))
    setattr(args, "data_dir", os.path.join(args.base_data_dir, get_tokenizer_cache_path(args)))

    # torchrun sets these env variables
    assert torch.cuda.is_available()

    # Check if we're in a distributed setting
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        # Multi-GPU distributed setup
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
        distributed = True
    else:
        # Single GPU setup
        world_size = 1
        rank = 0
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        distributed = False

    master_process = rank == 0

    # begin logging
    logfile = None
    run_id = None
    run_name = None
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)

    def print0(s, console=True):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    print0(code, console=False)
    print0("=" * 100, console=False)
    print0(f"Running Python {sys.version}", console=False)
    print0(
        f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}",
        console=False,
    )

    print0(nvidia_smi(), console=False)
    print0("=" * 100, console=False)

    # Get compositional tokenizer
    bundle = get_or_create_tokenizer_bundle(args)

    # Construct file paths based on tokenization mode
    train_files, val_files = get_data_files(args.data_dir, bundle.tokenization_mode)
    print0(f"Loading data files:")
    print0(f"  Train pattern: {train_files}")
    print0(f"  Val pattern: {val_files}")

    tokenizer = bundle.tokenizer
    model_init_data = bundle.model_init_data

    def _build_dual_stream_tokenizer_for_metrics():
        return None

    def _detect_type_ids_format(sample_items, total_types):
        for _, type_ids in sample_items:
            if type_ids is None or isinstance(type_ids, str):
                continue
            if hasattr(type_ids, "tolist"):
                type_ids = type_ids.tolist()
            try:
                size = len(type_ids)
            except TypeError:
                continue
            if size == 0:
                continue
            if size == total_types:
                is_one_hot = True
                for val in type_ids:
                    if isinstance(val, bool):
                        continue
                    if isinstance(val, int):
                        if val not in (0, 1):
                            is_one_hot = False
                            break
                    elif isinstance(val, float):
                        if val < -1e-6 or val > 1.0 + 1e-6:
                            is_one_hot = False
                            break
                    else:
                        is_one_hot = False
                        break
                return "one_hot" if is_one_hot else "indices"
            return "indices"
        return "unknown"

    def _normalize_base_vocab(model_init_data, tokenizer):
        base_token_indices = model_init_data.get("base_token_indices", [])
        token_id_to_base_id_mapping = model_init_data.get("token_id_to_base_id_mapping", {})
        non_inflection_indices = model_init_data.get("non_inflection_indices", [])
        unique_base_ids = sorted(
            set(token_id_to_base_id_mapping.values()) | set(base_token_indices)
        )
        if len(non_inflection_indices) != len(unique_base_ids):
            inv_vocab = {v: k for k, v in tokenizer.get_vocab().items()}
            non_inflection_set = set(non_inflection_indices)
            unique_base_set = set(unique_base_ids)
            missing_base_ids = sorted(unique_base_set - non_inflection_set)
            extra_non_inflection_ids = sorted(non_inflection_set - unique_base_set)

            print0(
                f"NOTE: base vocab mismatch detected in model_init_data:\n"
                f"  non_inflection_indices: {len(non_inflection_indices)}\n"
                f"  unique_base_ids: {len(unique_base_ids)}"
            )
            if missing_base_ids:
                print0("  base IDs present in mapping but missing from non_inflection_indices:")
                for base_id in missing_base_ids:
                    base_tok = inv_vocab.get(base_id, None)
                    base_tok_str = repr(base_tok) if base_tok is not None else "<unk_id>"
                    example_ext_ids = [
                        ext_id
                        for ext_id, b_id in token_id_to_base_id_mapping.items()
                        if b_id == base_id
                    ][:5]
                    example_ext_toks = [
                        repr(inv_vocab.get(eid, "<unk_id>")) for eid in example_ext_ids
                    ]
                    print0(
                        f"    base_id={base_id} token={base_tok_str} example_ext={list(zip(example_ext_ids, example_ext_toks))}"
                    )
            if extra_non_inflection_ids:
                print0("  base IDs present in non_inflection_indices but absent from mapping:")
                for base_id in extra_non_inflection_ids:
                    base_tok = inv_vocab.get(base_id, None)
                    base_tok_str = repr(base_tok) if base_tok is not None else "<unk_id>"
                    print0(f"    base_id={base_id} token={base_tok_str}")

            # Align base vocab sizing with actual base IDs used by the mapping.
            model_init_data["non_inflection_indices"] = unique_base_ids
            print0(
                f"NOTE: adjusted non_inflection_indices to match unique_base_ids "
                f"({len(unique_base_ids)} entries)."
            )

    # Debug: Check tokenizer vocabulary size
    print0(f"DEBUG: Tokenizer type: {type(tokenizer)}")
    print0(f"DEBUG: Tokenizer vocab size (len): {len(tokenizer)}")
    print0(
        f"DEBUG: Tokenizer vocab size (vocab): {len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 'N/A'}"
    )
    if hasattr(tokenizer, "vocab_size"):
        print0(f"DEBUG: Tokenizer.vocab_size attribute: {tokenizer.vocab_size}")

    # Verify tokenizer has required special tokens
    if tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer does not have an eos_token_id. "
            f"eos_token: {tokenizer.eos_token}, "
            f"bos_token: {tokenizer.bos_token}, "
            f"Base tokenizer: {args.base_tokenizer}"
        )

    # TODO add base vocab size here
    print0(vars(args), console=True)
    print0(f"Extended vocab size: {len(tokenizer)}")
    args.eos_token_id = tokenizer.eos_token_id
    print0(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print0(f"Transform groups: {model_init_data['transformation_names_to_int']}")
    _normalize_base_vocab(model_init_data, tokenizer)
    total_types = len(model_init_data.get("transformation_names_to_int", {}))
    final_decomp = model_init_data.get("final_decomposition_map", {})
    if final_decomp and total_types > 0:
        import itertools

        type_ids_format = _detect_type_ids_format(
            itertools.islice(final_decomp.values(), 5), total_types
        )
        print0(f"Type IDs format check: {type_ids_format} (total_types={total_types})")
        if type_ids_format == "one_hot":
            print0(
                "WARNING: final_decomposition_map appears to use one-hot vectors. "
                "Rebuild the tokenizer bundle cache to use list-of-indices."
            )

    # Create type config (pass types_loss_indices_map for correct index ordering)
    type_config = TypeConfig(
        model_init_data["type_groups"],
        transformation_names_to_int=model_init_data["transformation_names_to_int"],
        types_loss_indices_map=model_init_data["types_loss_indices_map"],
    )

    transform_analysis_config = None
    modifier_map_dict = None
    new_transform_group_indices = None
    unified_modifier_groups = None

    # Print vocab stats
    print0(
        f"Vocab stats - Base words: {len(model_init_data['decomposition_map'])}, Inflections: {len(model_init_data['all_inflections'])}, In-vocab inflections: {len(model_init_data['inflections_single_token'])}, Non inflections: {len(model_init_data['non_inflection_indices'])}"
    )
    print0(f"Transformation names to ID: {model_init_data['transformation_names_to_int']}")

    # Initialize model
    model: nn.Module = CompositionalGPT(
        vocab_size=len(model_init_data["non_inflection_indices"]),
        extended_vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        type_config=type_config,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        model_dim=args.model_dim,
        intermediate_dim=args.intermediate_dim,
        num_kv_heads=args.num_kv_heads,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
        use_gated_proj=True,
        use_rms_norm=False,
        use_gqa=True,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        rope_scaling_type="linear",
        reorder_norms=args.reorder_norms,
        init_strategy=args.init_strategy,
        init_std=args.init_std,
        base_dim=args.base_dim,
        base_loss_alpha=args.base_loss_alpha,
        type_loss_alpha=args.type_loss_alpha,
        use_shift=args.use_shift,
        softcap=args.softcap,
        types_addition_alpha=args.types_addition_alpha,
        ignore_na_types=args.ignore_na_types,
        collapse_na_types=args.collapse_na_types,
        ambiguous_type_loss_mode=args.ambiguous_type_loss_mode,
        ambiguous_type_loss_lambda=args.ambiguous_type_loss_lambda,
        transformation_conditioning_mode=args.transformation_conditioning_mode,
        transformation_base_rep_source=args.transformation_base_rep_source,
        transformation_normalize_base_rep=args.transformation_normalize_base_rep,
        transformation_project_base_rep=args.transformation_project_base_rep,
        transformation_base_rep_proj_dim=args.transformation_base_rep_proj_dim,
        modifier_map_dict=modifier_map_dict,
        new_transform_group_indices=new_transform_group_indices,
        unified_modifier_groups=unified_modifier_groups,
        use_linear_cross_entropy=args.use_linear_cross_entropy,
    ).cuda()

    if len(model_init_data["na_type_ids"]) > 0:
        model.freeze_rows_in_type_embeddings(
            model_init_data["na_type_ids"], freeze_embed=True, freeze_head=False
        )
    model.set_token_to_type_ids(
        model_init_data["final_decomposition_map"], model_init_data["na_type_ids"]
    )
    model.set_token_mappings(
        model_init_data["base_token_indices"],
        model_init_data["token_id_to_base_id_mapping"],
        model_init_data["base_or_inflection_token_ids"],
    )

    # Setup auxiliary LM head mappings if enabled
    if model.aux_lm_head is not None:
        model.aux_lm_head.setup_mappings(
            model_init_data["base_token_indices"], model_init_data["token_id_to_base_id_mapping"]
        )

    model.lm_head.bfloat16()
    if model.aux_lm_head is not None:
        model.aux_lm_head.bfloat16()
    for m in model.modules():
        if args.bf16:
            m.bfloat16()
        elif isinstance(m, nn.Embedding):
            m.bfloat16()
    if distributed:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
    total_parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(
        f"Model parameter count (start): total={total_parameter_count:,} trainable={trainable_parameter_count:,}",
        console=True,
    )
    # Initialize wandb
    if master_process and args.wandb_project:
        model_hparams = {
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "model_dim": model.model_dim,
            "intermediate_dim": model.intermediate_dim,
            "num_kv_heads": model.num_kv_heads,
            "max_seq_len": model.max_seq_len,
            "use_gated_proj": model.use_gated_proj,
            "use_rms_norm": model.use_rms_norm,
            "use_gqa": model.use_gqa,
            "extended_vocab_size": model.extended_vocab_size,
            "base_loss_alpha": model.base_loss_alpha,
            "type_loss_alpha": model.type_loss_alpha,
            "use_linear_cross_entropy": model.use_linear_cross_entropy,
        }

        wandb_config = vars(args).copy()
        wandb_config.update(model_hparams)
        # Add dataset info for easy tracking
        wandb_config["dataset_name"] = extract_dataset_name(args.dataset)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=get_wandb_run_name(args),
            config=wandb_config,
        )

    if master_process:
        run_name = get_run_name(args)
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        tokenizer_report = build_tokenizer_report(
            args=args,
            bundle=bundle,
            tokenizer=tokenizer,
            model_init_data=model_init_data,
            run_name=run_name,
            run_id=run_id,
            run_dir=run_dir,
            transform_group_indices=new_transform_group_indices,
            unified_modifier_groups=unified_modifier_groups,
        )
        with open(os.path.join(run_dir, "tokenizer_report.json"), "w") as f:
            json.dump(tokenizer_report, f, indent=2)

        vocab_stats = {
            "n_base_words": len(model_init_data["decomposition_map"]),
            "n_inflections": len(model_init_data["all_inflections"]),
            "n_in_vocab_inflections": len(model_init_data["inflections_single_token"]),
            "vocab_reduction_absolute": tokenizer_report["vocab_diet"]["vocab_reduction_absolute"],
            "vocab_reduction_relative": tokenizer_report["vocab_diet"]["vocab_reduction_relative"],
            "vocab_reduction_percent": tokenizer_report["vocab_diet"]["vocab_reduction_percent"],
        }
        with open(os.path.join(run_dir, "vocab_stats.json"), "w") as f:
            json.dump(vocab_stats, f, indent=2)
        print0(f"Saved tokenizer reports: {os.path.join(run_dir, 'tokenizer_report.json')}")

        # Clean up analysis directory for fresh run

        # Initialize logging utilities
        if args.disable_local_logging:
            loss_logger = logits_stats_collector = None
        else:
            loss_logger = LossLogger(os.path.join(run_dir, "loss_logs"), master_process)
            logits_stats_collector = LogitsStatsCollector(
                os.path.join(run_dir, "logits_stats"), master_process
            )
            run_tracker = RunTracker("runs_log.csv")
            run_tracker.log_run(args, "compositional", run_id, run_dir, logfile)
    else:
        loss_logger = logits_stats_collector = None

    # Collect parameters for optimization
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [
        model.embed.weight,
    ]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [
        model.lm_head.weight,
    ]
    type_embed_params = [model.embed_types.weight]
    type_head_params = []

    # Add conditioned transformation head parameters
    if hasattr(model.conditioned_transformation_head, "projection"):
        # Concat or add mode: single projection layer
        type_head_params.append(model.conditioned_transformation_head.projection.weight)
    elif hasattr(model.conditioned_transformation_head, "output_proj"):
        # Cross-attention mode: multiple projection layers
        type_head_params.append(model.conditioned_transformation_head.query_proj.weight)
        type_head_params.append(model.conditioned_transformation_head.key_proj.weight)
        type_head_params.append(model.conditioned_transformation_head.value_proj.weight)
        type_head_params.append(model.conditioned_transformation_head.output_proj.weight)

    # Add optional base representation projection
    if (
        hasattr(model.conditioned_transformation_head, "base_rep_proj")
        and model.conditioned_transformation_head.base_rep_proj is not None
    ):
        type_head_params.append(model.conditioned_transformation_head.base_rep_proj.weight)

    # Add auxiliary head parameters if enabled
    if model.aux_lm_head is not None and model.aux_lm_head.aux_weight is not None:
        head_params += [model.aux_lm_head.aux_weight]

    optimizer1 = DistAdam(
        scalar_params + embed_params + type_embed_params,
        lr=args.model_size_lr * args.base_embed_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
        weight_decay=args.weight_decay,
    )
    optimizer1b = DistAdam(
        head_params + type_head_params,
        lr=args.model_size_lr * args.base_lm_head_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
        weight_decay=args.weight_decay,
    )
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=args.model_size_lr * args.base_hidden_lr,
        momentum=args.muon_momentum,
        weight_decay=args.weight_decay,
    )

    optimizers = [optimizer1, optimizer1b, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from_checkpoint:
        print0(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        start_step = load_checkpoint(args.resume_from_checkpoint, model, optimizers)
        print0(f"Resumed from step {start_step}")

    # Learning rate schedule
    def get_lr(step: int):
        x = step / args.num_iterations
        assert 0 <= x <= 1
        if args.lr_warmup_steps > 0 and (x < (args.lr_warmup_steps / args.num_iterations)):
            w = 1 - (x / (args.lr_warmup_steps / args.num_iterations))
            return w * 0.1 + (1 - w) * 1.0
        elif x < 1 - args.cooldown_frac:
            return 1.0
        else:
            decay_progress = (x - (1 - args.cooldown_frac)) / args.cooldown_frac
            if args.lr_decay_type == "cosine":
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:
                return 1.0 - 0.9 * decay_progress

    # Attention window size schedule
    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

    def get_window_size_blocks(step: int):
        x = step / args.num_iterations
        assert 0 <= x <= 1
        window_size = next_multiple_of_n(1728 * x, n=128)
        return get_window_size_blocks_helper(window_size)

    if not args.dont_compile:
        model: nn.Module = torch.compile(
            model, dynamic=False
        )  # we need dynamic for this architecture
    hf_model = None

    # Warmup kernels
    warmup_steps = 10
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
    )
    train_loader = distributed_data_generator(
        train_files,
        world_size * args.train_seq_len,
        align_to_bos=True,
        dtype=torch.int32 if args.int32_data else torch.uint16,
        bos_token_id=tokenizer.eos_token_id,
        bundle=bundle,
    )
    for _ in range(warmup_steps):
        batch = next(train_loader)
        inputs, targets = batch
        loss_output = model(inputs, targets, get_window_size_blocks(1))
        loss_output.loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del train_loader, initial_state

    # Training and validation
    train_loader = distributed_data_generator(
        train_files,
        world_size * args.train_seq_len,
        align_to_bos=True,
        dtype=torch.int32 if args.int32_data else torch.uint16,
        bos_token_id=tokenizer.eos_token_id,
        bundle=bundle,
    )

    # Setup validation tokenization stats for BPB metrics
    avg_bytes_per_token = None
    token_fertility = None
    dual_stream_tokenizer = _build_dual_stream_tokenizer_for_metrics()
    if args.val_loss_every > 0:
        # Compute total bytes and word counts in validation set once (distributed across all processes)
        if master_process:
            print0("Computing validation tokenization metrics...")
        val_batch_size = world_size * args.val_seq_len
        val_steps = args.val_tokens // val_batch_size
        val_loader_for_bytes = distributed_data_generator(
            val_files,
            val_batch_size,
            align_to_bos=False,
            dtype=torch.int32 if args.int32_data else torch.uint16,
            bos_token_id=tokenizer.eos_token_id,
            bundle=bundle,
        )
        local_bytes, local_words = compute_validation_tokenization_stats(
            tokenizer,
            val_loader_for_bytes,
            val_steps,
            dual_stream_tokenizer=dual_stream_tokenizer,
        )
        del val_loader_for_bytes

        # Sum bytes and words across all processes
        total_val_bytes_tensor = torch.tensor(local_bytes, device="cuda", dtype=torch.float64)
        total_val_words_tensor = torch.tensor(local_words, device="cuda", dtype=torch.float64)
        if distributed:
            dist.all_reduce(total_val_bytes_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_words_tensor, op=dist.ReduceOp.SUM)
        total_val_bytes = total_val_bytes_tensor.item()
        total_val_words = total_val_words_tensor.item()
        avg_bytes_per_token = total_val_bytes / args.val_tokens
        token_fertility = (args.val_tokens / total_val_words) if total_val_words > 0 else 0.0

        if master_process:
            print0(
                f"Total validation bytes: {total_val_bytes}, avg bytes per token: {avg_bytes_per_token:.2f}, "
                f"token fertility: {token_fertility:.4f}"
            )

    training_time_ms = 0
    train_tokens_per_iteration = world_size * args.train_seq_len * args.gradient_accumulation_steps
    throughput_window_steps = max(
        1, int(math.ceil(args.num_iterations * args.throughput_window_frac))
    )
    throughput_window_start_step = max(start_step, args.num_iterations - throughput_window_steps)
    throughput_window_start_time_ms = None
    if master_process:
        print0(
            f"Throughput window: last {throughput_window_steps}/{args.num_iterations} iterations "
            f"(frac={args.throughput_window_frac:.2f}, start_step={throughput_window_start_step})",
            console=True,
        )
    memory_benchmark_active = bool(
        args.benchmark_regular_ce_memory and not args.use_linear_cross_entropy
    )
    memory_benchmark_summary = {
        "steps": 0,
        "max_peak_alloc_mb": 0.0,
        "max_peak_reserved_mb": 0.0,
        "sum_peak_alloc_mb": 0.0,
        "sum_peak_reserved_mb": 0.0,
    }
    if memory_benchmark_active and master_process:
        print0("GPU memory benchmark enabled (regular torch cross entropy)", console=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    train_steps = args.num_iterations
    for step in range(start_step, train_steps + 1):
        last_step = step == train_steps

        # VALIDATION SECTION
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0):
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(
                val_files,
                val_batch_size,
                align_to_bos=False,
                dtype=torch.int32 if args.int32_data else torch.uint16,
                bos_token_id=tokenizer.eos_token_id,
                bundle=bundle,
            )
            val_loss = 0
            val_type_loss = 0
            has_oracle_type_loss = False
            val_type_loss_pred_base = 0
            has_pred_base_type_loss = False
            val_aux_loss = 0
            val_accuracies = {"extended_accuracy": 0.0, "base_accuracy": 0.0, "aux_accuracy": 0.0}
            logits_for_stats = None
            prediction_analysis_data = []
            base_correct_extended_wrong_data = []
            base_correct_extended_wrong_by_group = {}
            base_correct_extended_wrong_group_counts = {}
            comprehensive_stats_list = []
            combo_error_samples = []
            transform_accuracy = None
            combo_resolution_stats = None
            pass

            # Start validation timing
            val_start_time = time.time()

            def _update_base_correct_group_samples(entries):
                if not base_correct_extended_wrong_by_group:
                    return
                for entry in entries:
                    group_kinds = entry.get("group_error_kinds", {})
                    for group_name, kind in group_kinds.items():
                        if kind not in ("spurious", "omitted", "wrong"):
                            continue
                        base_correct_extended_wrong_group_counts[group_name] = (
                            base_correct_extended_wrong_group_counts.get(group_name, 0) + 1
                        )
                        if (
                            args.base_correct_samples_per_group
                            and len(base_correct_extended_wrong_by_group[group_name])
                            >= args.base_correct_samples_per_group
                        ):
                            continue
                        base_correct_extended_wrong_by_group[group_name].append(entry)

            with torch.no_grad():
                for i in range(val_steps):
                    batch = next(val_loader)
                    inputs, targets = batch
                    input_modifiers, target_modifiers = None, None

                    # Determine if we should compute auxiliary logits for analysis
                    should_log_this_step = args.log_analysis_all_steps or last_step
                    compute_aux_logits = model.aux_lm_head is not None

                    loss_output = model(
                        inputs,
                        targets,
                        get_window_size_blocks(step),
                        compute_aux_logits=compute_aux_logits,
                        modifier_ids=input_modifiers,
                        target_modifier_ids=target_modifiers,
                    )
                    val_loss += loss_output.base_loss
                    if loss_output.type_logits is not None:
                        shift_amount = 1 if model.use_shift else 0
                        if shift_amount > 0:
                            target_old_type_ids_for_oracle = model.token_to_type_ids[
                                targets[shift_amount:]
                            ]
                            if target_modifiers is not None:
                                target_new_type_ids_for_oracle = model.map_modifier_ids_to_type_ids(
                                    target_modifiers[shift_amount:]
                                )
                                type_labels_for_oracle = model.merge_type_ids(
                                    target_old_type_ids_for_oracle, target_new_type_ids_for_oracle
                                )
                            else:
                                type_labels_for_oracle = target_old_type_ids_for_oracle
                            pred_type_logits_for_oracle = loss_output.type_logits[:-shift_amount]
                        else:
                            target_old_type_ids_for_oracle = model.token_to_type_ids[targets]
                            if target_modifiers is not None:
                                target_new_type_ids_for_oracle = model.map_modifier_ids_to_type_ids(
                                    target_modifiers
                                )
                                type_labels_for_oracle = model.merge_type_ids(
                                    target_old_type_ids_for_oracle, target_new_type_ids_for_oracle
                                )
                            else:
                                type_labels_for_oracle = target_old_type_ids_for_oracle
                            pred_type_logits_for_oracle = loss_output.type_logits
                        val_type_loss += model.compute_type_losses(
                            type_labels_for_oracle, pred_type_logits_for_oracle
                        )
                        has_oracle_type_loss = True
                    if loss_output.type_logits_pred_base is not None:
                        shift_amount = 1 if model.use_shift else 0
                        if shift_amount > 0:
                            target_old_type_ids_for_pred = model.token_to_type_ids[
                                targets[shift_amount:]
                            ]
                            if target_modifiers is not None:
                                target_new_type_ids_for_pred = model.map_modifier_ids_to_type_ids(
                                    target_modifiers[shift_amount:]
                                )
                                type_labels_for_pred = model.merge_type_ids(
                                    target_old_type_ids_for_pred, target_new_type_ids_for_pred
                                )
                            else:
                                type_labels_for_pred = target_old_type_ids_for_pred
                            pred_type_logits_for_pred = loss_output.type_logits_pred_base[
                                :-shift_amount
                            ]
                        else:
                            target_old_type_ids_for_pred = model.token_to_type_ids[targets]
                            if target_modifiers is not None:
                                target_new_type_ids_for_pred = model.map_modifier_ids_to_type_ids(
                                    target_modifiers
                                )
                                type_labels_for_pred = model.merge_type_ids(
                                    target_old_type_ids_for_pred, target_new_type_ids_for_pred
                                )
                            else:
                                type_labels_for_pred = target_old_type_ids_for_pred
                            pred_type_logits_for_pred = loss_output.type_logits_pred_base
                        val_type_loss_pred_base += model.compute_type_losses(
                            type_labels_for_pred, pred_type_logits_for_pred
                        )
                        has_pred_base_type_loss = True
                    if loss_output.aux_loss is not None:
                        val_aux_loss += loss_output.aux_loss
                    # Compute top-1 accuracies (if model outputs available)
                    with torch.no_grad():
                        base_targets = model.map_inputs_to_base_ids(targets)
                        accs = compute_top1_accuracies(
                            model,
                            loss_output,
                            extended_targets=targets,
                            base_targets=base_targets,
                            target_modifier_ids=target_modifiers,
                            collapse_na_types_metrics=(
                                args.collapse_na_types_metrics or args.collapse_na_types
                            ),
                        )
                        for k, v in accs.items():
                            val_accuracies[k] = val_accuracies.get(k, 0.0) + v

                    # Transformation accuracy analysis
                    if transform_accuracy and loss_output.type_logits is not None:
                        target_old_type_ids = model.token_to_type_ids[targets]
                        if target_modifiers is not None:
                            target_new_type_ids = model.map_modifier_ids_to_type_ids(
                                target_modifiers
                            )
                            type_labels = model.merge_type_ids(
                                target_old_type_ids, target_new_type_ids
                            )
                        else:
                            type_labels = target_old_type_ids
                        base_preds = (
                            loss_output.base_logits.argmax(dim=-1)
                            if loss_output.base_logits is not None
                            else None
                        )
                        valid_mask = targets != -1
                        transform_accuracy.update(
                            loss_output.type_logits,
                            type_labels,
                            base_preds,
                            base_targets,
                            valid_mask,
                        )
                        if combo_resolution_stats is not None and base_preds is not None:
                            batch_stats = model.conditioned_transformation_head.compute_combo_resolution_stats(
                                base_preds,
                                loss_output.type_logits,
                                valid_mask=valid_mask,
                                max_samples=args.combo_stats_max_samples,
                                entry_chunk_size=args.combo_stats_entry_chunk,
                                collapse_na_types_override=(
                                    args.collapse_na_types_metrics or args.collapse_na_types
                                ),
                                return_samples=bool(
                                    master_process and args.combo_error_sample_size
                                ),
                                max_error_samples=args.combo_error_sample_size,
                                context_ids=targets,
                                context_modifiers=target_modifiers,
                                target_base_ids=base_targets,
                            )
                            if master_process and batch_stats.get("error_samples"):
                                remaining = max(
                                    0, args.combo_error_sample_size - len(combo_error_samples)
                                )
                                if remaining > 0:
                                    combo_error_samples.extend(
                                        batch_stats["error_samples"][:remaining]
                                    )
                                batch_stats.pop("error_samples", None)
                            merge_combo_resolution_stats(combo_resolution_stats, batch_stats)

                    # Collect prediction analysis data
            val_loss /= val_steps
            val_aux_loss /= val_steps
            if has_oracle_type_loss:
                val_type_loss /= val_steps
            else:
                val_type_loss = None
            if has_pred_base_type_loss:
                val_type_loss_pred_base /= val_steps
            else:
                val_type_loss_pred_base = None
            for k in val_accuracies:
                val_accuracies[k] /= val_steps

            del val_loader

            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                if val_type_loss is not None:
                    dist.all_reduce(val_type_loss, op=dist.ReduceOp.AVG)
                if val_type_loss_pred_base is not None:
                    dist.all_reduce(val_type_loss_pred_base, op=dist.ReduceOp.AVG)
                if loss_output.aux_loss is not None:
                    dist.all_reduce(val_aux_loss, op=dist.ReduceOp.AVG)

                for k in val_accuracies:
                    tensor = torch.tensor(val_accuracies[k], device=val_loss.device)
                    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                    val_accuracies[k] = tensor.item()

                if transform_accuracy:
                    transform_accuracy.all_reduce()

            # Calculate validation timing
            val_end_time = time.time()
            val_time_ms = (val_end_time - val_start_time) * 1000

            current_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
            current_embed_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_EMBED_LR
            train_steps_completed = max(step - start_step, 0)
            train_tokens_seen = train_steps_completed * train_tokens_per_iteration
            global_steps_completed = start_step + train_steps_completed
            if (
                throughput_window_start_time_ms is None
                and global_steps_completed >= throughput_window_start_step
            ):
                throughput_window_start_time_ms = training_time_ms
                if global_steps_completed > throughput_window_start_step:
                    throughput_window_start_step = global_steps_completed
            throughput_window_steps_completed = max(
                0, global_steps_completed - throughput_window_start_step
            )
            throughput_window_time_ms = (
                0.0
                if throughput_window_start_time_ms is None
                else max(0.0, training_time_ms - throughput_window_start_time_ms)
            )
            train_tokens_per_sec = 0.0
            if throughput_window_steps_completed > 0 and throughput_window_time_ms > 0:
                train_tokens_per_sec = (
                    throughput_window_steps_completed * train_tokens_per_iteration
                ) / (throughput_window_time_ms / 1000.0)

            byte_metrics = None
            byte_metrics_pred_base = None
            loss_for_bpb = None
            loss_for_bpb_pred_base = None
            transform_summary = None
            # Prepare validation log message
            val_msg = f"***** VALIDATION: step:{step}/{train_steps} val_loss:{val_loss:.4f} val_aux_loss:{val_aux_loss:.4f}"
            if val_type_loss is not None:
                val_msg += f" val_type_loss:{val_type_loss:.4f}"
            if val_type_loss_pred_base is not None:
                val_msg += f" val_type_loss_pred_base:{val_type_loss_pred_base:.4f}"
            if avg_bytes_per_token is not None:
                num_groups = len(model.type_config.type_groups)
                factorized_metrics = compute_factorized_bytes_per_token_metrics(
                    val_loss, val_type_loss, num_groups, avg_bytes_per_token
                )
                loss_for_bpb = factorized_metrics["joint_nll"]
                byte_metrics = {"bpb": factorized_metrics["bpb"]}
                val_msg += f" bpb:{byte_metrics['bpb']:.4f}"
                val_msg += f" bpb_oracle_base:{byte_metrics['bpb']:.4f}"
                if val_type_loss_pred_base is not None:
                    pred_base_metrics = compute_factorized_bytes_per_token_metrics(
                        val_loss, val_type_loss_pred_base, num_groups, avg_bytes_per_token
                    )
                    loss_for_bpb_pred_base = pred_base_metrics["joint_nll"]
                    byte_metrics_pred_base = {"bpb": pred_base_metrics["bpb"]}
                    val_msg += f" bpb_pred_base:{byte_metrics_pred_base['bpb']:.4f}"
            for k, v in val_accuracies.items():
                val_msg += f" {k}:{v:.4f}"
            if transform_accuracy:
                transform_summary = transform_accuracy.to_summary()
                combo_acc = transform_summary.get("combo_accuracy", 0.0)
                combo_acc_base = transform_summary.get("combo_accuracy_base_correct", 0.0)
                transform_summary["next_token_accuracy"] = {
                    "extended": val_accuracies.get("extended_accuracy", 0.0),
                    "base": val_accuracies.get("base_accuracy", 0.0),
                }
                confusion_payload = transform_accuracy.get_confusion_payload()
                if dist.is_available() and dist.is_initialized():
                    if hasattr(dist, "gather_object"):
                        gathered = (
                            [None for _ in range(dist.get_world_size())] if master_process else None
                        )
                        dist.gather_object(confusion_payload, gathered, dst=0)
                        if master_process and gathered is not None:
                            transform_summary["group_confusion"] = merge_confusion_payloads(
                                gathered
                            )
                    elif master_process:
                        transform_summary["group_confusion"] = confusion_payload
                else:
                    transform_summary["group_confusion"] = confusion_payload
                if combo_resolution_stats is not None:
                    combo_payload = combo_resolution_stats
                    if (
                        dist.is_available()
                        and dist.is_initialized()
                        and hasattr(dist, "gather_object")
                    ):
                        gathered = (
                            [None for _ in range(dist.get_world_size())] if master_process else None
                        )
                        dist.gather_object(combo_payload, gathered, dst=0)
                        if master_process and gathered is not None:
                            merged = init_combo_resolution_stats(
                                list(model.type_config.type_groups.keys())
                            )
                            for item in gathered:
                                if item:
                                    merge_combo_resolution_stats(merged, item)
                            transform_summary["combo_resolution"] = finalize_combo_resolution_stats(
                                merged
                            )
                    elif master_process:
                        transform_summary["combo_resolution"] = finalize_combo_resolution_stats(
                            combo_payload
                        )
                    else:
                        transform_summary["combo_resolution"] = finalize_combo_resolution_stats(
                            combo_payload
                        )
                val_msg += f" type_combo:{combo_acc:.4f} type_combo_base:{combo_acc_base:.4f}"
            val_msg += f" train_time:{training_time_ms:.0f}ms val_time:{val_time_ms:.0f}ms step_avg:{training_time_ms / max(step - start_step, 1):.2f}ms tok_s:{train_tokens_per_sec:.1f} lr:{current_lr:.6f}_embed{current_embed_lr:.6f}"

            print0(val_msg, console=True)

            if master_process:
                # Save prediction analysis data if available
                pass

                # Log validation loss to file
                if loss_logger:
                    additional_losses = {
                        "val_aux_loss": val_aux_loss,
                        "train_tokens_seen": train_tokens_seen,
                        "train_tokens_per_sec": train_tokens_per_sec,
                    }
                    if val_type_loss is not None:
                        additional_losses["val_type_loss"] = val_type_loss
                    if val_type_loss_pred_base is not None:
                        additional_losses["val_type_loss_pred_base"] = val_type_loss_pred_base
                    for k, v in val_accuracies.items():
                        additional_losses[f"val_{k}"] = v
                    if byte_metrics is not None:
                        additional_losses["bpb"] = byte_metrics["bpb"]
                        additional_losses["bpb_oracle_base"] = byte_metrics["bpb"]
                        if val_type_loss is not None:
                            additional_losses["val_joint_nll"] = loss_for_bpb
                            additional_losses["joint_nll_oracle_base"] = loss_for_bpb
                    if byte_metrics_pred_base is not None:
                        additional_losses["bpb_pred_base"] = byte_metrics_pred_base["bpb"]
                        additional_losses["val_joint_nll_pred_base"] = loss_for_bpb_pred_base
                        additional_losses["joint_nll_pred_base"] = loss_for_bpb_pred_base
                    if avg_bytes_per_token is not None:
                        additional_losses["avg_bytes_per_token"] = avg_bytes_per_token
                    if token_fertility is not None:
                        additional_losses["token_fertility"] = token_fertility
                    if transform_accuracy:
                        additional_losses["val_type_combo_accuracy"] = combo_acc
                        additional_losses["val_type_combo_accuracy_base"] = combo_acc_base
                    loss_logger.log_val_loss(step, val_loss, training_time_ms, additional_losses)

                # Collect and log logits statistics
                logits_wandb_logs = {}
                if logits_stats_collector and logits_for_stats:
                    logits_stats = logits_stats_collector.collect_stats(logits_for_stats, step)
                    logits_wandb_logs = logits_stats_collector.get_wandb_logs(logits_stats)

                # Log to wandb
                if args.wandb_project:
                    wandb_logs = {
                        "val_loss": val_loss,
                        "val_aux_loss": val_aux_loss,
                        "lr": current_lr,
                        "step": step,
                        "train_tokens_seen": train_tokens_seen,
                        "train_tokens_per_sec": train_tokens_per_sec,
                    }
                    wandb_logs.update(logits_wandb_logs)

                    # Add accuracy metrics
                    for k, v in val_accuracies.items():
                        wandb_logs[f"val_{k}"] = v
                    if val_type_loss is not None:
                        wandb_logs["val_type_loss"] = val_type_loss
                    if val_type_loss_pred_base is not None:
                        wandb_logs["val_type_loss_pred_base"] = val_type_loss_pred_base

                    # Add bytes-per-token metrics if available
                    if byte_metrics is not None:
                        wandb_logs.update(byte_metrics)
                        wandb_logs["val_bpb"] = byte_metrics["bpb"]
                        wandb_logs["val_bpb_oracle_base"] = byte_metrics["bpb"]
                        if val_type_loss is not None:
                            wandb_logs["val_joint_nll"] = loss_for_bpb
                            wandb_logs["val_joint_nll_oracle_base"] = loss_for_bpb
                    if byte_metrics_pred_base is not None:
                        wandb_logs["val_bpb_pred_base"] = byte_metrics_pred_base["bpb"]
                        wandb_logs["val_joint_nll_pred_base"] = loss_for_bpb_pred_base
                    if avg_bytes_per_token is not None:
                        wandb_logs["avg_bytes_per_token"] = avg_bytes_per_token
                    if token_fertility is not None:
                        wandb_logs["token_fertility"] = token_fertility

                    # Add GPU utilization metrics
                    gpu_stats = get_combined_gpu_stats()
                    wandb_logs.update(gpu_stats)

                    wandb.log(wandb_logs)

            model.train()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Save intermediate checkpoints
        if (
            args.save_checkpoint_every > 0
            and step > 0
            and step % args.save_checkpoint_every == 0
            and not last_step
        ):
            current_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
            save_checkpoint(step, model, optimizers, args, run_name, current_lr)

        if last_step:
            if args.save_final_checkpoint:
                current_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
                save_checkpoint(step, model, optimizers, args, run_name, current_lr)
                tokenizer.save_pretrained(os.path.join(args.output_dir, run_name, "tokenizer"))
            break

        # TRAINING SECTION
        train_loss = 0
        vocab_loss = 0
        base_loss = 0
        type_loss = 0
        aux_loss = 0
        if memory_benchmark_active:
            torch.cuda.reset_peak_memory_stats()
        for micro_step in range(args.gradient_accumulation_steps):
            batch = next(train_loader)
            inputs, targets = batch
            input_modifiers, target_modifiers = None, None
            with torch.amp.autocast("cuda", enabled=False):
                loss_output = model(
                    inputs,
                    targets,
                    get_window_size_blocks(step),
                    modifier_ids=input_modifiers,
                    target_modifier_ids=target_modifiers,
                )
                loss = loss_output.loss / args.gradient_accumulation_steps
                base_loss += (
                    loss_output.base_loss.item() / args.gradient_accumulation_steps
                    if loss_output.base_loss
                    else 0.0
                )
                type_loss += (
                    loss_output.type_loss.item() / args.gradient_accumulation_steps
                    if loss_output.type_loss
                    else 0.0
                )
                aux_loss += (
                    loss_output.aux_loss.item() / args.gradient_accumulation_steps
                    if loss_output.aux_loss
                    else 0.0
                )
            loss.backward()
            train_loss += loss.item()

        # Set optimization hyperparameters
        current_lr = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * current_lr
        if args.momentum_warmup_steps > 0:
            frac = min(step / args.momentum_warmup_steps, 1)
            for group in optimizer2.param_groups:
                group["momentum"] = (1 - frac) * (
                    args.muon_momentum - args.momentum_warmup_gap
                ) + frac * args.muon_momentum
            for group in optimizer1.param_groups:
                group["betas"] = (
                    (1 - frac) * (args.adam_beta1 - args.momentum_warmup_gap)
                    + frac * args.adam_beta1,
                    args.adam_beta2,
                )
            for group in optimizer1b.param_groups:
                group["betas"] = (
                    (1 - frac) * (args.adam_beta1 - args.momentum_warmup_gap)
                    + frac * args.adam_beta1,
                    args.adam_beta2,
                )

        # gradient clipping
        if args.clip_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

        # Step optimizers
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        current_learning_rate = current_lr * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        train_steps_completed = step + 1 - start_step
        train_tokens_seen = train_steps_completed * train_tokens_per_iteration
        global_steps_completed = start_step + train_steps_completed
        if (
            throughput_window_start_time_ms is None
            and global_steps_completed >= throughput_window_start_step
        ):
            throughput_window_start_time_ms = approx_training_time_ms
            if global_steps_completed > throughput_window_start_step:
                throughput_window_start_step = global_steps_completed
        throughput_window_steps_completed = max(
            0, global_steps_completed - throughput_window_start_step
        )
        throughput_window_time_ms = (
            0.0
            if throughput_window_start_time_ms is None
            else max(0.0, approx_training_time_ms - throughput_window_start_time_ms)
        )
        train_tokens_per_sec = 0.0
        if throughput_window_steps_completed > 0 and throughput_window_time_ms > 0:
            train_tokens_per_sec = (
                throughput_window_steps_completed * train_tokens_per_iteration
            ) / (throughput_window_time_ms / 1000.0)
        memory_metrics = {}
        if memory_benchmark_active:
            torch.cuda.synchronize()
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            memory_metrics = {
                "gpu_peak_allocated_mb": peak_alloc_mb,
                "gpu_peak_reserved_mb": peak_reserved_mb,
                "gpu_end_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_end_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            }
            memory_benchmark_summary["steps"] += 1
            memory_benchmark_summary["max_peak_alloc_mb"] = max(
                memory_benchmark_summary["max_peak_alloc_mb"], peak_alloc_mb
            )
            memory_benchmark_summary["max_peak_reserved_mb"] = max(
                memory_benchmark_summary["max_peak_reserved_mb"], peak_reserved_mb
            )
            memory_benchmark_summary["sum_peak_alloc_mb"] += peak_alloc_mb
            memory_benchmark_summary["sum_peak_reserved_mb"] += peak_reserved_mb

        if master_process:
            # Prepare additional loss components
            additional_losses = {}
            if hasattr(loss_output, "base_loss") and loss_output.base_loss is not None:
                additional_losses["base_loss"] = loss_output.base_loss.item()
            if hasattr(loss_output, "type_loss") and loss_output.type_loss is not None:
                additional_losses["type_loss"] = loss_output.type_loss.item()
            if hasattr(loss_output, "aux_loss") and loss_output.aux_loss is not None:
                additional_losses["aux_loss"] = loss_output.aux_loss.item()
            additional_losses["train_tokens_seen"] = train_tokens_seen
            additional_losses["train_tokens_per_sec"] = train_tokens_per_sec
            additional_losses.update(memory_metrics)

            # Log to file
            if loss_logger:
                loss_logger.log_train_loss(
                    step, train_loss, current_learning_rate, additional_losses
                )

            # Log to wandb
            if args.wandb_project:
                log_dict = {"train_loss": train_loss, "lr": current_learning_rate, "step": step}
                log_dict.update(additional_losses)
                wandb.log(log_dict)

        # Build dynamic loss logging string - only show active (non-zero) losses
        loss_parts = [f"step:{step + 1}/{train_steps}"]
        if base_loss > 0:
            loss_parts.append(f"base:{base_loss:.4f}")
        if type_loss > 0:
            loss_parts.append(f"types:{type_loss:.4f}")
        if aux_loss > 0:
            loss_parts.append(f"aux:{aux_loss:.4f}")
        loss_parts.append(f"train_loss:{train_loss:.4f}")
        loss_parts.append(f"train_time:{approx_training_time_ms:.0f}ms")
        loss_parts.append(
            f"step_avg:{approx_training_time_ms / max(train_steps_completed, 1):.2f}ms"
        )
        loss_parts.append(f"tok_s:{train_tokens_per_sec:.1f}")
        if memory_metrics:
            loss_parts.append(f"mem_peak_alloc:{memory_metrics['gpu_peak_allocated_mb']:.1f}MB")
            loss_parts.append(f"mem_peak_res:{memory_metrics['gpu_peak_reserved_mb']:.1f}MB")

        print0(" ".join(loss_parts), console=True)

    if memory_benchmark_active and memory_benchmark_summary["steps"] > 0:
        steps = memory_benchmark_summary["steps"]
        avg_peak_alloc_mb = memory_benchmark_summary["sum_peak_alloc_mb"] / steps
        avg_peak_reserved_mb = memory_benchmark_summary["sum_peak_reserved_mb"] / steps
        print0(
            f"memory benchmark summary: steps={steps} "
            f"max_peak_alloc={memory_benchmark_summary['max_peak_alloc_mb']:.1f}MB "
            f"max_peak_reserved={memory_benchmark_summary['max_peak_reserved_mb']:.1f}MB "
            f"avg_peak_alloc={avg_peak_alloc_mb:.1f}MB "
            f"avg_peak_reserved={avg_peak_reserved_mb:.1f}MB",
            console=True,
        )
    else:
        print0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
            console=True,
        )

    if master_process and args.wandb_project:
        # Log final GPU utilization stats
        final_gpu_stats = get_combined_gpu_stats()
        if final_gpu_stats:
            print0(f"Final GPU stats: {final_gpu_stats}", console=True)
            wandb.log({"final_" + k: v for k, v in final_gpu_stats.items()})
        if memory_benchmark_active and memory_benchmark_summary["steps"] > 0:
            steps = memory_benchmark_summary["steps"]
            wandb.log(
                {
                    "final_memory_benchmark_steps": steps,
                    "final_memory_benchmark_max_peak_allocated_mb": memory_benchmark_summary[
                        "max_peak_alloc_mb"
                    ],
                    "final_memory_benchmark_max_peak_reserved_mb": memory_benchmark_summary[
                        "max_peak_reserved_mb"
                    ],
                    "final_memory_benchmark_avg_peak_allocated_mb": memory_benchmark_summary[
                        "sum_peak_alloc_mb"
                    ]
                    / steps,
                    "final_memory_benchmark_avg_peak_reserved_mb": memory_benchmark_summary[
                        "sum_peak_reserved_mb"
                    ]
                    / steps,
                }
            )
        wandb.finish()
    print0(
        f"Model parameter count (end): total={total_parameter_count:,} trainable={trainable_parameter_count:,}",
        console=True,
    )

    if distributed:
        dist.destroy_process_group()
