import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import RMSNorm
import math
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from typing import Optional, Tuple, Union, Dict, List, Any
from dataclasses import dataclass

from modeling_gpt import next_multiple_of_n, init_weights
from modeling_gpt import norm, NORM_EPSILON
from modeling_gpt import GPT
from softcap_utils import generate_tanh_softcap

from cut_cross_entropy import linear_cross_entropy

ORTHOGONAL_LOSS_DIM = -1


class EmbeddingCombiner(nn.Module):
    """Base class for combining base and type embeddings"""

    def forward(self, base_embeds: Tensor, type_embeds: Tensor) -> Tensor:
        raise NotImplementedError

    def reinitialize_for_alpha(self, alpha: float):
        """Re-initialize weights to match scaled addition with given alpha. Called after init_weights."""
        pass


class ScaledAdditionCombiner(EmbeddingCombiner):
    """Combines embeddings using scaled addition: base_embeds + alpha * type_embeds"""

    def __init__(self, types_addition_alpha: float = 1.0):
        super().__init__()
        self.alpha = types_addition_alpha

    def forward(self, base_embeds: Tensor, type_embeds: Tensor) -> Tensor:
        return base_embeds + self.alpha * type_embeds


class TypeConfig:
    def __init__(
        self,
        type_groups: Dict[str, int],
        loss_weights: Dict[str, Tuple[float, float]] = None,
        pos_weight: float = 3.0,
        neg_weight: float = 1.0,
        transformation_names_to_int: Dict[str, int] = None,
        types_loss_indices_map: Dict[str, Tuple[int, int]] = None,
    ):
        self.type_groups = type_groups
        self.loss_weights = loss_weights or {
            k: (pos_weight, neg_weight) for k in type_groups.keys()
        }

        # Use provided types_loss_indices_map if available (preferred for correctness)
        # Otherwise compute offsets from type_groups (legacy behavior)
        if types_loss_indices_map is not None:
            self.offsets = types_loss_indices_map.copy()
            # Compute total_types from the max end index
            self.total_types = max(end for start, end in types_loss_indices_map.values())
        else:
            # Legacy: compute offsets by iterating through type_groups
            self.offsets = {}
            self.total_types = 0
            for name, size in type_groups.items():
                self.offsets[name] = (self.total_types, self.total_types + size)
                self.total_types += size

        # Add this line to round the total types for optimization (needs to be divisable by world size)
        self.total_types_rounded = next_multiple_of_n(self.total_types, n=8)

        # Create transformation index to name mapping for analysis
        self.transformation_int_to_names = {}
        if transformation_names_to_int:
            self.transformation_int_to_names = {
                v: k for k, v in transformation_names_to_int.items()
            }

    def get_slice(self, group_name: str):
        start, end = self.offsets[group_name]
        return slice(start, end)


class ConditionedTransformationHead(nn.Module):
    """
    Transformation prediction head that conditions on the selected base token.
    Supports: concat / add / cross_attention modes.
    Optimized version with tensorized composition map and vectorized composition lookup.
    """

    def __init__(
        self,
        model_dim: int,
        type_vocab_size: int,
        base_vocab_size: int,
        type_config: "TypeConfig",
        conditioning_mode: str = "concat",
        base_rep_source: str = "unembedding",
        normalize_base_rep: bool = True,
        project_base_rep: bool = False,
        base_rep_proj_dim: int = None,
        use_rms_norm: bool = False,
        base_embeddings: nn.Embedding = None,
        lm_head: nn.Linear = None,
        decomposition_map: Dict[int, Tuple[int, List[int]]] = None,
        base_vocab_idx_to_extended_token_id_mapping: torch.Tensor = None,
        collapse_na_types: bool = False,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.type_vocab_size = type_vocab_size
        self.base_vocab_size = base_vocab_size
        self.type_config = type_config
        self.conditioning_mode = conditioning_mode
        self.base_rep_source = base_rep_source
        self.normalize_base_rep = normalize_base_rep
        self.project_base_rep = project_base_rep
        self.collapse_na_types = collapse_na_types

        self.base_embeddings = base_embeddings
        self.lm_head = lm_head
        self.decomposition_map = decomposition_map
        self.composition_map = None  # will be tensorized later
        if base_vocab_idx_to_extended_token_id_mapping is not None:
            self.register_buffer(
                "base_vocab_idx_to_extended_token_id_mapping",
                base_vocab_idx_to_extended_token_id_mapping.contiguous(),
                persistent=False,
            )

        # Normalization layer (if enabled)
        self.normalize_base_rep = normalize_base_rep
        self.base_rep_norm = RMSNorm(model_dim, eps=NORM_EPSILON) if use_rms_norm else None

        # Optional projection layer
        if project_base_rep:
            proj_dim = base_rep_proj_dim if base_rep_proj_dim is not None else model_dim
            # For "add" mode, projection must output model_dim
            if conditioning_mode == "add" and proj_dim != model_dim:
                raise ValueError(
                    f"For 'add' mode, base_rep_proj_dim must equal model_dim ({model_dim}), got {proj_dim}"
                )
            self.base_rep_proj = nn.Linear(model_dim, proj_dim, bias=False)
            effective_base_dim = proj_dim
        else:
            self.base_rep_proj = None
            effective_base_dim = model_dim

        # Build conditioning projection
        if conditioning_mode == "concat":
            self.projection = nn.Linear(model_dim + effective_base_dim, type_vocab_size, bias=False)
        elif conditioning_mode == "add":
            self.projection = nn.Linear(model_dim, type_vocab_size, bias=False)
        elif conditioning_mode == "cross_attention":
            self.query_proj = nn.Linear(model_dim, model_dim, bias=False)
            self.key_proj = nn.Linear(effective_base_dim, model_dim, bias=False)
            self.value_proj = nn.Linear(effective_base_dim, model_dim, bias=False)
            self.output_proj = nn.Linear(model_dim, type_vocab_size, bias=False)
        else:
            raise ValueError(f"Unknown conditioning mode: {conditioning_mode}")

    def set_decomposition_map(self, decomposition_map: Dict[int, Tuple[int, List[int]]]):
        """
        Build tensor buffers for fast vectorized lookup, inspired by ExtendedLogitsMapper.
        Properly handles list values and initializes tensors on the model's device.
        """
        self.decomposition_map = decomposition_map
        device = next(self.parameters()).device

        if not decomposition_map:
            self.register_buffer(
                "composition_lemma_base_indices", torch.empty(0, dtype=torch.long, device=device)
            )
            self.register_buffer(
                "composition_transform_indices", torch.empty(0, dtype=torch.long, device=device)
            )
            self.register_buffer(
                "composition_transform_combinations_indices",
                torch.empty(0, 0, dtype=torch.long, device=device),
            )
            self.register_buffer(
                "composition_transformable_indices", torch.empty(0, dtype=torch.long, device=device)
            )
            self.register_buffer(
                "composition_combo_default_mask",
                torch.empty(0, 0, dtype=torch.bool, device=device),
                persistent=False,
            )
            self.register_buffer(
                "composition_group_default_starts",
                torch.empty(0, dtype=torch.long, device=device),
                persistent=False,
            )
            self.register_buffer(
                "composition_group_default_ends",
                torch.empty(0, dtype=torch.long, device=device),
                persistent=False,
            )
            return

        # ---- Convert lists to tuples so they're hashable ----
        group_names = list(self.type_config.type_groups.keys())
        group_slices = [self.type_config.get_slice(name) for name in group_names]

        default_combo = [sl.start for sl in group_slices]

        def _combo_from_transform_list(transform_list: List[int]) -> Tuple[int, ...]:
            combo = list(default_combo)
            for transform_idx in transform_list:
                if not isinstance(transform_idx, int):
                    continue
                for group_pos, sl in enumerate(group_slices):
                    if sl.start <= transform_idx < sl.stop:
                        combo[group_pos] = transform_idx
                        break
            return tuple(combo)

        inverse_map = {}
        for ext_id, (base_id, transform_list) in decomposition_map.items():
            combo = _combo_from_transform_list(transform_list)
            inverse_map[ext_id] = (base_id, combo)

        transformable_indices = list(inverse_map.keys())  # [num_entries]
        base_ids = [inverse_map[e][0] for e in transformable_indices]
        transform_combos = [inverse_map[e][1] for e in transformable_indices]

        # ---- Compute unique transform combinations ----
        unique_combos = sorted(set(transform_combos))
        combo_to_idx = {combo: i for i, combo in enumerate(unique_combos)}

        for combo in unique_combos:
            if len(combo) != len(group_slices):
                raise ValueError(
                    f"Invalid transform combo length {len(combo)} (expected {len(group_slices)})"
                )
            for group_pos, sl in enumerate(group_slices):
                val = combo[group_pos]
                if val == -1:
                    continue
                if not (sl.start <= val < sl.stop):
                    raise ValueError(
                        f"Transform index {val} out of slice for group {group_pos}: {sl.start}-{sl.stop - 1}"
                    )

        num_groups = len(self.type_config.type_groups)
        transform_combinations_indices = torch.tensor(
            unique_combos, dtype=torch.long, device=device
        )

        transform_indices = torch.tensor(
            [combo_to_idx[c] for c in transform_combos], dtype=torch.long, device=device
        )
        lemma_base_indices = torch.tensor(base_ids, dtype=torch.long, device=device)
        transformable_indices = torch.tensor(transformable_indices, dtype=torch.long, device=device)

        # ---- Register buffers so they move with model.to(device) ----
        self.register_buffer(
            "composition_lemma_base_indices", lemma_base_indices.contiguous(), persistent=False
        )
        self.register_buffer(
            "composition_transform_indices", transform_indices.contiguous(), persistent=False
        )
        self.register_buffer(
            "composition_transform_combinations_indices",
            transform_combinations_indices.contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "composition_transformable_indices",
            transformable_indices.contiguous(),
            persistent=False,
        )

        group_names = list(self.type_config.type_groups.keys())
        name_to_idx = {v: k for k, v in self.type_config.transformation_int_to_names.items()}
        self.composition_group_allowed = []
        group_starts = []
        group_ends = []
        for group_name in group_names:
            sl = self.type_config.get_slice(group_name)
            group_starts.append(sl.start)
            group_ends.append(sl.stop - 1)

            group_size = sl.stop - sl.start
            allowed = torch.eye(group_size, dtype=torch.bool, device=device)
            prefix = f"{group_name}_"
            for rel_label in range(group_size):
                label_idx = sl.start + rel_label
                label_name = self.type_config.transformation_int_to_names.get(label_idx, "")
                if not label_name.startswith(prefix):
                    continue
                value = label_name[len(prefix) :]
                if "+" not in value:
                    continue
                for part in value.split("+"):
                    component = prefix + part
                    comp_idx = name_to_idx.get(component)
                    if comp_idx is None:
                        continue
                    comp_rel = comp_idx - sl.start
                    if 0 <= comp_rel < group_size:
                        allowed[rel_label, comp_rel] = True
            self.composition_group_allowed.append(allowed)
        starts_tensor = torch.tensor(group_starts, dtype=torch.long, device=device)
        ends_tensor = torch.tensor(group_ends, dtype=torch.long, device=device)
        combo_default_mask = (
            (transform_combinations_indices == starts_tensor.unsqueeze(0))
            | (transform_combinations_indices == ends_tensor.unsqueeze(0))
            | (transform_combinations_indices == -1)
        )
        self.register_buffer(
            "composition_combo_default_mask", combo_default_mask.contiguous(), persistent=False
        )
        self.register_buffer(
            "composition_group_default_starts", starts_tensor.contiguous(), persistent=False
        )
        self.register_buffer(
            "composition_group_default_ends", ends_tensor.contiguous(), persistent=False
        )

        # Precompute per-base allowed transforms for each group (used to turn off impossible predictions).
        max_base_id = int(lemma_base_indices.max().item()) if lemma_base_indices.numel() else 0
        if hasattr(self, "base_vocab_idx_to_extended_token_id_mapping"):
            try:
                max_base_id = max(
                    max_base_id, int(self.base_vocab_idx_to_extended_token_id_mapping.max().item())
                )
            except RuntimeError:
                pass
        num_base_rows = max_base_id + 1
        base_allowed_by_group = []
        if num_base_rows > 0 and transform_indices.numel():
            combo_per_entry = transform_combinations_indices[
                transform_indices
            ]  # [num_entries, num_groups]
            for group_pos, sl in enumerate(group_slices):
                group_size = sl.stop - sl.start
                allowed = torch.zeros((num_base_rows, group_size), dtype=torch.bool, device=device)
                vals = combo_per_entry[:, group_pos]
                vals = torch.where(vals == -1, torch.full_like(vals, sl.start), vals)
                rel = vals - sl.start
                rel = torch.clamp(rel, 0, group_size - 1)
                allowed[lemma_base_indices, rel] = True
                if lemma_base_indices.numel():
                    allowed[lemma_base_indices, 0] = True
                base_allowed_by_group.append(allowed)
        self.composition_base_allowed = base_allowed_by_group

    def _compute_invalid_transform_mask(
        self,
        base_extended_ids: torch.Tensor,
        top1_transforms: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not getattr(self, "composition_base_allowed", None):
            return None
        invalid_masks = []
        for group_idx, allowed in enumerate(self.composition_base_allowed):
            if allowed is None or allowed.numel() == 0:
                invalid_masks.append(
                    torch.zeros(
                        base_extended_ids.shape[0],
                        dtype=torch.bool,
                        device=base_extended_ids.device,
                    )
                )
                continue
            group_start = self.composition_group_default_starts[group_idx]
            group_size = allowed.shape[1]
            pred = top1_transforms[:, group_idx]
            rel = pred - group_start
            rel = torch.clamp(rel, 0, group_size - 1)
            safe_ids = base_extended_ids
            in_range = safe_ids < allowed.shape[0]
            if allowed.shape[0] > 0:
                safe_ids = torch.clamp(safe_ids, max=allowed.shape[0] - 1)
            is_allowed = allowed[safe_ids, rel]
            is_allowed = is_allowed | (~in_range)
            invalid_masks.append(~is_allowed)
        return torch.stack(invalid_masks, dim=-1)

    def _apply_base_allowed_transforms(
        self,
        base_extended_ids: torch.Tensor,
        top1_transforms: torch.Tensor,
    ) -> torch.Tensor:
        """Turn off predicted transforms that are not available for the base token."""
        if not getattr(self, "composition_base_allowed", None):
            return top1_transforms

        invalid_mask = self._compute_invalid_transform_mask(base_extended_ids, top1_transforms)
        if invalid_mask is None:
            return top1_transforms

        adjusted = top1_transforms.clone()
        for group_idx, invalid in enumerate(invalid_mask.unbind(dim=-1)):
            if not invalid.any():
                continue
            group_start = self.composition_group_default_starts[group_idx]
            adjusted[:, group_idx] = torch.where(invalid, group_start, adjusted[:, group_idx])
        return adjusted

    def _combo_compatibility(self, top1_transforms: torch.Tensor) -> torch.Tensor:
        """Return compatibility mask [batch, num_combos, num_groups] for predicted transforms."""
        if not getattr(self, "composition_group_allowed", None):
            return top1_transforms.unsqueeze(
                1
            ) == self.composition_transform_combinations_indices.unsqueeze(0)

        batch_size = top1_transforms.shape[0]
        num_combos = self.composition_transform_combinations_indices.shape[0]
        num_groups = self.composition_transform_combinations_indices.shape[1]
        compat = torch.zeros(
            (batch_size, num_combos, num_groups), dtype=torch.bool, device=top1_transforms.device
        )
        for group_idx, group_name in enumerate(self.type_config.type_groups.keys()):
            sl = self.type_config.get_slice(group_name)
            group_size = sl.stop - sl.start
            label_rel = self.composition_transform_combinations_indices[:, group_idx] - sl.start
            valid_label = (label_rel >= 0) & (label_rel < group_size)
            pred_rel = top1_transforms[:, group_idx] - sl.start
            allowed_rel = self.composition_group_allowed[group_idx]
            if allowed_rel.device != top1_transforms.device:
                allowed_rel = allowed_rel.to(top1_transforms.device)
                self.composition_group_allowed[group_idx] = allowed_rel
            label_rel_safe = torch.where(valid_label, label_rel, torch.zeros_like(label_rel))
            allowed = allowed_rel[label_rel_safe][:, pred_rel]  # [num_combos, batch]
            allowed = allowed.transpose(0, 1)
            allowed = allowed & valid_label.unsqueeze(0)
            compat[:, :, group_idx] = allowed
        return compat

    def get_base_representation(self, base_indices: torch.Tensor) -> torch.Tensor:
        """Get base token embeddings or unembeddings, with optional normalization and projection."""
        # Get raw representation
        if self.base_rep_source == "embedding":
            base_rep = self.base_embeddings(base_indices)
        else:
            base_rep = F.embedding(base_indices, self.lm_head.weight)

        # Apply normalization if enabled
        if self.normalize_base_rep:
            base_rep = norm(base_rep, self.base_rep_norm)

        # Apply projection if enabled
        if self.base_rep_proj is not None:
            base_rep = self.base_rep_proj(base_rep)

        return base_rep

    def forward(
        self,
        hidden_states: torch.Tensor,
        top1_base_indices: torch.Tensor,
        return_top1_extended: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass: predict type logits conditioned on selected base token."""
        base_rep = self.get_base_representation(top1_base_indices)

        if self.conditioning_mode == "concat":
            combined = torch.cat([hidden_states, base_rep], dim=-1)
            type_logits = self.projection(combined)
        elif self.conditioning_mode == "add":
            combined = hidden_states + base_rep
            type_logits = self.projection(combined)
        elif self.conditioning_mode == "cross_attention":
            Q = self.query_proj(hidden_states)
            K = self.key_proj(base_rep)
            V = self.value_proj(base_rep)
            attn = (Q * K).sum(dim=-1, keepdim=True) / math.sqrt(self.model_dim)
            attn_weights = torch.sigmoid(attn)
            attended = attn_weights * V
            type_logits = self.output_proj(attended)

        if return_top1_extended:
            top1_extended_idx = self.compose_extended_token(top1_base_indices, type_logits)
            return type_logits, top1_extended_idx
        else:
            return type_logits

    def compose_extended_token(
        self,
        base_indices: torch.Tensor,
        type_logits: torch.Tensor,
        collapse_na_types_override: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Vectorized composition following the structure of ExtendedLogitsMapper.forward_advanced_indexing.
        """
        collapse_na_types = (
            self.collapse_na_types
            if collapse_na_types_override is None
            else collapse_na_types_override
        )

        # ---- Compute top-1 transform for each group ----
        top1_per_group = []
        for group_name, _ in self.type_config.type_groups.items():
            sl = self.type_config.get_slice(group_name)

            # NA type (last index) is inactive - so slice up to one before it
            top1_idx = type_logits[:, sl].argmax(dim=-1)
            if collapse_na_types:
                group_size = sl.stop - sl.start
                if group_size > 1:
                    na_rel = group_size - 1
                    top1_idx = torch.where(top1_idx == na_rel, torch.zeros_like(top1_idx), top1_idx)
            top1_per_group.append(top1_idx + sl.start)
        top1_transforms = torch.stack(top1_per_group, dim=-1)  # [batch, num_groups]

        return self._compose_from_top1_transforms(base_indices, top1_transforms)

    def compose_extended_token_from_type_ids(
        self,
        base_indices: torch.Tensor,
        type_ids: torch.Tensor,
        collapse_na_types_override: Optional[bool] = None,
    ) -> torch.Tensor:
        """Compose extended token IDs from explicit per-group type IDs."""
        collapse_na_types = (
            self.collapse_na_types
            if collapse_na_types_override is None
            else collapse_na_types_override
        )

        top1_per_group = []
        for group_name, _ in self.type_config.type_groups.items():
            sl = self.type_config.get_slice(group_name)
            top1_idx = type_ids[:, sl].argmax(dim=-1)
            if collapse_na_types:
                group_size = sl.stop - sl.start
                if group_size > 1:
                    na_rel = group_size - 1
                    top1_idx = torch.where(top1_idx == na_rel, torch.zeros_like(top1_idx), top1_idx)
            top1_per_group.append(top1_idx + sl.start)
        top1_transforms = torch.stack(top1_per_group, dim=-1)

        return self._compose_from_top1_transforms(base_indices, top1_transforms)

    def _compose_from_top1_transforms(
        self,
        base_indices: torch.Tensor,
        top1_transforms: torch.Tensor,
    ) -> torch.Tensor:
        if self.composition_transformable_indices.numel() == 0:
            if self.base_vocab_idx_to_extended_token_id_mapping is not None:
                return self.base_vocab_idx_to_extended_token_id_mapping[base_indices]
            return base_indices

        # ---- Map base_indices (like lemma_base_indices in ExtendedLogitsMapper) ----
        # Convert to extended vocab IDs
        if self.base_vocab_idx_to_extended_token_id_mapping is not None:
            base_extended_ids = self.base_vocab_idx_to_extended_token_id_mapping[base_indices]
        else:
            base_extended_ids = base_indices

        # Turn off transforms that are not available for this base token.
        top1_transforms = self._apply_base_allowed_transforms(base_extended_ids, top1_transforms)

        # ---- Build a combination index for each batch element ----
        # Compare batch’s top1 combo to all known transform_combinations
        compat = self._combo_compatibility(top1_transforms)
        match_count = compat.sum(dim=-1)  # [batch, num_combinations]
        valid_counts = (self.composition_transform_combinations_indices != -1).sum(
            dim=-1
        )  # [num_combinations]
        combo_match = match_count == valid_counts.unsqueeze(0)  # [batch, num_combinations]
        # Pick best matching combination
        combo_idx = combo_match.float().argmax(dim=-1)

        # ---- Compose via precomputed tensors (chunked to avoid OOM) ----
        # composition_transformable_indices: extended token IDs
        # composition_lemma_base_indices: base vocab index for each extended token
        # composition_transform_indices: combo index for each extended token
        # combo_idx: selected combo for each batch item
        entry_combo_idx = self.composition_transform_indices
        num_entries = int(entry_combo_idx.numel())
        chunk_size = max(1, int(getattr(self, "compose_entry_chunk_size", 1024)))

        batch_size = base_extended_ids.shape[0]
        has_match = torch.zeros(batch_size, dtype=torch.bool, device=base_extended_ids.device)
        matched_idx = torch.zeros(batch_size, dtype=torch.long, device=base_extended_ids.device)

        for start in range(0, num_entries, chunk_size):
            end = min(start + chunk_size, num_entries)
            lemma_chunk = self.composition_lemma_base_indices[start:end]
            base_match_chunk = base_extended_ids.unsqueeze(1) == lemma_chunk.unsqueeze(0)
            combo_chunk = entry_combo_idx[start:end]
            combo_match_chunk = combo_idx.unsqueeze(1) == combo_chunk.unsqueeze(0)
            final_match = base_match_chunk & combo_match_chunk
            any_match = final_match.any(dim=-1)
            if any_match.any():
                match_idx_chunk = final_match.float().argmax(dim=-1)
                update = (~has_match) & any_match
                matched_idx = torch.where(update, match_idx_chunk + start, matched_idx)
                has_match = has_match | any_match
            if has_match.all():
                break

        composed_ids = self.composition_transformable_indices[matched_idx]
        extended_indices = torch.where(has_match, composed_ids, base_extended_ids)
        return extended_indices

    def compute_combo_resolution_stats(
        self,
        base_indices: torch.Tensor,
        type_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        max_samples: Optional[int] = 4096,
        entry_chunk_size: int = 65536,
        collapse_na_types_override: Optional[bool] = None,
        return_samples: bool = False,
        max_error_samples: int = 50,
        context_ids: Optional[torch.Tensor] = None,
        context_before: int = 5,
        context_after: int = 5,
        context_modifiers: Optional[torch.Tensor] = None,
        target_base_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute statistics about invalid transformation combos and drop-based recovery."""
        collapse_na_types = (
            self.collapse_na_types
            if collapse_na_types_override is None
            else collapse_na_types_override
        )
        if self.composition_transformable_indices.numel() == 0:
            total = (
                int(valid_mask.sum().item())
                if valid_mask is not None
                else int(base_indices.numel())
            )
            group_names = list(self.type_config.type_groups.keys())
            return {
                "total": total,
                "eligible": 0,
                "exact": 0,
                "drop": 0,
                "no_match": 0,
                "base_no_combo": total,
                "drop_counts": {},
                "drop_groups": {name: 0 for name in group_names},
            }

        positions = torch.arange(base_indices.shape[0], device=base_indices.device)
        if valid_mask is not None:
            mask = valid_mask.bool()
            base_indices = base_indices[mask]
            type_logits = type_logits[mask]
            positions = positions[mask]

        total = int(base_indices.numel())
        if max_samples is not None:
            if max_samples <= 0:
                group_names = list(self.type_config.type_groups.keys())
                return {
                    "total": 0,
                    "eligible": 0,
                    "exact": 0,
                    "drop": 0,
                    "no_match": 0,
                    "base_no_combo": 0,
                    "drop_counts": {},
                    "drop_groups": {name: 0 for name in group_names},
                }
            if total > max_samples:
                perm = torch.randperm(total, device=base_indices.device)[:max_samples]
                base_indices = base_indices[perm]
                type_logits = type_logits[perm]
                positions = positions[perm]
                total = int(base_indices.numel())
        if total == 0:
            group_names = list(self.type_config.type_groups.keys())
            return {
                "total": 0,
                "eligible": 0,
                "exact": 0,
                "drop": 0,
                "no_match": 0,
                "base_no_combo": 0,
                "drop_counts": {},
                "drop_groups": {name: 0 for name in group_names},
            }

        top1_per_group = []
        for group_name, _ in self.type_config.type_groups.items():
            sl = self.type_config.get_slice(group_name)
            top1_idx = type_logits[:, sl].argmax(dim=-1)
            if collapse_na_types:
                group_size = sl.stop - sl.start
                if group_size > 1:
                    na_rel = group_size - 1
                    top1_idx = torch.where(top1_idx == na_rel, torch.zeros_like(top1_idx), top1_idx)
            top1_per_group.append(top1_idx + sl.start)
        top1_transforms = torch.stack(top1_per_group, dim=-1)

        if self.base_vocab_idx_to_extended_token_id_mapping is not None:
            base_extended_ids = self.base_vocab_idx_to_extended_token_id_mapping[base_indices]
        else:
            base_extended_ids = base_indices

        top1_original = top1_transforms
        invalid_mask = self._compute_invalid_transform_mask(base_extended_ids, top1_original)
        top1_transforms = self._apply_base_allowed_transforms(base_extended_ids, top1_original)

        combo_indices = self.composition_transform_combinations_indices
        if collapse_na_types:
            combo_indices = combo_indices.clone()
            for group_idx, group_name in enumerate(self.type_config.type_groups.keys()):
                sl = self.type_config.get_slice(group_name)
                group_size = sl.stop - sl.start
                if group_size > 1:
                    na_idx = sl.stop - 1
                    combo_indices[:, group_idx] = torch.where(
                        combo_indices[:, group_idx] == na_idx,
                        torch.full_like(combo_indices[:, group_idx], sl.start),
                        combo_indices[:, group_idx],
                    )

        if not getattr(self, "composition_group_allowed", None):
            compat = top1_transforms.unsqueeze(1) == combo_indices.unsqueeze(0)
        else:
            batch_size = top1_transforms.shape[0]
            num_combos = combo_indices.shape[0]
            num_groups = combo_indices.shape[1]
            compat = torch.zeros(
                (batch_size, num_combos, num_groups),
                dtype=torch.bool,
                device=top1_transforms.device,
            )
            for group_idx, group_name in enumerate(self.type_config.type_groups.keys()):
                sl = self.type_config.get_slice(group_name)
                group_size = sl.stop - sl.start
                label_rel = combo_indices[:, group_idx] - sl.start
                valid_label = (label_rel >= 0) & (label_rel < group_size)
                pred_rel = top1_transforms[:, group_idx] - sl.start
                allowed_rel = self.composition_group_allowed[group_idx]
                if allowed_rel.device != top1_transforms.device:
                    allowed_rel = allowed_rel.to(top1_transforms.device)
                    self.composition_group_allowed[group_idx] = allowed_rel
                label_rel_safe = torch.where(valid_label, label_rel, torch.zeros_like(label_rel))
                allowed = allowed_rel[label_rel_safe][:, pred_rel]  # [num_combos, batch]
                allowed = allowed.transpose(0, 1)
                allowed = allowed & valid_label.unsqueeze(0)
                compat[:, :, group_idx] = allowed
        match_count = compat.sum(dim=-1)
        valid_counts = (self.composition_transform_combinations_indices != -1).sum(dim=-1)
        combo_match = match_count == valid_counts.unsqueeze(0)
        combo_idx = combo_match.float().argmax(dim=-1)

        num_groups = self.composition_transform_combinations_indices.shape[1]
        pred_default = (top1_transforms == self.composition_group_default_starts.unsqueeze(0)) | (
            top1_transforms == self.composition_group_default_ends.unsqueeze(0)
        )

        num_entries = int(self.composition_lemma_base_indices.numel())
        chunk_size = max(1, int(entry_chunk_size))

        base_has_combo = torch.zeros(total, dtype=torch.bool, device=base_indices.device)
        has_exact = torch.zeros(total, dtype=torch.bool, device=base_indices.device)
        for start in range(0, num_entries, chunk_size):
            end = min(start + chunk_size, num_entries)
            lemma_chunk = self.composition_lemma_base_indices[start:end]
            base_match_chunk = base_extended_ids.unsqueeze(1) == lemma_chunk.unsqueeze(0)
            base_has_combo |= base_match_chunk.any(dim=-1)

            combo_match_entries = combo_idx.unsqueeze(1) == self.composition_transform_indices[
                start:end
            ].unsqueeze(0)
            has_exact |= (base_match_chunk & combo_match_entries).any(dim=-1)

        eligible = base_has_combo
        invalid = eligible & (~has_exact)
        no_match = invalid
        base_no_combo = ~base_has_combo

        group_names = list(self.type_config.type_groups.keys())
        invalid_transform_counts = {name: {} for name in group_names}
        invalid_transform_totals = {name: 0 for name in group_names}
        if invalid_mask is not None:
            invalid_mask_cpu = invalid_mask.detach().cpu()
            top1_cpu = top1_original.detach().cpu()
            name_map = self.type_config.transformation_int_to_names
            for group_idx, group_name in enumerate(group_names):
                bad_rows = invalid_mask_cpu[:, group_idx]
                if not bad_rows.any():
                    continue
                values = top1_cpu[bad_rows, group_idx].tolist()
                for val in values:
                    key = name_map.get(int(val), f"type_{int(val)}")
                    invalid_transform_counts[group_name][key] = (
                        invalid_transform_counts[group_name].get(key, 0) + 1
                    )
                invalid_transform_totals[group_name] = int(
                    sum(invalid_transform_counts[group_name].values())
                )
        stats = {
            "total": total,
            "eligible": int(eligible.sum().item()),
            "exact": int(has_exact.sum().item()),
            "drop": 0,
            "no_match": int(no_match.sum().item()),
            "base_no_combo": int(base_no_combo.sum().item()),
            "drop_counts": {},
            "drop_groups": {name: 0 for name in group_names},
            "invalid_transform_counts": invalid_transform_counts,
            "invalid_transform_totals": invalid_transform_totals,
        }
        target_base_ext = None
        if target_base_ids is not None:
            if self.base_vocab_idx_to_extended_token_id_mapping is not None:
                target_base_ext = self.base_vocab_idx_to_extended_token_id_mapping[target_base_ids]
            else:
                target_base_ext = target_base_ids

        if return_samples and max_error_samples:
            invalid_idx = no_match.nonzero(as_tuple=True)[0]
            if invalid_idx.numel() > 0:
                if invalid_idx.numel() > max_error_samples:
                    perm = torch.randperm(invalid_idx.numel(), device=invalid_idx.device)[
                        :max_error_samples
                    ]
                    invalid_idx = invalid_idx[perm]
                sample_positions = positions[invalid_idx]
                name_map = self.type_config.transformation_int_to_names
                pred_names = [
                    [name_map.get(int(idx), f"type_{int(idx)}") for idx in row]
                    for row in top1_transforms[invalid_idx].detach().cpu().tolist()
                ]
                base_ids_sel = base_extended_ids[invalid_idx].detach().cpu().tolist()
                pred_all_default_sel = pred_default[invalid_idx].all(dim=-1).detach().cpu().tolist()
                entry_combo_all_default = self.composition_combo_default_mask[
                    self.composition_transform_indices
                ].all(dim=-1)
                samples = []
                for i, base_id in enumerate(base_ids_sel):
                    error_type = "no_match"
                    predicted = [
                        {"group": group_names[g_idx], "value": pred_names[i][g_idx]}
                        for g_idx in range(len(group_names))
                    ]
                    has_default_combo = False
                    if entry_combo_all_default.numel() > 0:
                        base_mask = self.composition_lemma_base_indices == int(base_id)
                        if base_mask.any():
                            has_default_combo = bool(
                                (entry_combo_all_default & base_mask).any().item()
                            )
                    context_tokens = []
                    context_mods = []
                    target_offset = None
                    target_token = None
                    target_base_id = None
                    position_idx = int(sample_positions[i].item())
                    if context_ids is not None and 0 <= position_idx < context_ids.shape[0]:
                        start = max(0, position_idx - context_before)
                        end = min(int(context_ids.shape[0]), position_idx + 1 + context_after)
                        context_tokens = context_ids[start:end].detach().cpu().tolist()
                        if context_modifiers is not None and context_modifiers.shape[0] >= end:
                            context_mods = context_modifiers[start:end].detach().cpu().tolist()
                        target_offset = position_idx - start
                        target_token = int(context_ids[position_idx].item())
                    if target_base_ext is not None and 0 <= position_idx < target_base_ext.shape[0]:
                        target_base_id = int(target_base_ext[position_idx].item())
                    samples.append(
                        {
                            "base_id": int(base_id),
                            "error_type": error_type,
                            "drop_count": 0,
                            "predicted_transforms": predicted,
                            "drop_groups": [],
                            "resolved_transforms": [],
                            "pred_all_default": bool(pred_all_default_sel[i]),
                            "has_default_combo": has_default_combo,
                            "context_tokens": context_tokens,
                            "context_modifiers": context_mods,
                            "target_offset": target_offset,
                            "target_token": target_token,
                            "target_base_id": target_base_id,
                            "position": position_idx,
                        }
                    )
                stats["error_samples"] = samples
        return stats


@dataclass
class CompositionalOutputs:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    base_logits: Optional[torch.FloatTensor] = None
    type_logits: Optional[torch.FloatTensor] = None
    type_logits_pred_base: Optional[torch.FloatTensor] = None
    aux_logits: Optional[torch.FloatTensor] = None
    base_loss: Optional[torch.FloatTensor] = None
    type_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    top1_extended_idx: Optional[torch.Tensor] = None


class CompositionalGPT(GPT):
    def __init__(
        self,
        vocab_size: int,
        extended_vocab_size: int,
        type_config: TypeConfig,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        intermediate_dim: int,
        max_seq_len: int,
        use_gated_proj: bool = False,
        use_rms_norm: bool = False,
        use_gqa: bool = False,
        num_kv_heads: int = None,
        use_rope_scaling: bool = False,
        rope_scaling_factor: float = 1.0,
        rope_scaling_type: str = "linear",
        reorder_norms: bool = True,
        init_strategy: str = "default",
        init_std: float = 0.02,
        base_dim: int = None,
        base_loss_alpha: float = 1.0,
        type_loss_alpha: float = None,
        use_shift: bool = False,
        softcap: float = 30.0,
        types_addition_alpha: float = 1.0,
        ignore_na_types: bool = True,
        collapse_na_types: bool = False,
        ambiguous_type_loss_mode: str = "equal",
        ambiguous_type_loss_lambda: float = 0.0,
        transformation_conditioning_mode: str = "concat",
        transformation_base_rep_source: str = "unembedding",
        transformation_normalize_base_rep: bool = True,
        transformation_project_base_rep: bool = False,
        transformation_base_rep_proj_dim: int = None,
        modifier_map_dict: dict = None,
        new_transform_group_indices: dict = None,
        unified_modifier_groups: list = None,
        use_linear_cross_entropy: bool = True,
        eos_token_id: int = 50256,
    ):
        constructor_args = {k: v for k, v in locals().items() if k not in {"self", "__class__"}}
        # Store original unpadded vocab size before parent init (parent will pad it)
        self.base_vocab_size_unpadded = vocab_size

        # Initialize parent GPT class
        super().__init__(
            vocab_size,
            num_layers,
            num_heads,
            model_dim,
            intermediate_dim,
            max_seq_len,
            use_gated_proj,
            use_rms_norm,
            use_gqa,
            num_kv_heads,
            use_rope_scaling,
            rope_scaling_factor,
            rope_scaling_type,
            reorder_norms,
            init_strategy,
            init_std,
            base_dim,
            use_linear_cross_entropy,
            eos_token_id,
        )

        self._checkpoint_constructor = constructor_args
        # Add compositional-specific attributes
        self.extended_vocab_size = extended_vocab_size
        self.type_config = type_config
        self.base_loss_alpha = base_loss_alpha

        # Dual-stream mode attributes
        self.new_transform_group_indices = new_transform_group_indices
        self.new_transform_group_names = (
            list(new_transform_group_indices.keys()) if new_transform_group_indices else []
        )
        self.unified_modifier_groups = (
            unified_modifier_groups  # For 2D modifier arrays covering all groups
        )

        # Validate unified modifier groups against type_config
        if unified_modifier_groups is not None:
            for group_name in unified_modifier_groups:
                if (
                    group_name not in type_config.type_groups
                    and group_name not in new_transform_group_indices
                ):
                    print(
                        f"WARNING: Unified modifier group '{group_name}' not found in type_config or new_transform_group_indices"
                    )
            print(
                f"Model initialized with {len(unified_modifier_groups)} unified modifier groups: {unified_modifier_groups}"
            )

        # Build modifier_map tensor from dict if provided
        # Note: modifier_map_tensor will be registered as a buffer in _build_modifier_map_tensor()
        if modifier_map_dict is not None:
            self._build_modifier_map_tensor(modifier_map_dict)
        self.type_loss_alpha = (
            len(type_config.type_groups) if type_loss_alpha is None else type_loss_alpha
        )
        self.ambiguous_type_loss_mode = ambiguous_type_loss_mode
        self.ambiguous_type_loss_lambda = ambiguous_type_loss_lambda
        self.use_shift = use_shift
        self.softcap = softcap
        self.types_addition_alpha = types_addition_alpha
        self.ignore_na_types = ignore_na_types
        self.collapse_na_types = collapse_na_types
        self._na_to_no_map = {}
        self._no_type_ids = []
        for group_name in self.type_config.type_groups.keys():
            start, end = self.type_config.offsets[group_name]
            if end <= start:
                continue
            no_idx = start
            na_idx = end - 1
            self._no_type_ids.append(no_idx)
            if na_idx != no_idx:
                self._na_to_no_map[na_idx] = no_idx
        self.transformation_conditioning_mode = transformation_conditioning_mode
        self.transformation_base_rep_source = transformation_base_rep_source
        self.tanh_softcap = generate_tanh_softcap(softcap, approx=False) if softcap else None

        # Compositional components
        self.embed_types = nn.Embedding(type_config.total_types_rounded, model_dim, padding_idx=0)
        self.register_buffer(
            "token_to_type_ids",
            torch.zeros(self.extended_vocab_size, type_config.total_types_rounded),
            persistent=False,
        )

        # Initialize conditioned transformation head
        self.conditioned_transformation_head = ConditionedTransformationHead(
            model_dim=model_dim,
            type_vocab_size=type_config.total_types_rounded,
            base_vocab_size=vocab_size,
            type_config=type_config,
            conditioning_mode=transformation_conditioning_mode,
            base_rep_source=transformation_base_rep_source,
            normalize_base_rep=transformation_normalize_base_rep,
            project_base_rep=transformation_project_base_rep,
            base_rep_proj_dim=transformation_base_rep_proj_dim,
            use_rms_norm=use_rms_norm,
            base_embeddings=self.embed,  # From parent GPT class
            lm_head=self.lm_head,  # From parent GPT class
            decomposition_map=None,  # Will be set later
            base_vocab_idx_to_extended_token_id_mapping=None,  # Will be set later
            collapse_na_types=self.collapse_na_types,
        )

        # Initialize auxiliary LM head if enabled
        self.aux_lm_head = None

        # Initialize embedding combiner
        self.embedding_combiner = ScaledAdditionCombiner(types_addition_alpha)

        # Initialize orthogonal projector if enabled
        self.orthogonal_projector = None

        # Apply custom initialization
        if init_strategy in ["olmo2", "minicpm"]:
            init_weights(self, init_strategy, init_std, model_dim, base_dim)

        # Re-initialize embedding combiner after init_weights (which may have overwritten it)
        self.embedding_combiner.reinitialize_for_alpha(types_addition_alpha)

        # Type loss functions
        self.type_loss_functions = nn.ModuleDict()
        for group_name in type_config.type_groups.keys():
            group_size = type_config.type_groups[group_name]
            pos_weight, neg_weight = type_config.loss_weights[group_name]
            weights = torch.tensor([pos_weight] * (group_size - 1) + [neg_weight])
            self.type_loss_functions[group_name] = nn.CrossEntropyLoss(weight=weights)

        # Mappings (set later)
        self.token_id_to_base_id_mapping = None
        self.token_id_is_base_or_inflection_mapping = None
        self.base_vocab_idx_to_extended_token_id_mapping = None
        self.decomposition_map = None  # Will store decomposition map for composing tokens

    def cuda(self, device=None):
        """Override cuda() to ensure all components are moved"""
        result = super().cuda(device)

        # Ensure cached tensors are moved (using correct names)
        if hasattr(self, "_token_id_to_base_id_mapping_tensor"):
            self._token_id_to_base_id_mapping_tensor = (
                self._token_id_to_base_id_mapping_tensor.cuda(device)
            )
        if hasattr(self, "_token_id_is_base_or_inflection_mapping_tensor"):
            self._token_id_is_base_or_inflection_mapping_tensor = (
                self._token_id_is_base_or_inflection_mapping_tensor.cuda(device)
            )

        return result

    def set_token_mappings(
        self,
        base_token_indices,
        token_id_to_base_id_mapping: Dict[int, int],
        base_or_inflection_token_ids: List[int],
    ):
        self._checkpoint_token_mappings = {
            "base_token_indices": list(base_token_indices),
            "token_id_to_base_id_mapping": token_id_to_base_id_mapping,
            "base_or_inflection_token_ids": list(base_or_inflection_token_ids),
        }
        # Create mapping from extended vocab indices to base vocab indices
        self.token_id_to_base_id_mapping = -1 * torch.ones(
            self.extended_vocab_size, dtype=torch.long, device=self.embed.weight.device
        )

        # Create a mapping from extended vocab base IDs to base vocab indices
        unique_base_ids = sorted(
            set(token_id_to_base_id_mapping.values()) | set(base_token_indices)
        )
        expected_base_vocab = getattr(self, "base_vocab_size_unpadded", self.vocab_size)
        if len(unique_base_ids) != expected_base_vocab:
            raise ValueError(
                f"Base vocab size mismatch in set_token_mappings:\n"
                f"  model.base_vocab_size_unpadded: {expected_base_vocab}\n"
                f"  model.vocab_size (padded): {self.vocab_size}\n"
                f"  unique_base_ids: {len(unique_base_ids)}\n"
                f"This usually means the tokenizer bundle cache was built with different base-vocab sizing.\n"
                f"Rebuild the tokenizer bundle cache (e.g., --overwrite_bundle_cache) and retry."
            )
        base_id_to_base_vocab_idx = {base_id: idx for idx, base_id in enumerate(unique_base_ids)}
        base_vocab_idx_to_base_id = {v: k for k, v in base_id_to_base_vocab_idx.items()}

        # Create a tensor mapping from base vocab indices to extended vocab base token IDs
        # This is used for analysis purposes to map base predictions back to extended vocab space
        self.base_vocab_idx_to_extended_token_id_mapping = -1 * torch.ones(
            self.vocab_size, dtype=torch.long, device=self.embed.weight.device
        )
        for base_vocab_idx, base_id in base_vocab_idx_to_base_id.items():
            # Find the extended vocab token ID for this base ID (should be the base ID itself for base tokens)
            self.base_vocab_idx_to_extended_token_id_mapping[base_vocab_idx] = base_id
        # Apply the mappings
        for ext_token_id, base_id in token_id_to_base_id_mapping.items():
            self.token_id_to_base_id_mapping[ext_token_id] = base_id_to_base_vocab_idx[base_id]

        for ext_token_id in base_token_indices:
            if self.token_id_to_base_id_mapping[ext_token_id] == -1:
                print(f"token id {ext_token_id} not in decomposition map")
                self.token_id_to_base_id_mapping[ext_token_id] = base_id_to_base_vocab_idx[
                    ext_token_id
                ]

        # Base or inflection token mask
        indices = torch.tensor(base_or_inflection_token_ids).to(torch.long)
        self.token_id_is_base_or_inflection_mapping = torch.zeros(
            self.extended_vocab_size, dtype=torch.bool, device=self.embed.weight.device
        )
        self.token_id_is_base_or_inflection_mapping[indices] = True

        # Update conditioned transformation head with mapping
        if hasattr(
            self.conditioned_transformation_head, "base_vocab_idx_to_extended_token_id_mapping"
        ):
            # Replace existing buffer if already present
            self.conditioned_transformation_head._buffers[
                "base_vocab_idx_to_extended_token_id_mapping"
            ] = self.base_vocab_idx_to_extended_token_id_mapping.contiguous()
        else:
            # First time setting it
            self.conditioned_transformation_head.register_buffer(
                "base_vocab_idx_to_extended_token_id_mapping",
                self.base_vocab_idx_to_extended_token_id_mapping.contiguous(),
                persistent=False,
            )

    def freeze_rows_in_type_embeddings(self, rows_to_freeze, freeze_embed=True, freeze_head=False):
        rows_to_freeze = torch.Tensor(list(rows_to_freeze)).to(torch.long)

        # Zero out the weights for frozen rows
        with torch.no_grad():
            if freeze_embed:
                self.embed_types.weight[rows_to_freeze] = 0
            if freeze_head:
                # Zero out weights in the conditioned transformation head
                if hasattr(self.conditioned_transformation_head, "projection"):
                    self.conditioned_transformation_head.projection.weight[:, rows_to_freeze] = 0
                elif hasattr(self.conditioned_transformation_head, "output_proj"):
                    self.conditioned_transformation_head.output_proj.weight[:, rows_to_freeze] = 0

        def _mask_grads(grad):
            grad[rows_to_freeze] = 0
            return grad

        if freeze_embed and self.embed_types.weight.requires_grad:
            self.embed_types.weight.register_hook(_mask_grads)

        if freeze_head:
            # Add hooks to the conditioned transformation head weights
            if hasattr(self.conditioned_transformation_head, "projection"):
                if self.conditioned_transformation_head.projection.weight.requires_grad:

                    def _mask_grads_head(grad):
                        grad[:, rows_to_freeze] = 0
                        return grad

                    self.conditioned_transformation_head.projection.weight.register_hook(
                        _mask_grads_head
                    )
            elif hasattr(self.conditioned_transformation_head, "output_proj"):
                if self.conditioned_transformation_head.output_proj.weight.requires_grad:

                    def _mask_grads_head(grad):
                        grad[:, rows_to_freeze] = 0
                        return grad

                    self.conditioned_transformation_head.output_proj.weight.register_hook(
                        _mask_grads_head
                    )

    def set_token_to_type_ids(
        self, decomposition_map: Dict[int, Tuple[int, List[int]]], negative_type_ids=None
    ):
        self._checkpoint_decomposition = {
            "map": decomposition_map,
            "negative_type_ids": negative_type_ids,
        }

        def _collapse_type_ids(type_ids: List[int]) -> List[int]:
            if not self.collapse_na_types or not type_ids:
                return type_ids
            collapsed = [self._na_to_no_map.get(t, t) for t in type_ids]
            return list(dict.fromkeys(collapsed))

        def _normalize_type_ids(type_ids):
            if type_ids is None or isinstance(type_ids, str):
                return []
            if torch.is_tensor(type_ids):
                type_ids = type_ids.tolist()
            try:
                size = len(type_ids)
            except TypeError:
                return []
            if size == 0:
                return []

            total_types = self.type_config.total_types
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
                if is_one_hot:
                    return [i for i, v in enumerate(type_ids) if v > 0.5]

            return [int(v.item() if hasattr(v, "item") else v) for v in type_ids]

        with torch.no_grad():
            self.token_to_type_ids.fill_(0)
            default_type_ids = negative_type_ids
            if self.collapse_na_types:
                default_type_ids = self._no_type_ids
            if default_type_ids is not None:
                if isinstance(default_type_ids, set):
                    default_type_ids = list(default_type_ids)
                self.token_to_type_ids[:, default_type_ids] = 1

            # Pre-compute NA type mask for efficient loss computation
            # Shape: [extended_vocab_size, num_groups] - True where token has NA type in that group
            num_groups = len(self.type_config.type_groups)
            na_type_mask = torch.zeros(
                self.extended_vocab_size,
                num_groups,
                dtype=torch.bool,
                device=self.token_to_type_ids.device,
            )

            # Verify that NA types are last in each group (as per assumption)
            for group_idx, (group_name, group_size) in enumerate(
                self.type_config.type_groups.items()
            ):
                start, end = self.type_config.offsets[group_name]
                na_type_idx = end - 1  # NA type should be last index in group
                if not self.collapse_na_types and negative_type_ids is not None:
                    assert na_type_idx in negative_type_ids, (
                        f"NA type {na_type_idx} not found in negative_type_ids for group {group_name}"
                    )

            normalized_decomposition_map = {}
            for token_id, (base_token_id, type_ids) in decomposition_map.items():
                type_ids = _normalize_type_ids(type_ids)
                type_ids = _collapse_type_ids(type_ids)
                normalized_decomposition_map[token_id] = (base_token_id, type_ids)
                self.token_to_type_ids[token_id, :] = 0
                self.token_to_type_ids[token_id, type_ids] = 1

                # Compute NA mask for this token across all groups
                for group_idx, (group_name, group_size) in enumerate(
                    self.type_config.type_groups.items()
                ):
                    start, end = self.type_config.offsets[group_name]
                    na_type_idx = end - 1  # Last index in group is NA type

                    # Check if this token has the NA type for this group
                    if na_type_idx in type_ids:
                        na_type_mask[token_id, group_idx] = True

            # Register the NA mask as a buffer for efficient access during training
            self.register_buffer("na_type_mask", na_type_mask, persistent=False)

            # Update conditioned transformation head with decomposition map
            self.decomposition_map = normalized_decomposition_map
            self.conditioned_transformation_head.set_decomposition_map(normalized_decomposition_map)

    def _build_modifier_map_tensor(self, modifier_map_dict: dict):
        """Build tensor for fast modifier_id → transform_indices lookup.

        Args:
            modifier_map_dict: Dictionary from ModifierMap.to_dict()
        """
        modifier_to_transforms = modifier_map_dict["modifier_to_transforms"]

        # Find max modifier_id
        max_modifier_id = max(int(k) for k in modifier_to_transforms.keys())
        num_groups = len(self.new_transform_group_names)

        # Create tensor: [max_modifier_id + 1, num_groups]
        modifier_tensor = torch.zeros((max_modifier_id + 1, num_groups), dtype=torch.long)

        # Fill in the mappings
        for modifier_id_str, transform_tuple in modifier_to_transforms.items():
            modifier_id = int(modifier_id_str)
            for group_idx, transform_idx in enumerate(transform_tuple):
                modifier_tensor[modifier_id, group_idx] = transform_idx

        # Register as buffer
        self.register_buffer("modifier_map_tensor", modifier_tensor, persistent=False)

    def map_inputs_to_base_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_id_to_base_id_mapping is None:
            return tokens

        # Use registered buffer instead of creating new tensors
        if not hasattr(self, "_token_id_to_base_id_mapping_tensor"):
            # Size mapping tensor to cover all possible token IDs
            max_token_id = max(len(self.token_id_to_base_id_mapping), self.extended_vocab_size)
            mapping_tensor = torch.full((max_token_id,), -1, dtype=torch.long, device=tokens.device)
            for ext_id, base_id in enumerate(self.token_id_to_base_id_mapping):
                if base_id != -1:
                    mapping_tensor[ext_id] = base_id
            # For any unmapped tokens, map to themselves
            for i in range(len(mapping_tensor)):
                if mapping_tensor[i] == -1:
                    mapping_tensor[i] = i
            # Register as buffer so it moves with the model
            self.register_buffer(
                "_token_id_to_base_id_mapping_tensor", mapping_tensor.contiguous(), persistent=False
            )

        return self._token_id_to_base_id_mapping_tensor[tokens]

    def map_inputs_to_is_base_or_inflection_token(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_id_is_base_or_inflection_mapping is None:
            return torch.ones_like(tokens, dtype=torch.bool)

        if not hasattr(self, "_token_id_is_base_or_inflection_mapping_tensor"):
            # Register as buffer so it moves with the model
            self.register_buffer(
                "_token_id_is_base_or_inflection_mapping_tensor",
                self.token_id_is_base_or_inflection_mapping.to(tokens.device).contiguous(),
                persistent=False,
            )

        return self._token_id_is_base_or_inflection_mapping_tensor[tokens]

    def map_modifier_ids_to_type_ids(self, modifier_ids: torch.Tensor) -> torch.Tensor:
        """Map modifier IDs to per-group transformation indices for new groups.

        Args:
            modifier_ids: [seq_len] scalar modifier IDs from tokenization (1D format)
                         OR [seq_len, num_groups] group-relative indices (2D format)

        Returns:
            type_ids: [seq_len, num_new_transform_groups] transformation indices
        """
        # Handle 2D modifier array (new format)
        if modifier_ids.dim() == 2:
            # Already have group-relative indices, just return as-is
            # The tensor is already [seq_len, num_groups]
            return modifier_ids.long()

        # Handle 1D modifier array (old format) - need lookup
        if not hasattr(self, "modifier_map_tensor") or self.new_transform_group_names is None:
            # No modifier map available - return zeros
            return torch.zeros(
                modifier_ids.shape[0], 0, dtype=torch.long, device=modifier_ids.device
            )

        # Lookup transforms for each modifier_id
        # modifier_map_tensor: [max_modifier_id, num_new_groups]
        return self.modifier_map_tensor[modifier_ids]

    def merge_type_ids(
        self, old_type_ids: torch.Tensor, new_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """Merge old transformation IDs (from token_id) with new IDs (from modifier_ids).

        Args:
            old_type_ids: [seq_len, total_num_transforms] from token_to_type_ids lookup
            new_type_ids: [seq_len, num_new_groups] from modifier_ids lookup

        Returns:
            merged_type_ids: [seq_len, total_num_transforms] with new groups updated
        """
        if new_type_ids.shape[1] == 0 or self.new_transform_group_indices is None:
            # No new transforms to merge
            return old_type_ids

        # Clone to avoid modifying original
        merged = old_type_ids.clone()

        # Check if we have unified modifiers (all groups) or just new groups
        num_modifier_groups = new_type_ids.shape[1]

        if hasattr(self, "unified_modifier_groups") and self.unified_modifier_groups is not None:
            # Unified modifiers: new_type_ids covers ALL transformation groups
            # The modifier indices correspond to unified_modifier_groups order
            for group_idx, group_name in enumerate(self.unified_modifier_groups):
                if group_idx < num_modifier_groups:
                    # Find this group in the type_config
                    if (
                        hasattr(self.type_config, "get_slice")
                        and group_name in self.type_config.type_groups
                    ):
                        group_slice = self.type_config.get_slice(group_name)
                        start_idx = group_slice.start
                        end_idx = group_slice.stop
                    elif group_name in self.new_transform_group_indices:
                        start_idx, end_idx = self.new_transform_group_indices[group_name]
                    else:
                        continue

                    # Zero out the old values in this group
                    merged[:, start_idx:end_idx] = 0
                    # Set the new value (one-hot encoding)
                    transform_idx = new_type_ids[:, group_idx]
                    if self.collapse_na_types:
                        group_size = end_idx - start_idx
                        if group_size > 1:
                            na_rel = group_size - 1
                            transform_idx = torch.where(
                                transform_idx == na_rel,
                                torch.zeros_like(transform_idx),
                                transform_idx,
                            )
                    # Add start_idx offset to get global transform index
                    global_transform_idx = transform_idx + start_idx
                    # Clamp to valid range
                    global_transform_idx = torch.clamp(global_transform_idx, start_idx, end_idx - 1)
                    merged[
                        torch.arange(merged.shape[0], device=merged.device),
                        global_transform_idx.long(),
                    ] = 1
        else:
            # Original behavior: only update new transform groups
            for group_idx, group_name in enumerate(self.new_transform_group_names):
                if group_name in self.new_transform_group_indices:
                    start_idx, end_idx = self.new_transform_group_indices[group_name]
                    # Zero out the old values in this group
                    merged[:, start_idx:end_idx] = 0
                    # Set the new value (one-hot encoding)
                    transform_idx = new_type_ids[:, group_idx]
                    if self.collapse_na_types:
                        group_size = end_idx - start_idx
                        if group_size > 1:
                            na_rel = group_size - 1
                            transform_idx = torch.where(
                                transform_idx == na_rel,
                                torch.zeros_like(transform_idx),
                                transform_idx,
                            )
                    # Add start_idx offset to get global transform index
                    global_transform_idx = transform_idx + start_idx
                    merged[
                        torch.arange(merged.shape[0], device=merged.device),
                        global_transform_idx.long(),
                    ] = 1

        return merged

    def compute_type_losses(
        self, type_labels: torch.Tensor, type_logits: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute type losses using ignore_index for better efficiency"""
        total_loss = 0.0
        num_groups = len(self.type_config.type_groups)

        for group_idx, group_name in enumerate(self.type_config.type_groups.keys()):
            group_slice = self.type_config.get_slice(group_name)
            group_labels = type_labels[..., group_slice].argmax(-1)  # [seq_len]
            group_logits = type_logits[..., group_slice]  # [seq_len, group_size]

            if self.collapse_na_types:
                group_size = group_slice.stop - group_slice.start
                if group_size > 1:
                    na_idx = group_size - 1
                    group_labels = group_labels.clone()
                    group_labels[group_labels == na_idx] = 0

            # Optionally ignore positions where ground truth has NA type (last index in group)
            if self.ignore_na_types:
                group_labels = group_labels.clone()
                group_size = group_slice.stop - group_slice.start
                na_positions = group_labels == group_size - 1  # NA type is last index
                group_labels[na_positions] = -100  # Ignore NA positions

            group_logits_flat = group_logits.view(-1, group_logits.size(-1))
            group_labels_flat = group_labels.view(-1)
            valid_mask = group_labels_flat != -100

            if valid_mask.any():
                allowed = None
                if (
                    self.ambiguous_type_loss_mode in ("equal", "favor_merge")
                    or self.collapse_na_types
                ):
                    allowed_map = getattr(self, "_group_allowed", None)
                    if allowed_map is None or group_name not in allowed_map:
                        allowed_map = self._get_group_allowed(group_logits.device)
                    allowed = allowed_map.get(group_name)

                if allowed is not None:
                    valid_logits = group_logits_flat[valid_mask]
                    valid_labels = group_labels_flat[valid_mask]
                    allowed_for_label = allowed[valid_labels]
                    masked_logits = valid_logits.masked_fill(~allowed_for_label, -1e9)
                    logsumexp_allowed = torch.logsumexp(masked_logits, dim=-1)
                    logsumexp_all = torch.logsumexp(valid_logits, dim=-1)
                    group_loss = (
                        logsumexp_all - logsumexp_allowed
                    ).sum() / group_labels_flat.numel()
                    if (
                        self.ambiguous_type_loss_mode == "favor_merge"
                        and self.ambiguous_type_loss_lambda > 0.0
                    ):
                        merged_map = getattr(self, "_group_merged", None)
                        if merged_map is None or group_name not in merged_map:
                            merged_map = self._get_group_merged(group_logits.device)
                        merged_mask = merged_map.get(group_name)
                        if merged_mask is not None:
                            merged_labels_mask = merged_mask[valid_labels]
                            if merged_labels_mask.any():
                                merged_logits = valid_logits[merged_labels_mask]
                                merged_labels = valid_labels[merged_labels_mask]
                                strict_loss = F.cross_entropy(
                                    merged_logits, merged_labels, reduction="mean"
                                )
                                group_loss = (
                                    group_loss + self.ambiguous_type_loss_lambda * strict_loss
                                )
                else:
                    group_loss = (
                        F.cross_entropy(
                            group_logits_flat, group_labels_flat, ignore_index=-100, reduction="sum"
                        )
                        / group_labels_flat.numel()
                    )
            else:
                group_loss = torch.tensor(0.0, device=group_logits.device)

            total_loss += group_loss

        return total_loss / num_groups

    def _get_group_allowed(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Build group-level compatibility masks for ambiguous (A+B) labels."""
        cached = getattr(self, "_group_allowed", None)
        cache_device = getattr(self, "_group_allowed_device", None)
        if cached is not None and cache_device == device:
            return cached

        allowed_map: Dict[str, torch.Tensor] = {}
        name_to_idx = {v: k for k, v in self.type_config.transformation_int_to_names.items()}
        for group_name in self.type_config.type_groups.keys():
            sl = self.type_config.get_slice(group_name)
            group_size = sl.stop - sl.start
            allowed = torch.eye(group_size, dtype=torch.bool, device=device)
            prefix = f"{group_name}_"
            for rel_label in range(group_size):
                label_idx = sl.start + rel_label
                label_name = self.type_config.transformation_int_to_names.get(label_idx, "")
                if not label_name.startswith(prefix):
                    continue
                value = label_name[len(prefix) :]
                if "+" not in value:
                    continue
                for part in value.split("+"):
                    comp_name = prefix + part
                    comp_idx = name_to_idx.get(comp_name)
                    if comp_idx is None:
                        continue
                    comp_rel = comp_idx - sl.start
                    if 0 <= comp_rel < group_size:
                        allowed[rel_label, comp_rel] = True
            if self.collapse_na_types and group_size > 1:
                na_rel = group_size - 1
                allowed[0, na_rel] = True
                allowed[na_rel, 0] = True
            allowed_map[group_name] = allowed

        self._group_allowed = allowed_map
        self._group_allowed_device = device
        return allowed_map

    def _get_group_merged(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Build group-level masks for merged (A+B) labels."""
        cached = getattr(self, "_group_merged", None)
        cache_device = getattr(self, "_group_merged_device", None)
        if cached is not None and cache_device == device:
            return cached

        merged_map: Dict[str, torch.Tensor] = {}
        for group_name in self.type_config.type_groups.keys():
            sl = self.type_config.get_slice(group_name)
            group_size = sl.stop - sl.start
            merged = torch.zeros(group_size, dtype=torch.bool, device=device)
            prefix = f"{group_name}_"
            for rel_label in range(group_size):
                label_idx = sl.start + rel_label
                label_name = self.type_config.transformation_int_to_names.get(label_idx, "")
                if not label_name.startswith(prefix):
                    continue
                value = label_name[len(prefix) :]
                if "+" in value:
                    merged[rel_label] = True
            merged_map[group_name] = merged

        self._group_merged = merged_map
        self._group_merged_device = device
        return merged_map

    def orthogonality_loss(self, x_base: Tensor, x_type: Tensor) -> Tensor:
        """Compute loss to encourage orthogonal outputs x_base ⊥ x_type"""
        # Normalize the representations
        x_base_norm = F.normalize(x_base, dim=-1)  # [batch*seq, dim]
        x_type_norm = F.normalize(x_type, dim=-1)  # [batch*seq, dim]

        if x_base.shape[-1] == x_type.shape[-1]:
            # Same dimensions: use direct orthogonality <x_base, x_type> ≈ 0
            dot_products = (x_base_norm * x_type_norm).sum(dim=-1)  # [batch*seq]
            return dot_products.pow(2).sum()
        else:
            # Different dimensions: use cross-correlation matrix
            cross_corr = x_base_norm.T @ x_type_norm / x_base_norm.shape[0]
            return torch.norm(cross_corr, p="fro") ** 2

    def _regular_cross_entropy_from_hidden(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        shift: int = 0,
    ) -> torch.Tensor:
        logits = F.linear(hidden_states, weight)
        if self.softcap > 0 and self.tanh_softcap is not None:
            logits = self.tanh_softcap(logits)
        if shift > 0:
            logits = logits[:-shift]
            targets = targets[shift:]
        return F.cross_entropy(logits.float(), targets, reduction="mean", ignore_index=-1)

    def forward(
        self,
        input_seq: Tensor,
        target_seq: Optional[Tensor],
        sliding_window_num_blocks: Tensor,
        compute_aux_logits: bool = False,
        return_top1_extended: bool = False,
        modifier_ids: Optional[Tensor] = None,
        target_modifier_ids: Optional[Tensor] = None,
    ) -> CompositionalOutputs:
        """
        Simplified forward pass: compute base and type predictions separately, no extended vocab mapping.

        Args:
            input_seq: [seq_len] input token IDs in extended vocab space
            target_seq: [seq_len] target token IDs (None for inference)
            sliding_window_num_blocks: Sliding window configuration
            compute_aux_logits: Whether to compute auxiliary head logits
            return_top1_extended: Whether to compose and return top-1 extended token prediction
            modifier_ids: [seq_len] optional modifier IDs for new transformation groups (dual-stream mode)
            target_modifier_ids: [seq_len] optional modifier IDs for targets (dual-stream mode)

        Returns:
            CompositionalOutputs with base_logits, type_logits, losses, and optionally top1_extended_idx
        """
        assert input_seq.ndim == 1

        # Pre-compute mappings
        base_inputs = self.map_inputs_to_base_ids(input_seq)
        old_type_ids = self.token_to_type_ids[input_seq].to(self.embed_types.weight.dtype)

        # Handle modifier_ids for dual-stream mode
        if modifier_ids is not None:
            # Validate modifier dimensions
            if modifier_ids.dim() == 2 and self.unified_modifier_groups is not None:
                expected_groups = len(self.unified_modifier_groups)
                actual_groups = modifier_ids.shape[1]
                if actual_groups != expected_groups:
                    raise ValueError(
                        f"Modifier array has {actual_groups} groups but model expects {expected_groups}"
                    )
            new_type_ids = self.map_modifier_ids_to_type_ids(modifier_ids)
            type_ids = self.merge_type_ids(old_type_ids, new_type_ids)
        else:
            type_ids = old_type_ids

        # Get embeddings and combine
        base_embeds = self.embed(base_inputs)[None]
        type_embeds = torch.matmul(type_ids, self.embed_types.weight)[None]
        x = norm(self.embedding_combiner(base_embeds, type_embeds), self.embedding_norm)

        # Run through transformer blocks
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm if i % 4 in [0, 3] else short_bm for i in range(self.num_layers)]

        for i in range(self.num_layers):
            x = self.blocks[i](x, block_masks[i])

        x_flat = x.flatten(0, -2)

        # Normalize hidden states
        x_flat = norm(x_flat, self.final_norm)
        x_base, x_type = x_flat, x_flat

        base_logits = type_logits = type_logits_pred_base = aux_logits = top1_extended_idx = None
        # Only compute full logits when we need to (i.e., we don't need them for training)
        if target_seq is None or (target_seq is not None and not self.training):
            # Compute base logits
            base_logits = F.linear(x_base, self.lm_head.weight[: self.base_vocab_size_unpadded])
            if self.softcap > 0:
                base_logits = self.tanh_softcap(base_logits)

            # Compute type logits
            # use model's base token prediction
            # Need top1_extended for metrics during training or when explicitly requested
            top1_base_indices = base_logits.argmax(dim=-1)
            type_logits_pred_base, top1_extended_idx = self.conditioned_transformation_head.forward(
                x_type, top1_base_indices, return_top1_extended=True
            )
            type_logits = type_logits_pred_base

            if self.softcap > 0:
                type_logits = self.tanh_softcap(type_logits)

            # Compute auxiliary logits if requested
            aux_logits = None
            pass

        # If target_seq is None, return logits for inference/stats collection without computing loss
        if target_seq is None:
            return CompositionalOutputs(
                loss=None,
                logits=None,
                base_logits=base_logits,
                type_logits=type_logits,
                type_logits_pred_base=type_logits_pred_base,
                aux_logits=aux_logits,
                base_loss=None,
                type_loss=None,
                aux_loss=None,
                top1_extended_idx=top1_extended_idx,
            )

        # Training mode: compute losses
        base_loss = type_loss = aux_loss = None
        total_loss = 0.0

        # Compute base loss
        if self.base_loss_alpha > 0.0:
            base_targets = self.map_inputs_to_base_ids(target_seq)
            if self.use_linear_cross_entropy:
                base_loss = linear_cross_entropy(
                    x_base,
                    self.lm_head.weight[: self.base_vocab_size_unpadded],
                    base_targets,
                    softcap=self.softcap,
                    reduction="mean",
                    shift=1 if self.use_shift else 0,
                )
            else:
                base_loss = self._regular_cross_entropy_from_hidden(
                    x_base,
                    self.lm_head.weight[: self.base_vocab_size_unpadded],
                    base_targets,
                    shift=1 if self.use_shift else 0,
                )
            total_loss += self.base_loss_alpha * base_loss

        if self.type_loss_alpha > 0.0 or not self.training:
            top1_base_indices = self.map_inputs_to_base_ids(target_seq)
            type_logits_oracle = self.conditioned_transformation_head.forward(
                x_type, top1_base_indices, return_top1_extended=False
            )
            type_logits = type_logits_oracle

            if self.type_loss_alpha > 0.0:
                shift_amount = 1 if self.use_shift else 0
                if shift_amount > 0:
                    target_old_type_ids = self.token_to_type_ids[target_seq[shift_amount:]]
                    # Handle target modifier_ids for dual-stream mode
                    if target_modifier_ids is not None:
                        target_new_type_ids = self.map_modifier_ids_to_type_ids(
                            target_modifier_ids[shift_amount:]
                        )
                        type_labels = self.merge_type_ids(target_old_type_ids, target_new_type_ids)
                    else:
                        type_labels = target_old_type_ids
                    pred_type_logits = type_logits[:-shift_amount]
                else:
                    target_old_type_ids = self.token_to_type_ids[target_seq]
                    # Handle target modifier_ids for dual-stream mode
                    if target_modifier_ids is not None:
                        target_new_type_ids = self.map_modifier_ids_to_type_ids(target_modifier_ids)
                        type_labels = self.merge_type_ids(target_old_type_ids, target_new_type_ids)
                    else:
                        type_labels = target_old_type_ids
                    pred_type_logits = type_logits

                type_loss = self.compute_type_losses(type_labels, pred_type_logits)
                total_loss += self.type_loss_alpha * type_loss

        # Compute auxiliary loss
        pass
        return CompositionalOutputs(
            loss=total_loss,
            logits=None,
            base_logits=base_logits,
            type_logits=type_logits,
            type_logits_pred_base=type_logits_pred_base,
            aux_logits=aux_logits,
            base_loss=base_loss,
            type_loss=type_loss,
            aux_loss=aux_loss,
            top1_extended_idx=top1_extended_idx,
        )
