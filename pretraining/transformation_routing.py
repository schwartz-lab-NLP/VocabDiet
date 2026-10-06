"""
Multi-Expert Transformation Routing for Compositional Token Prediction

This module implements a routing mechanism that allows the model to use different
transformation experts for different base tokens, enabling the model to express
multiple distinct transformation profiles within a single prediction.

Key idea:
- Multiple expert transformation heads, each computing P(transform|context)
- Router selects which expert(s) to use for each base token
- Per-base routing: each base token gets its own expert mixture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RoutedTransformationOutput:
    """
    Output from RoutedTransformationHead.forward().

    Attributes:
        type_logits: Transformation type logits.
                    Shape: [batch, base_vocab, type_vocab] with no pruning
                           [batch, M, type_vocab] with top-M pruning
        selected_indices: Optional indices of selected bases when using top-M pruning.
                         Shape: [batch, M] or None
        load_balance_loss: Routing load balancing loss (scalar tensor, 0.0 if disabled)
        diversity_loss: Expert diversity regularization loss (scalar tensor, 0.0 if disabled)
    """

    type_logits: torch.Tensor
    selected_indices: Optional[torch.Tensor] = None
    load_balance_loss: Optional[torch.Tensor] = None
    diversity_loss: Optional[torch.Tensor] = None


def gumbel_softmax(
    logits: torch.Tensor, temperature: float = 1.0, hard: bool = False, dim: int = -1
) -> torch.Tensor:
    """
    Gumbel-softmax sampling for differentiable expert selection.

    Args:
        logits: [*, num_experts] routing logits
        temperature: Sampling temperature (lower = more discrete)
        hard: If True, use straight-through estimator for discrete selection
        dim: Dimension to apply softmax

    Returns:
        [*, num_experts] expert weights (probabilities or one-hot)
    """
    if temperature <= 0:
        # Hard argmax without Gumbel noise
        indices = logits.argmax(dim=dim, keepdim=True)
        return F.one_hot(indices.squeeze(dim), num_classes=logits.shape[dim]).float()

    # Sample Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

    # Add noise and apply softmax with temperature
    y_soft = F.softmax((logits + gumbel_noise) / temperature, dim=dim)

    if hard:
        # Straight-through estimator: forward with one-hot, backward with soft
        indices = y_soft.argmax(dim=dim, keepdim=True)
        y_hard = F.one_hot(indices.squeeze(dim), num_classes=logits.shape[dim]).float()
        # Straight-through: same forward value as one-hot, but gradient from soft
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


class BaseRouter(nn.Module, ABC):
    """
    Abstract base class for transformation routing mechanisms.

    Provides common functionality for managing base token keys, expert selection,
    and load balancing. Subclasses implement specific routing computation logic
    via compute_routing_logits().

    IMPORTANT: When creating new router subclasses or modifying parameter structures:
    - Update parameter collection code in train_compositional.py
    - Search for "Add expert queries" to find the relevant section
    """

    def __init__(
        self,
        base_vocab_size: int,
        num_experts: int,
        model_dim: int,
        key_dim: int = None,
        top_k: int = 1,
        temperature: float = 0.0,
        load_balance_alpha: float = 0.01,
        use_projected_keys: bool = False,
        projected_keys_source: str = "lm_head",
    ):
        """
        Args:
            base_vocab_size: Number of base tokens
            num_experts: Number of transformation experts
            model_dim: Hidden dimension of the model
            key_dim: Dimension of key/query space (defaults to model_dim)
            top_k: Number of experts to use per base (1 = hard routing)
            temperature: Temperature for Gumbel-softmax (0 = hard argmax)
            load_balance_alpha: Weight for load balancing loss
            use_projected_keys: If True, project base embeddings/unembeddings to create keys
            projected_keys_source: Source for projected keys - "lm_head" or "embed"
        """
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.num_experts = num_experts
        self.model_dim = model_dim
        self.key_dim = key_dim or model_dim
        self.top_k = top_k
        self.temperature = temperature
        self.load_balance_alpha = load_balance_alpha
        self.use_projected_keys = use_projected_keys
        self.projected_keys_source = projected_keys_source

        # Base token keys: either learned directly or projected from embeddings/unembeddings
        if use_projected_keys:
            # Learn projection from base embeddings/unembeddings to key space
            self.base_key_projection = nn.Linear(model_dim, self.key_dim, bias=False)

            # Initialize as identity if dimensions match (preserves input magnitudes)
            if self.key_dim == model_dim:
                nn.init.eye_(self.base_key_projection.weight)

            # Buffer to cache precomputed keys during inference
            self.register_buffer("cached_base_keys", None, persistent=False)
        else:
            # Learnable base token keys (original behavior)
            self.base_keys = nn.Parameter(
                torch.randn(base_vocab_size, self.key_dim) * (self.key_dim**-0.5)
            )

        # For load balancing loss tracking
        self.register_buffer("expert_usage_count", torch.zeros(num_experts))

    def get_base_keys(self, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get base keys - either cached, projected, or learned directly.

        Args:
            base_weight: [base_vocab_size, model_dim] base embedding or unembedding weights
                        (required if use_projected_keys=True and not cached)

        Returns:
            base_keys: [base_vocab_size, key_dim] base token keys for routing
        """
        if self.use_projected_keys:
            if self.cached_base_keys is not None:
                # Use precomputed keys (inference mode)
                return self.cached_base_keys
            else:
                # Compute dynamically from base embeddings/unembeddings (training mode)
                assert base_weight is not None, (
                    "base_weight required for projected keys when not cached"
                )
                return self.base_key_projection(base_weight)
        else:
            # Use learned base_keys (original behavior)
            return self.base_keys

    def precompute_keys(self, base_weight: torch.Tensor):
        """
        Precompute and cache base keys for inference mode.

        Call this before inference to avoid recomputing projection at each forward pass.

        Args:
            base_weight: [base_vocab_size, model_dim] base embedding or unembedding weights
        """
        if self.use_projected_keys:
            with torch.no_grad():
                self.cached_base_keys = self.base_key_projection(base_weight)

    def clear_cached_keys(self):
        """Clear cached keys to return to training mode (dynamic projection)."""
        if self.use_projected_keys:
            self.cached_base_keys = None

    @abstractmethod
    def compute_routing_logits(
        self, hidden_states: torch.Tensor, base_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute routing logits from hidden states and base keys.

        This is the core routing computation that differs between router types.

        Args:
            hidden_states: [batch_size, model_dim] context hidden states
            base_keys: [batch_size, M (or base_vocab), key_dim] base token keys

        Returns:
            routing_logits: [batch_size, M (or base_vocab), num_experts] routing scores
        """
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_m_bases: Optional[torch.Tensor] = None,
        base_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights for each base token.

        Args:
            hidden_states: [batch_size, model_dim] context hidden states
            top_m_bases: Optional[batch_size, M] indices of top-M base candidates for pruning
            base_weight: Optional[base_vocab_size, model_dim] base embedding or unembedding weights
                        (required if use_projected_keys=True and not cached)

        Returns:
            routing_weights: [batch_size, base_vocab (or M), num_experts] routing weights
            routing_logits: [batch_size, base_vocab (or M), num_experts] routing logits (for loss)
        """
        batch_size = hidden_states.shape[0]

        # 1. Get all base keys (cached, projected, or learned)
        all_base_keys = self.get_base_keys(base_weight)  # [base_vocab_size, key_dim]

        # 2. Select base keys (all or top-M)
        if top_m_bases is not None:
            # Pruning: only compute routing for top-M bases
            # top_m_bases: [batch_size, M]
            base_keys = all_base_keys[top_m_bases]  # [batch_size, M, key_dim]
        else:
            # No pruning: compute for all bases
            base_keys = all_base_keys.unsqueeze(0).expand(batch_size, -1, -1)
            # [batch_size, base_vocab, key_dim]

        # 3. Compute routing logits (subclass-specific)
        routing_logits = self.compute_routing_logits(hidden_states, base_keys)
        # routing_logits: [batch_size, base_vocab (or M), num_experts]

        # 4. Apply routing strategy (top-k or soft mixing)
        if self.top_k == 1 and self.training:
            # Hard routing with Gumbel-softmax for training
            routing_weights = gumbel_softmax(
                routing_logits, temperature=self.temperature, hard=True, dim=-1
            )
        elif self.top_k == 1:
            # Hard routing for inference (no Gumbel noise)
            indices = routing_logits.argmax(dim=-1, keepdim=True)
            routing_weights = F.one_hot(indices.squeeze(-1), num_classes=self.num_experts).float()
        elif self.top_k >= self.num_experts:
            # Soft mixing of all experts
            routing_weights = F.softmax(routing_logits, dim=-1)
        else:
            # Top-k soft mixing
            top_k_logits, top_k_indices = torch.topk(routing_logits, k=self.top_k, dim=-1)
            routing_weights = torch.zeros_like(routing_logits)
            routing_weights.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))

        return routing_weights, routing_logits

    def compute_load_balancing_loss(self, routing_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage balanced expert usage.

        Uses the standard MoE load balancing formulation:
        - Compute fraction of total probability mass assigned to each expert
        - Penalize variance from uniform distribution

        Args:
            routing_logits: [batch_size, base_vocab (or M), num_experts] routing logits from forward pass

        Returns:
            Scalar load balancing loss
        """
        if self.load_balance_alpha <= 0.0:
            return torch.tensor(0.0, device=routing_logits.device)

        # Compute routing probabilities
        routing_probs = F.softmax(routing_logits, dim=-1)
        # [batch_size, base_vocab (or M), num_experts]

        # Compute fraction of total probability mass going to each expert
        # Sum probabilities across all (batch, base) positions, then normalize
        num_positions = routing_probs.shape[0] * routing_probs.shape[1]  # batch_size * num_bases
        expert_usage = routing_probs.sum(dim=(0, 1)) / num_positions  # [num_experts]

        # Synchronize expert_usage across GPUs for consistent load balance loss
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(expert_usage, op=dist.ReduceOp.AVG)

        # Load balancing: penalize squared deviation from uniform distribution
        uniform_usage = 1.0 / self.num_experts
        load_balance_loss = self.load_balance_alpha * torch.sum((expert_usage - uniform_usage) ** 2)
        return load_balance_loss


class TransformationRouter(BaseRouter):
    """
    Routes base tokens to transformation experts using key-query mechanism.

    Architecture:
    - Base keys: Learnable embeddings for each base token [base_vocab, key_dim]
    - Expert queries: Per-expert projections of hidden state [num_experts, key_dim]
    - Routing: R[b,e] = base_key[b] · query[e]
    - Mixing: α[b,e] = softmax_e(R[b,:])
    """

    def __init__(
        self,
        base_vocab_size: int,
        num_experts: int,
        model_dim: int,
        key_dim: int = None,
        top_k: int = 1,
        temperature: float = 0.0,
        load_balance_alpha: float = 0.01,
        use_projected_keys: bool = False,
        projected_keys_source: str = "lm_head",
    ):
        """
        Args:
            base_vocab_size: Number of base tokens
            num_experts: Number of transformation experts
            model_dim: Hidden dimension of the model
            key_dim: Dimension of key/query space (defaults to model_dim)
            top_k: Number of experts to use per base (1 = hard routing)
            temperature: Temperature for Gumbel-softmax (0 = hard argmax)
            load_balance_alpha: Weight for load balancing loss
            use_projected_keys: If True, project base embeddings/unembeddings to create keys instead of learning them directly
            projected_keys_source: Source for projected keys - "lm_head" (unembeddings) or "embed" (embeddings)
        """
        # Initialize base class with common parameters
        super().__init__(
            base_vocab_size=base_vocab_size,
            num_experts=num_experts,
            model_dim=model_dim,
            key_dim=key_dim,
            top_k=top_k,
            temperature=temperature,
            load_balance_alpha=load_balance_alpha,
            use_projected_keys=use_projected_keys,
            projected_keys_source=projected_keys_source,
        )

        # Per-expert query projections (specific to TransformationRouter)
        # NOTE: If you modify this parameter structure, update the parameter collection
        # code in train_compositional.py (search for "Add expert queries")
        self.expert_queries = nn.ParameterList(
            [
                nn.Parameter(torch.randn(model_dim, self.key_dim) * (model_dim**-0.5))
                for _ in range(num_experts)
            ]
        )

    def compute_routing_logits(
        self, hidden_states: torch.Tensor, base_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute routing logits using query-key attention mechanism.

        Projects hidden states through per-expert query matrices and computes
        dot product with base keys.

        Args:
            hidden_states: [batch_size, model_dim] context hidden states
            base_keys: [batch_size, M (or base_vocab), key_dim] base token keys

        Returns:
            routing_logits: [batch_size, M (or base_vocab), num_experts] routing scores
        """
        # 1. Normalize base keys
        base_keys = F.rms_norm(base_keys, (base_keys.size(-1),))

        # 2. Compute expert queries from hidden states
        # queries: [batch_size, num_experts, key_dim]
        queries = torch.stack(
            [
                hidden_states @ expert_query  # [batch_size, key_dim]
                for expert_query in self.expert_queries
            ],
            dim=1,
        )

        # 3. Normalize queries
        queries = F.rms_norm(queries, (queries.size(-1),))

        # 4. Compute routing scores: R[b,e] = base_key[b] · query[e]
        # Using Einstein notation for clarity: batch×bases×key_dim @ batch×experts×key_dim
        routing_logits = torch.einsum("bmk,bek->bme", base_keys, queries)
        # routing_logits: [batch_size, base_vocab (or M), num_experts]

        return routing_logits


class ContextualMatchingRouter(BaseRouter):
    """
    Routes base tokens to transformation experts using contextual matching.

    Architecture:
    - Base keys: Learnable embeddings for each base token [base_vocab, key_dim]
    - Hidden projection: Projects hidden state to key_dim
    - Expert queries: Learned parameters [num_experts, 2*key_dim] (not projected)
    - Routing: Concatenate [base_keys, hidden_proj] and match against expert_queries
             R[b,e] = concat(base_key[b], hidden_proj) · expert_query[e]

    Differences from TransformationRouter:
    - Expert queries are learned parameters (not projections of hidden state)
    - Routing considers both base token identity AND context jointly
    - Expert queries operate in 2*key_dim space to capture both modalities
    """

    def __init__(
        self,
        base_vocab_size: int,
        num_experts: int,
        model_dim: int,
        key_dim: int = None,
        top_k: int = 1,
        temperature: float = 0.0,
        load_balance_alpha: float = 0.01,
        use_projected_keys: bool = False,
        projected_keys_source: str = "lm_head",
    ):
        """
        Args:
            base_vocab_size: Number of base tokens
            num_experts: Number of transformation experts
            model_dim: Hidden dimension of the model
            key_dim: Dimension of key/query space (defaults to model_dim)
            top_k: Number of experts to use per base (1 = hard routing)
            temperature: Temperature for Gumbel-softmax (0 = hard argmax)
            load_balance_alpha: Weight for load balancing loss
            use_projected_keys: If True, project base embeddings/unembeddings to create keys
            projected_keys_source: Source for projected keys - "lm_head" or "embed"
        """
        # Initialize base class with common parameters
        super().__init__(
            base_vocab_size=base_vocab_size,
            num_experts=num_experts,
            model_dim=model_dim,
            key_dim=key_dim,
            top_k=top_k,
            temperature=temperature,
            load_balance_alpha=load_balance_alpha,
            use_projected_keys=use_projected_keys,
            projected_keys_source=projected_keys_source,
        )

        # Hidden state projection to key_dim (specific to ContextualMatchingRouter)
        # NOTE: If you modify this parameter structure, update the parameter collection
        # code in train_compositional.py (search for "Add expert queries")
        self.hidden_projection = nn.Linear(model_dim, self.key_dim, bias=False)
        # Initialize with small values
        nn.init.normal_(self.hidden_projection.weight, mean=0.0, std=(model_dim**-0.5))

        # Expert queries as learned parameters [num_experts, 2*key_dim]
        # These match against concatenated [base_keys, hidden_proj]
        # NOTE: If you modify this parameter structure, update the parameter collection
        # code in train_compositional.py (search for "Add expert queries")
        self.expert_queries = nn.Parameter(
            torch.randn(num_experts, 2 * self.key_dim) * ((2 * self.key_dim) ** -0.5)
        )

    def compute_routing_logits(
        self, hidden_states: torch.Tensor, base_keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute routing logits using contextual matching mechanism.

        Concatenates base keys with projected hidden states and matches against
        learned expert query parameters.

        Args:
            hidden_states: [batch_size, model_dim] context hidden states
            base_keys: [batch_size, M (or base_vocab), key_dim] base token keys

        Returns:
            routing_logits: [batch_size, M (or base_vocab), num_experts] routing scores
        """
        batch_size = hidden_states.shape[0]
        num_bases = base_keys.shape[1]  # M or base_vocab

        # 1. Normalize base keys
        base_keys = F.rms_norm(base_keys, (base_keys.size(-1),))

        # 2. Project hidden states to key_dim
        # hidden_proj: [batch_size, key_dim]
        hidden_proj = self.hidden_projection(hidden_states)

        # 3. Normalize hidden projection
        hidden_proj = F.rms_norm(hidden_proj, (hidden_proj.size(-1),))

        # 4. Expand hidden_proj to match base dimensions
        # [batch_size, num_bases, key_dim]
        hidden_proj_expanded = hidden_proj.unsqueeze(1).expand(batch_size, num_bases, self.key_dim)

        # 5. Concatenate base_keys and hidden_proj
        # combined: [batch_size, num_bases, 2*key_dim]
        combined = torch.cat([base_keys, hidden_proj_expanded], dim=-1)

        # 6. Compute routing logits: combined @ expert_queries.T
        # combined: [batch_size, num_bases, 2*key_dim]
        # expert_queries: [num_experts, 2*key_dim]
        # Result: [batch_size, num_bases, num_experts]
        routing_logits = torch.matmul(combined, self.expert_queries.T)

        return routing_logits


class RoutedTransformationHead(nn.Module):
    """
    Multi-expert transformation prediction head with per-base routing.

    Combines multiple transformation expert heads with a router to enable
    different base tokens to use different transformation distributions.
    """

    def __init__(
        self,
        base_vocab_size: int,
        type_vocab_size: int,
        model_dim: int,
        num_experts: int = 2,
        key_dim: int = None,
        top_k: int = 1,
        temperature: float = 1.0,
        load_balance_alpha: float = 0.01,
        use_projected_keys: bool = False,
        projected_keys_source: str = "lm_head",
        router_type: str = "transformation",
        sync_top_m_bases_in_eval: bool = False,
        diversity_loss_alpha: float = 0.0,
        diversity_loss_type: str = "variance",
    ):
        """
        Args:
            base_vocab_size: Number of base tokens
            type_vocab_size: Number of transformation types
            model_dim: Hidden dimension
            num_experts: Number of transformation experts
            key_dim: Dimension of routing key space
            top_k: Number of experts to use per base
            temperature: Temperature for Gumbel-softmax
            load_balance_alpha: Weight for load balancing loss
            use_projected_keys: If True, project base embeddings/unembeddings to create routing keys
            projected_keys_source: Source for projected keys - "lm_head" (unembeddings) or "embed" (embeddings)
            router_type: Type of router to use - "transformation" or "contextual_matching"
            sync_top_m_bases_in_eval: If True, synchronize top-M base selection across GPUs during eval
            diversity_loss_alpha: Weight for expert diversity regularization loss (set to 0 to disable)
            diversity_loss_type: Type of diversity loss - "variance" (per-type variance) or "kl" (pairwise KL divergence)
        """
        super().__init__()
        self.base_vocab_size = base_vocab_size
        self.type_vocab_size = type_vocab_size
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_projected_keys = use_projected_keys
        self.projected_keys_source = projected_keys_source
        self.router_type = router_type
        self.sync_top_m_bases_in_eval = sync_top_m_bases_in_eval
        self.diversity_loss_alpha = diversity_loss_alpha
        self.diversity_loss_type = diversity_loss_type

        # Router - instantiate based on router_type
        router_kwargs = {
            "base_vocab_size": base_vocab_size,
            "num_experts": num_experts,
            "model_dim": model_dim,
            "key_dim": key_dim,
            "top_k": top_k,
            "temperature": temperature,
            "load_balance_alpha": load_balance_alpha,
            "use_projected_keys": use_projected_keys,
            "projected_keys_source": projected_keys_source,
        }

        if router_type == "transformation":
            self.router = TransformationRouter(**router_kwargs)
        elif router_type == "contextual_matching":
            self.router = ContextualMatchingRouter(**router_kwargs)
        else:
            raise ValueError(
                f"Unknown router_type: {router_type}. Must be 'transformation' or 'contextual_matching'."
            )

        # Expert transformation heads
        self.expert_heads = nn.ModuleList(
            [nn.Linear(model_dim, type_vocab_size, bias=False) for _ in range(num_experts)]
        )

    def precompute_keys(self, base_weight: torch.Tensor):
        """
        Precompute and cache routing keys for inference mode.

        Delegates to the router's precompute_keys method.

        Args:
            base_weight: [base_vocab_size, model_dim] base embedding or unembedding weights
        """
        self.router.precompute_keys(base_weight)

    def clear_cached_keys(self):
        """Clear cached routing keys to return to training mode."""
        self.router.clear_cached_keys()

    def compute_expert_diversity_loss(
        self, expert_logits: torch.Tensor, valid_type_size: int
    ) -> torch.Tensor:
        """
        Compute diversity loss to encourage experts to have different predictions.

        Measures "lack of diversity" (always >= 0, should be minimized to 0):
        - variance: How far variance is from maximum (0.25)
        - kl: How far average KL is from target (log(num_experts))

        Args:
            expert_logits: [batch, num_experts, type_vocab] expert transformation logits
            valid_type_size: Number of valid (non-padded) transformation types

        Returns:
            Scalar diversity loss (already weighted by diversity_loss_alpha)
        """
        if self.diversity_loss_alpha <= 0:
            return torch.tensor(0.0, device=expert_logits.device)

        # Only consider valid (non-padded) types
        expert_logits_valid = expert_logits[:, :, :valid_type_size]

        if self.diversity_loss_type == "variance":
            # Per-type variance: measure lack of diversity
            # Maximum variance for probabilities is ~0.25 (when experts maximally disagree)
            # probs: [batch, num_experts, valid_type_size]
            probs = F.softmax(expert_logits_valid, dim=-1)
            # Compute variance across experts (dim=1): [batch, valid_type_size]
            per_type_var = probs.var(dim=1)
            max_variance = 0.25
            diversity_loss = torch.clamp(max_variance - per_type_var, min=0.0).mean()

        elif self.diversity_loss_type == "kl":
            # Pairwise symmetric KL divergence: measure lack of diversity
            # For well-separated experts, average KL should be ~log(num_experts)
            probs = F.softmax(expert_logits_valid, dim=-1)
            log_probs = F.log_softmax(expert_logits_valid, dim=-1)

            kl_sum = 0.0
            count = 0
            for i in range(self.num_experts):
                for j in range(i + 1, self.num_experts):
                    kl_ij = F.kl_div(log_probs[:, j], probs[:, i], reduction="batchmean")
                    kl_ji = F.kl_div(log_probs[:, i], probs[:, j], reduction="batchmean")
                    kl_sum += (kl_ij + kl_ji) / 2.0
                    count += 1

            avg_kl = kl_sum / count if count > 0 else 0.0
            target_kl = math.log(self.num_experts)
            diversity_loss = torch.clamp(
                torch.tensor(target_kl, device=expert_logits.device) - avg_kl, min=0.0
            )

        else:
            raise ValueError(
                f"Unknown diversity_loss_type: {self.diversity_loss_type}. Must be 'variance' or 'kl'."
            )

        return self.diversity_loss_alpha * diversity_loss

    def _merge_target_indices_with_top_m(
        self,
        top_m_indices: torch.Tensor,
        top_m_values: torch.Tensor,
        target_indices: torch.Tensor,
        base_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Efficiently merge target_indices into top_m_indices, replacing lowest-scoring entries if needed.

        Strategy:
        - For each batch element, identify targets NOT already in top_m
        - Replace the lowest-scoring top_m entries with these missing targets
        - Maintains output shape [batch, M] for torch.compile compatibility
        - Fully vectorized for maximum performance

        Args:
            top_m_indices: [batch, M] indices of top-M bases
            top_m_values: [batch, M] probability values of top-M bases
            target_indices: [batch, K] target base indices to ensure inclusion
            base_probs: [batch, base_vocab] full base probability distribution

        Returns:
            merged_indices: [batch, M] indices with targets merged in
        """
        batch_size, M = top_m_indices.shape
        K = target_indices.shape[1]

        # If K >= M, just use targets (they already fill the whole budget)
        if K >= M:
            return target_indices[:, :M]

        # Vectorized approach: use gather/scatter operations
        # 1. Find which targets are already in top_m
        # Shape: [batch, K, M] = [batch, K, 1] == [batch, 1, M]
        target_in_top_m = target_indices.unsqueeze(2) == top_m_indices.unsqueeze(1)  # [batch, K, M]
        target_is_present = target_in_top_m.any(dim=2)  # [batch, K] - True if target is in top_m

        # 2. Create a unified set by concatenating targets and top_m, then selecting unique M entries
        # Strategy: Assign scores to each candidate, take top-M by score
        # - Targets get their actual probability + large bonus to prioritize them
        # - Top-M entries keep their original probability

        # Gather target probabilities
        target_probs = torch.gather(base_probs, 1, target_indices)  # [batch, K]

        # Create candidate pool: [targets..., top_m...]
        # Indices: [batch, K+M]
        candidate_indices = torch.cat([target_indices, top_m_indices], dim=1)
        # Scores: [batch, K+M]
        # Give targets a boost (e.g., +1000) to ensure they're selected
        # This is just for selection - doesn't affect actual computation
        target_boost = 1000.0
        candidate_scores = torch.cat(
            [
                target_probs + target_boost,  # Targets with boost
                top_m_values,  # Top-M with original scores
            ],
            dim=1,
        )

        # Select top-M from candidates
        # Note: This will prefer targets due to boost, then fall back to original top_m
        _, top_m_candidate_indices = torch.topk(candidate_scores, k=M, dim=1)  # [batch, M]

        # Gather the actual base indices
        merged_indices = torch.gather(candidate_indices, 1, top_m_candidate_indices)  # [batch, M]

        return merged_indices

    def forward(
        self,
        hidden_states: torch.Tensor,
        base_logits: torch.Tensor = None,
        top_m: int = None,
        target_base_indices: torch.Tensor = None,
        merge_targets_with_top_m: bool = True,
        base_weight: torch.Tensor = None,
        valid_type_size: int = None,
    ) -> RoutedTransformationOutput:
        """
        Compute routed transformation logits with optional top-M base pruning or specific target bases.

        Args:
            hidden_states: [batch_size, model_dim] context hidden states
            base_logits: Optional[batch_size, base_vocab] base token logits for top-M selection
            top_m: Optional[int] number of top base candidates to consider (for efficiency)
            target_base_indices: Optional[batch_size, K] specific base indices to compute for
                               (e.g., ground truth bases). When merge_targets_with_top_m=False,
                               takes precedence over top_m.
                           If False, return mixed transformation logits (weighted average)
            merge_targets_with_top_m: If True and both top_m and target_base_indices are provided,
                                     merge targets into top_m (replacing lowest-scoring bases if needed).
                                     Makes training more similar to inference. Default: True.
            base_weight: Optional[base_vocab_size, model_dim] base embedding or unembedding weights
                        (required if use_projected_keys=True and keys not cached)
            valid_type_size: Number of valid (non-padded) transformation types (type_config.total_types)
                           Required for diversity loss computation

        Returns:
            RoutedTransformationOutput with:
                - type_logits: [batch, type_vocab] or [batch, base_vocab, type_vocab] or [batch, M, type_vocab]
                - selected_indices: [batch, M] if using top-M pruning, else None
                - load_balance_loss: Scalar tensor (0.0 if disabled)
                - diversity_loss: Scalar tensor (0.0 if disabled)
        """
        batch_size = hidden_states.shape[0]

        # Select which base indices to compute for
        selected_indices = None
        if (
            target_base_indices is not None
            and base_logits is not None
            and top_m is not None
            and merge_targets_with_top_m
        ):
            # Merge mode: combine top_m from base_logits with target_base_indices
            # This makes training more similar to inference while ensuring targets are included
            base_probs = F.softmax(base_logits, dim=-1)  # [batch, base_vocab]
            top_m_values, top_m_indices = torch.topk(base_probs, k=top_m, dim=-1)  # [batch, M]

            # Merge targets into top_m, replacing lowest-scoring entries if needed
            selected_indices = self._merge_target_indices_with_top_m(
                top_m_indices, top_m_values, target_base_indices, base_probs
            )
        elif target_base_indices is not None:
            # Use provided target indices directly (e.g., ground truth bases)
            selected_indices = target_base_indices
        elif base_logits is not None and top_m is not None and top_m < self.base_vocab_size:
            # Compute base probabilities and select top-M
            base_probs = F.softmax(base_logits, dim=-1)  # [batch, base_vocab]
            _, selected_indices = torch.topk(base_probs, k=top_m, dim=-1)  # [batch, M]

        # Synchronize top-M base selection across GPUs during evaluation (optional)
        if selected_indices is not None and self.sync_top_m_bases_in_eval and not self.training:
            if dist.is_available() and dist.is_initialized():
                # Broadcast selected_indices from rank 0 to all other ranks
                # This ensures all GPUs use the same top-M bases during eval
                dist.broadcast(selected_indices, src=0)

        # 1. Compute routing weights
        if selected_indices is not None:
            # routing_weights: [batch_size, K, num_experts]
            routing_weights, routing_logits = self.router(
                hidden_states, top_m_bases=selected_indices, base_weight=base_weight
            )
        else:
            # routing_weights: [batch_size, base_vocab, num_experts]
            routing_weights, routing_logits = self.router(hidden_states, base_weight=base_weight)

        # 2. Compute load balancing loss inline (torch.compile safe)
        load_balance_loss = self.router.compute_load_balancing_loss(routing_logits)

        # 3. Compute expert transformation logits
        # expert_logits: [num_experts, batch_size, type_vocab]
        expert_logits = torch.stack(
            [
                expert_head(hidden_states)  # [batch_size, type_vocab]
                for expert_head in self.expert_heads
            ],
            dim=0,
        )

        # 4. Mix expert logits per base token
        # We want: mixed_logits[batch, base, type] = Σ_e routing_weights[batch, base, e] * expert_logits[e, batch, type]
        # Rearrange expert_logits to [batch, num_experts, type_vocab]
        expert_logits = expert_logits.transpose(0, 1)  # [batch, num_experts, type_vocab]

        # 5. Compute diversity loss (requires valid_type_size)
        if valid_type_size is not None:
            diversity_loss = self.compute_expert_diversity_loss(expert_logits, valid_type_size)
        else:
            diversity_loss = torch.tensor(0.0, device=expert_logits.device)

        # 6. Compute per-base mixed logits using einsum
        # routing_weights: [batch, M or base_vocab, num_experts]
        # expert_logits: [batch, num_experts, type_vocab]
        # Result: [batch, M or base_vocab, type_vocab]
        per_base_type_logits = torch.einsum(
            "bme,bet->bmt", routing_weights.to(expert_logits.dtype), expert_logits
        )

        # Return per-base logits with optional indices
        return RoutedTransformationOutput(
            type_logits=per_base_type_logits,
            selected_indices=selected_indices,
            load_balance_loss=load_balance_loss,
            diversity_loss=diversity_loss,
        )
