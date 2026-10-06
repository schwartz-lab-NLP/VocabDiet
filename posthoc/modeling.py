from typing import Callable, List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
import copy
import os
from transformers import LlamaForCausalLM
from transformers import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.streamers import BaseStreamer
from transformers.generation import GenerationMixin, GenerationConfig, GenerateEncoderDecoderOutput
from transformers.generation.utils import (
    ModelOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput,
    GenerateNonBeamOutput,
    GenerateDecoderOnlyOutput,
)
from dataclasses import dataclass
from collections import defaultdict
import types
from .utils.probes import FlexProbe

from typing import Type
from transformers import PreTrainedModel


class EfficientMaskedLMHead(nn.Module):
    """
    A compute-efficient LM head that only computes logits for non-ignored indices.

    Args:
        original_lm_head: The original nn.Linear layer
        ignored_indices: List or tensor of vocabulary indices to ignore
        ignored_value: Value to fill in for ignored indices
    """

    def __init__(self, original_lm_head, ignored_indices, ignored_value=float("-inf")):
        super().__init__()

        self.original_out_features = original_lm_head.out_features
        self.ignored_value = ignored_value

        # Convert ignored_indices to tensor and sort for easier handling
        if isinstance(ignored_indices, (list, tuple)):
            ignored_indices = torch.tensor(ignored_indices)
        self.register_buffer("ignored_indices", ignored_indices.long())

        # Create keep_indices (all indices except ignored ones)
        all_indices = torch.arange(self.original_out_features)
        keep_mask = ~torch.isin(all_indices, self.ignored_indices)
        keep_indices = all_indices[keep_mask]
        self.register_buffer("keep_indices", keep_indices.long())

        # Create the shortened linear layer with only the kept weights
        self.shortened_lm_head = nn.Linear(
            original_lm_head.in_features, len(keep_indices), bias=original_lm_head.bias is not None
        )

        # Copy the relevant weights and biases
        with torch.no_grad():
            self.shortened_lm_head.weight.data = original_lm_head.weight[keep_indices].detach()
            if original_lm_head.bias is not None:
                self.shortened_lm_head.bias.data = original_lm_head.bias[keep_indices].detach()

    def forward(self, hidden_states):
        """
        Forward pass that computes only non-ignored logits and reconstructs full tensor.

        Args:
            hidden_states: Input tensor of shape (..., hidden_size)

        Returns:
            logits: Full logits tensor of shape (..., original_vocab_size)
        """
        # Compute only the kept logits - this saves computation!
        kept_logits = self.shortened_lm_head(hidden_states)

        # Get the output shape
        output_shape = list(hidden_states.shape[:-1]) + [self.original_out_features]

        # Create full logits tensor filled with ignored_value
        full_logits = torch.full(
            output_shape, self.ignored_value, device=hidden_states.device, dtype=kept_logits.dtype
        )

        # Fill in the computed logits at the correct positions
        full_logits[..., self.keep_indices] = kept_logits

        return full_logits

    def update_ignored_indices(self, new_ignored_indices, new_ignored_value=None):
        """
        Update the ignored indices (requires rebuilding the shortened layer).
        This is expensive and should be done sparingly.
        """
        if new_ignored_value is not None:
            self.ignored_value = new_ignored_value

        # Store current weights
        current_full_weight = torch.zeros(
            self.original_out_features, self.shortened_lm_head.in_features
        )
        current_full_weight[self.keep_indices] = self.shortened_lm_head.weight.data

        current_full_bias = None
        if self.shortened_lm_head.bias is not None:
            current_full_bias = torch.zeros(self.original_out_features)
            current_full_bias[self.keep_indices] = self.shortened_lm_head.bias.data

        # Update indices
        if isinstance(new_ignored_indices, (list, tuple)):
            new_ignored_indices = torch.tensor(new_ignored_indices)
        self.ignored_indices = new_ignored_indices.long()

        all_indices = torch.arange(self.original_out_features)
        keep_mask = ~torch.isin(all_indices, self.ignored_indices)
        self.keep_indices = all_indices[keep_mask].long()

        # Rebuild shortened layer
        old_layer = self.shortened_lm_head
        self.shortened_lm_head = nn.Linear(
            old_layer.in_features, len(self.keep_indices), bias=old_layer.bias is not None
        )

        # Copy weights
        with torch.no_grad():
            self.shortened_lm_head.weight.data = current_full_weight[self.keep_indices]
            if current_full_bias is not None:
                self.shortened_lm_head.bias.data = current_full_bias[self.keep_indices]

    @property
    def weight(self):
        return self.shortened_lm_head.weight


class EfficientInputEmbedding(nn.Module):
    """
    A memory-efficient input embedding layer that only loads weights for non-ignored indices.

    Args:
        original_embedding: The original nn.Embedding layer
        ignored_indices: List or tensor of vocabulary indices to ignore
    """

    def __init__(self, original_embedding, ignored_indices):
        super().__init__()

        self.original_num_embeddings = original_embedding.num_embeddings
        self.embedding_dim = original_embedding.embedding_dim
        self.padding_idx = original_embedding.padding_idx

        # Convert ignored_indices to tensor and sort for easier handling
        if isinstance(ignored_indices, (list, tuple)):
            ignored_indices = torch.tensor(ignored_indices)
        self.register_buffer("ignored_indices", ignored_indices.long())

        # Create keep_indices (all indices except ignored ones)
        all_indices = torch.arange(self.original_num_embeddings)
        keep_mask = ~torch.isin(all_indices, self.ignored_indices)
        keep_indices = all_indices[keep_mask]
        self.register_buffer("keep_indices", keep_indices.long())

        # Create mapping from original indices to shortened indices
        # This will be used to map input_ids to the shortened embedding space
        index_mapping = torch.full((self.original_num_embeddings,), -1, dtype=torch.long)
        index_mapping[keep_indices] = torch.arange(len(keep_indices))
        self.register_buffer("index_mapping", index_mapping)

        # Create the shortened embedding layer with only the kept embeddings
        self.shortened_embedding = nn.Embedding(
            len(keep_indices),
            self.embedding_dim,
            padding_idx=None,  # We'll handle padding separately if needed
        )

        # Copy the relevant embeddings
        with torch.no_grad():
            self.shortened_embedding.weight.data = original_embedding.weight[keep_indices].clone()

    def forward(self, input_ids):
        """
        Forward pass that returns embeddings for non-ignored tokens.

        Args:
            input_ids: Input tensor of token indices of shape (...,)

        Returns:
            embeddings: Embedding tensor of shape (..., embedding_dim)
        """
        shortened_indices = self.index_mapping[input_ids]
        # Get embeddings for valid indices
        return self.shortened_embedding(shortened_indices)


class ExtendedLogitsMapper(nn.Module):
    def __init__(
        self,
        base_token_indices,
        extended_vocab_size,
        transform_groups,
        decomposition_map,
        untouched_indices=None,
        default_value=-18.0,
        transformable_token_indices=None,
        group_applicability=None,
        device="cuda",
    ):
        """
        Initialize the ExtendedLogitsMapper with optimized computation for transformable tokens.

        Args:
            base_token_indices: List or tensor of indices in the model's output logits
                                that correspond to base tokens.
            extended_vocab_size: Size of the extended vocabulary V
            transform_groups: List of dictionaries, each representing a transform group
                Each dict contains:
                - 'name': str, name of the transform group
                - 'transforms': list of transform names in this group
            decomposition_map: Dictionary mapping (base_word_idx, transform_combination) to extended_word_idx,
                where transform_combination is a tuple of transform indices, one for each group.
            untouched_indices: List or tensor of indices in the model's output logits that should be
                               copied directly to the extended logits without any transformation.
            group_applicability: Dictionary mapping (base_word_idx, group_idx)
                to a boolean indicating if the entire transform group is applicable to the base word.
                For example, {(0, 1): False} means transform group 1 is not applicable to base word 0.
            transformable_token_indices: List or tensor of indices in the extended vocabulary that
                                        can be created by applying transforms to base tokens.
                                        If None, it will be inferred from decomposition_map.
            device: Device to store tensors on
        """
        super().__init__()
        self.base_vocab_size = len(base_token_indices)
        self.extended_vocab_size = extended_vocab_size
        self.transform_groups = transform_groups
        self.device = device
        self.default_value = default_value

        # Map between model output indices and base token indices
        self.base_token_indices = torch.tensor(base_token_indices, device=device)
        assert len(self.base_token_indices) == self.base_vocab_size, (
            "Number of base token indices must match base_vocab_size"
        )

        # Register base token indices as a buffer
        self.register_buffer("model_to_base_indices", self.base_token_indices)

        # Store untouched indices to be copied directly
        self.untouched_indices = None
        if untouched_indices is not None and len(untouched_indices) > 0:
            self.untouched_indices = torch.tensor(untouched_indices, device=device).to(torch.long)
            self.register_buffer("untouched_logits_indices", self.untouched_indices)

        # Create inverse mapping from base token indices to positions in model logits
        base_to_model_indices = torch.zeros(self.base_vocab_size, dtype=torch.long, device=device)
        for i, idx in enumerate(self.base_token_indices):
            base_to_model_indices[i] = idx
        self.register_buffer("base_to_model_indices", base_to_model_indices)

        # Calculate total number of transforms
        self.num_transform_groups = len(transform_groups)
        self.num_transforms_per_group = [len(group["transforms"]) for group in transform_groups]
        self.total_transforms = sum(self.num_transforms_per_group)

        # Save group applicability information
        self.group_applicability = group_applicability or {}

        # Create mapping from base words and transforms to extended vocabulary
        self.base_to_extended_mapping = decomposition_map

        # Calculate offsets for transform logits
        self.transform_group_offsets = [0]
        for num_transforms in self.num_transforms_per_group:
            self.transform_group_offsets.append(self.transform_group_offsets[-1] + num_transforms)

        # Determine transformable token indices if not provided
        if transformable_token_indices is None:
            transformable_token_indices = list(self.base_to_extended_mapping.values())

        # Convert to tensor and store
        self.transformable_token_indices = torch.tensor(
            transformable_token_indices, device=device, dtype=torch.long
        )
        self.register_buffer("transformable_indices", self.transformable_token_indices)
        self.num_transformable_tokens = len(self.transformable_token_indices)

        # Create mapping from transformable indices to their position in the transformable array
        self.ext_to_transformable_map = torch.zeros(
            self.extended_vocab_size, dtype=torch.long, device=device
        )
        for i, idx in enumerate(self.transformable_token_indices):
            self.ext_to_transformable_map[idx] = i
        self.register_buffer("ext_to_transformable_indices", self.ext_to_transformable_map)

        # Precompute and register indices as buffers for the transformable subset
        base_word_indices, transform_indices, base_word_positions = self._compute_indices()
        self.register_buffer("base_word_indices", base_word_indices)
        self.register_buffer("transform_indices", transform_indices)
        self.register_buffer("base_logits_positions", base_word_positions)

        # Create and register group applicability mask
        group_mask = self._compute_group_applicability_mask()
        self.register_buffer("group_mask", group_mask)

    def _compute_group_applicability_mask(self):
        """
        Compute a mask indicating which transform groups apply to which transformable words.

        Returns:
            group_mask: Tensor of shape [num_transformable_tokens, num_transform_groups]
                        with 1.0 for applicable groups and 0.0 for non-applicable groups
        """
        if not self.group_applicability:
            return None

        # Initialize mask with all ones (all groups applicable by default)
        group_mask = torch.ones(
            (self.num_transformable_tokens, self.num_transform_groups),
            dtype=torch.float,
            device=self.device,
        )

        # For each transformable index, check which groups are applicable to its base word
        for i, ext_idx in enumerate(self.transformable_token_indices):
            base_idx = self.base_word_indices[i].item()

            for group_idx in range(self.num_transform_groups):
                # Check if this group is not applicable to this base word
                if (
                    base_idx,
                    group_idx,
                ) in self.group_applicability and not self.group_applicability[
                    (base_idx, group_idx)
                ]:
                    group_mask[i, group_idx] = 0.0

        return group_mask

    def _compute_indices(self):
        """
        Compute indices for efficient forward pass, mapping for transformable tokens only.

        Returns:
            base_word_indices: Tensor mapping each transformable token to its base word
            transform_indices: Tensor mapping each transformable token to its transforms
        """
        base_word_indices = torch.zeros(
            self.num_transformable_tokens, dtype=torch.long, device=self.device
        )
        transform_indices = torch.zeros(
            (self.num_transformable_tokens, self.num_transform_groups),
            dtype=torch.long,
            device=self.device,
        )
        inverse_base_to_extended_mapping = {v: k for k, v in self.base_to_extended_mapping.items()}
        # For each transformable token, find its base word and transforms
        for i, ext_idx in enumerate(self.transformable_token_indices):
            # Find the base_idx and transform_combo for this extended token
            (base_idx, transform_combo) = inverse_base_to_extended_mapping[ext_idx.item()]
            base_word_indices[i] = base_idx

            for group_idx, transform_idx in enumerate(sorted(transform_combo)):
                # For non-applicable groups, we'll use the "none" transform (index 0)
                if (
                    base_idx,
                    group_idx,
                ) in self.group_applicability and not self.group_applicability[
                    (base_idx, group_idx)
                ]:
                    transform_indices[i, group_idx] = 0  # Use "none" transform
                else:
                    transform_indices[i, group_idx] = transform_idx

        # Map transformable token base words to positions in the base logits tensor
        base_word_positions = torch.zeros(
            self.num_transformable_tokens, dtype=torch.long, device=self.device
        )
        base_token_to_position = {
            idx.item(): pos for pos, idx in enumerate(self.base_token_indices)
        }
        for i in range(self.num_transformable_tokens):
            base_idx = base_word_indices[i].item()
            base_word_positions[i] = base_token_to_position[base_idx]

        return base_word_indices, transform_indices, base_word_positions

    def extract_base_logits(self, model_logits):
        """
        Extract base token logits from the model's output logits.

        Args:
            model_logits: Tensor of shape [batch_size, model_logits_size]

        Returns:
            base_logits: Tensor of shape [batch_size, base_vocab_size]
        """
        # Use the base_to_model_indices to gather the relevant logits
        batch_size = model_logits.shape[0]
        indices = self.base_to_model_indices.unsqueeze(0).expand(batch_size, -1)

        # Gather the base logits from model_logits
        base_logits = torch.gather(model_logits, -1, indices)

        return base_logits

    def additive_forward_optimized(self, model_logits, transform_logits):
        """
        Optimized additive implementation that operates only on transformable tokens.
        Instead of using softmax and probabilities, we directly add transform logits to base logits.

        Args:
            model_logits: Tensor of shape [batch_size, model_logits_size]
            transform_logits: Tensor of shape [batch_size, total_transforms]

        Returns:
            extended_logits: Tensor of shape [batch_size, extended_vocab_size]
        """
        model_logits = model_logits.to(torch.float)
        transform_logits = transform_logits.to(torch.float)
        batch_size = model_logits.shape[0]

        # Extract base logits from model logits
        base_logits = self.extract_base_logits(model_logits)

        # Initialize extended logits with default value
        extended_logits = self.default_value * torch.ones(
            (batch_size, self.extended_vocab_size),
            dtype=model_logits.dtype,
            device=model_logits.device,
        )

        # Copy untouched logits directly first
        if self.untouched_indices is not None:
            # For each untouched index, copy the logit directly
            extended_logits[:, self.untouched_indices] = model_logits[:, self.untouched_indices]

        # If no transformable tokens, return early
        if self.num_transformable_tokens == 0:
            return extended_logits

        # Get base logits for each transformable token's base word
        # [batch_size, num_transformable_tokens]
        base_positions = self.base_logits_positions.unsqueeze(0).expand(batch_size, -1)
        transformable_base_logits = torch.gather(base_logits, 1, base_positions)

        # Initialize transformed logits with base logits
        transformable_logits = transformable_base_logits.clone()

        # For each transform group, add the appropriate transform logits
        for group_idx in range(self.num_transform_groups):
            # Get transform logits for this group
            start_idx = self.transform_group_offsets[group_idx]
            end_idx = self.transform_group_offsets[group_idx + 1]
            group_logits = transform_logits[:, start_idx:end_idx]  # [batch_size, num_transforms]

            # Select logits for each transformable token based on its transform index
            # [batch_size, num_transformable_tokens]
            batch_transform_indices = (
                self.transform_indices[:, group_idx].unsqueeze(0).expand(batch_size, -1) - start_idx
            )
            selected_logits = torch.gather(group_logits, 1, batch_transform_indices)

            # Apply group applicability mask
            if self.group_mask is not None:
                group_applicability = self.group_mask[:, group_idx].unsqueeze(
                    0
                )  # [1, num_transformable_tokens]
                effective_logits = torch.where(
                    group_applicability > 0, selected_logits, torch.zeros_like(selected_logits)
                )
            else:
                effective_logits = selected_logits

            # Add transform logits to the base logits
            transformable_logits += effective_logits

        # Copy transformable logits to their positions in the extended logits tensor
        extended_logits[:, self.transformable_token_indices] = transformable_logits.to(
            extended_logits.dtype
        )
        return extended_logits

    def forward(self, model_logits, transform_logits):
        """
        Forward pass using additive approach by default.

        Args:
            model_logits: Tensor of shape [batch_size, model_logits_size] or [batch_size, seq_len, model_logits_size]
            transform_logits: Tensor of shape [batch_size, total_transforms] or [batch_size, seq_len, total_transforms]
            use_addition: Whether to use additive implementation instead of the original probability-based approach

        Returns:
            extended_logits: Tensor of shape [batch_size, extended_vocab_size] or [batch_size, seq_len, extended_vocab_size]
        """
        reshape = model_logits.dim() > 2
        if reshape:
            batch_size = model_logits.shape[0]
            seq_len = model_logits.shape[1]
            model_logits = model_logits.reshape(batch_size * seq_len, -1)
            transform_logits = transform_logits.reshape(batch_size * seq_len, -1)

        extended_logits = self.additive_forward_optimized(model_logits, transform_logits)

        if reshape:
            extended_logits = extended_logits.reshape(batch_size, seq_len, -1)

        return extended_logits


class CustomGenerationMixin(GenerationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_curr_model_kwargs_update = False
        self.history = None
        self._reset_metrics()

    def _reset_metrics(self):
        self.input_inflection_counts = []
        self.output_inflection_counts = []
        self.generation_inputs = []
        self.generation_outputs = []
        self.input_inflection_token_counts = {}
        self.output_inflection_token_counts = {}
        self.input_total_tokens = 0
        self.output_total_tokens = 0

    def _count_inflections(self, token_ids):
        """Count inflections and their token counts for given token_ids."""
        inflection_mask = self.map_input_ids_to_is_inflection_token(token_ids)
        inflection_count = inflection_mask.sum().item()

        inflection_token_counts = {}
        if inflection_count > 0:
            inflection_indices = torch.where(inflection_mask)[0]
            for idx in inflection_indices:
                token_count = self.map_input_ids_to_inflection_token_count(token_ids[idx]).item()
                if token_count > 0:
                    inflection_token_counts[token_count] = (
                        inflection_token_counts.get(token_count, 0) + 1
                    )

        return inflection_count, inflection_token_counts

    def _update_metrics(self, input_ids, generated_ids):
        """Update cached metrics with new generation results."""
        batch_size = input_ids.shape[0]
        input_len = input_ids.shape[1]

        for i in range(batch_size):
            # Input metrics
            input_tokens = input_ids[i]
            input_inflection_count, input_inflection_token_counts = self._count_inflections(
                input_tokens
            )
            self.input_inflection_counts.append(input_inflection_count)
            self.input_total_tokens += input_len

            for count, freq in input_inflection_token_counts.items():
                self.input_inflection_token_counts[count] = (
                    self.input_inflection_token_counts.get(count, 0) + freq
                )

            # Output metrics (excluding input prompt)
            output_tokens = generated_ids[i][input_len:]
            output_inflection_count, output_inflection_token_counts = self._count_inflections(
                output_tokens
            )
            self.output_inflection_counts.append(output_inflection_count)
            self.output_total_tokens += len(output_tokens)

            for count, freq in output_inflection_token_counts.items():
                self.output_inflection_token_counts[count] = (
                    self.output_inflection_token_counts.get(count, 0) + freq
                )

            self.generation_inputs.append(self.tokenizer.decode(input_tokens))
            self.generation_outputs.append(self.tokenizer.decode(output_tokens))

    def get_metrics_and_reset(self):
        """Return all cached metrics and reset for next task."""
        metrics = {
            "input_inflection_counts": self.input_inflection_counts.copy(),
            "output_inflection_counts": self.output_inflection_counts.copy(),
            "input_inflection_token_counts": self.input_inflection_token_counts.copy(),
            "output_inflection_token_counts": self.output_inflection_token_counts.copy(),
            "input_total_tokens": self.input_total_tokens,
            "output_total_tokens": self.output_total_tokens,
            "inputs": self.generation_inputs,
            "outputs": self.generation_outputs,
        }
        self._reset_metrics()
        return metrics

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        # Store original input for metrics
        original_input_ids = input_ids.clone()

        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = (
                model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            )
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states")
                if output_hidden_states
                else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        if isinstance(model_kwargs.get("past_key_values"), Cache):
            is_compileable = (
                model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
            )
            is_compileable = is_compileable and not self.generation_config.disable_compile
            if is_compileable and (
                self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
            ):
                os.environ["TOKENIZERS_PARALLELISM"] = "0"
                model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(
            this_peer_finished, synced_gpus, device=input_ids.device
        ):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update(
                {"output_attentions": output_attentions} if output_attentions else {}
            )
            model_inputs.update(
                {"output_hidden_states": output_hidden_states} if output_hidden_states else {}
            )

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone().float()
            next_token_logits = next_token_logits.to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,)
                        if self.config.is_encoder_decoder
                        else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        # Update metrics after generation is complete
        self._update_metrics(original_input_ids, input_ids)

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


@dataclass
class AdditiveCausalLMOutputWithPast(CausalLMOutputWithPast):
    base_logits: torch.FloatTensor = None
    original_logits: torch.FloatTensor = None
    type_logits: torch.FloatTensor = None
    all_type_logits: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None


def create_additive_model_class(
    base_model_class: Type[PreTrainedModel], custom_generation_mixin: Type = CustomGenerationMixin
) -> Type[PreTrainedModel]:
    """
    Factory function to create an AdditiveFooForCausalLM class for different model types.

    Args:
        base_model_class: The base model class to extend (e.g., QwenForCausalLM)
        custom_generation_mixin: The custom generation mixin to use (defaults to CustomGenerationMixin)

    Returns:
        A dynamically created class that combines the base model with custom generation
    """

    class AdditiveFooForCausalLM(custom_generation_mixin, base_model_class):
        def __init__(
            self,
            config,
            types_vocab_size,
            model_vocab_size=None,
            extended_vocab_size=None,
            clm_loss_alpha=1.0,
            types_loss_alpha=1.0,
            base_loss_alpha=1.0,
            probe_types=False,
            num_probe_layers=1,
            types_prediction_layer=-1,
            base_prediction_layer=-1,
            types_loss_indices=None,
            use_lm_preds_for_type_labels=False,
            use_type_labels_for_lm_preds=True,
            modeling_option=None,
            lora_ft=False,
            lora_adapter_toggling=False,
            k_last_layers=0,
            base_pred_layer=None,
            train_input_types_only=False,
            extended_logits_loss=True,
            tokenizer=None,
            # Weight tying parameters
            tie_word_embeddings=False,  # Enable manual weight tying for models with tied embeddings
            use_efficient_tied_embeddings=False,  # Use EfficientInputEmbedding for tied weight models (experimental)
        ):
            super().__init__(config)
            self.tokenizer = tokenizer
            self.types_vocab_size = types_vocab_size
            self.model_vocab_size = model_vocab_size
            self.extended_vocab_size = extended_vocab_size

            # Post-hoc adaptation uses extended-vocabulary scoring.
            self.compositional_prediction = False

            self.logits_mapper = None

            self.base_tokens_map = None

            self.embed_types = nn.Embedding(self.types_vocab_size, config.hidden_size, 0)

            self.token_types_filter = nn.Embedding(
                self.extended_vocab_size, self.types_vocab_size, self.model.padding_idx
            )
            self.token_to_type_ids = nn.Embedding(
                self.extended_vocab_size, self.types_vocab_size, self.model.padding_idx
            )

            self.token_id_to_base_id_mapping = None
            self.token_id_to_orig_first_id_mapping = None
            self.token_id_is_base_or_inflection_mapping = None
            self.token_id_is_inflection_mapping = None
            self.token_id_to_inflection_token_count = None

            self.clm_loss_alpha = clm_loss_alpha
            self.clm_loss_function = nn.CrossEntropyLoss()
            self.distill_loss_function = nn.KLDivLoss(reduction="sum")
            self.distill_temperature = 2.0
            self.distill_mse_loss_function = nn.HuberLoss(reduction="sum")
            self.distill_use_mse_on_logits = False
            self.last_layer_copy = None
            self.k_last_layers = k_last_layers
            self.base_pred_layer = base_pred_layer

            self.types_negative_ids = None
            self.ignored_logits_idx = None
            self.ignored_logits_val = -18
            self.tie_word_embeddings = tie_word_embeddings
            self.use_efficient_tied_embeddings = use_efficient_tied_embeddings
            self.original_lm_head = self.lm_head
            for param in self.original_lm_head.parameters():
                param.requires_grad = False

            self.probe_types = probe_types
            self.num_probe_layers = (
                min(num_probe_layers, config.num_hidden_layers) if num_probe_layers > 0 else 1
            )
            self.types_prediction_layer = types_prediction_layer
            self.base_prediction_layer = base_prediction_layer
            self.types_loss_alpha = types_loss_alpha
            self.base_loss_alpha = base_loss_alpha
            self.use_type_labels_for_lm_preds = use_type_labels_for_lm_preds
            self.use_lm_preds_for_type_labels = use_lm_preds_for_type_labels
            self.train_input_types_only = train_input_types_only
            self.inference_with_input_types_only = False
            self.distill_to_model_copy = False
            self.distill_to_self = True
            self.distill_hidden_states = False
            self.distill_hidden_layer = 7
            self.distill_hidden_alpha = 1.0
            self.always_distill_to_original = (
                False  # True distillation: always use original embeddings for teacher
            )
            self.extended_logits_loss = extended_logits_loss
            self.modeling_option = modeling_option
            self.lora_ft = lora_ft
            self.lora_adapter_toggling = lora_adapter_toggling
            self.lora_applied = False  # Track whether LoRA has been applied to model
            self.lora_trainable = False
            self.lora_trained = False

            # Store conditioning parameters

            self.act_fn = nn.SiLU()

            self.types_prediction_head = FlexProbe(
                config.hidden_size, self.types_vocab_size, dtype=self.dtype
            )
            self.types_lm_head = nn.Linear(config.hidden_size, self.types_vocab_size, bias=False)
            self.types_lm_proj_mat = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

            self.types_loss_indices = types_loss_indices
            if self.types_loss_indices is None:
                raise ValueError(f"types_loss_indices is None, unsupported right now")
            self.types_num_prefix = (
                self.types_loss_indices["prefix"][1] - self.types_loss_indices["prefix"][0]
            )
            self.types_num_capital = (
                self.types_loss_indices["capitalization"][1]
                - self.types_loss_indices["capitalization"][0]
            )
            self.types_num_inflections = (
                self.types_loss_indices["inflections"][1]
                - self.types_loss_indices["inflections"][0]
            )
            self.total_type_logits = (
                self.types_num_inflections + self.types_num_prefix + self.types_num_capital
            )

            self.types_loss_pos_weight = 3
            self.types_loss_neg_weight = 1
            self.types_prefix_loss_fct = nn.CrossEntropyLoss(
                weight=torch.Tensor(
                    [self.types_loss_pos_weight] * (self.types_num_prefix - 1)
                    + [self.types_loss_neg_weight]
                )
            )
            self.types_capital_loss_fct = nn.CrossEntropyLoss(
                weight=torch.Tensor(
                    [self.types_loss_pos_weight] * (self.types_num_capital - 1)
                    + [self.types_loss_neg_weight]
                )
            )
            self.types_inflection_loss_fct = nn.CrossEntropyLoss(
                weight=torch.Tensor(
                    [self.types_loss_pos_weight] * (self.types_num_inflections - 1)
                    + [self.types_loss_neg_weight]
                )
            )

            self.post_from_pretrained_init()

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            # Extract the tokenizer if provided
            tokenizer = kwargs.pop("tokenizer", None)

            # Load the model normally
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

            model.post_from_pretrained_init()

            # Add the tokenizer back after loading
            if tokenizer is not None:
                model.tokenizer = tokenizer

            return model

        def clone_last_layer(self):
            # Skip cloning if using adapter toggling (memory-efficient mode)
            if self.lora_adapter_toggling:
                self.last_layer_copy = None
                return

            self.last_layer_copy = nn.ModuleList()
            for layer in self.model.layers[-self.k_last_layers :]:
                detached_layer = copy.deepcopy(layer)

                for param in detached_layer.parameters():
                    param.requires_grad_(False)

                self.last_layer_copy.append(detached_layer)

        def set_token_id_to_base_id_mapping(self, mapping_dict: Dict[int, int]):
            """Set token mapping using tensors for efficient lookup."""
            indices = torch.Tensor(list(mapping_dict.keys())).to(torch.long)
            values = torch.Tensor(list(mapping_dict.values())).to(torch.long)
            self.token_id_to_base_id_mapping = torch.arange(
                self.extended_vocab_size, dtype=torch.long
            )
            self.token_id_to_base_id_mapping[indices] = values

        def set_token_id_is_base_or_inflection_token(
            self,
            base_token_ids: List[int],
            inflection_token_ids: List[int] = None,
            inflection_token_counts: List[int] = None,
        ):
            """Set token mapping using tensors for efficient lookup."""
            if inflection_token_ids is not None:
                base_token_ids = base_token_ids + inflection_token_ids
            indices = torch.Tensor(base_token_ids).to(torch.long)
            self.token_id_is_base_or_inflection_mapping = torch.zeros(
                self.extended_vocab_size, dtype=self.dtype
            )
            self.token_id_is_base_or_inflection_mapping[indices] = 1
            if inflection_token_ids is not None:
                indices = torch.Tensor(inflection_token_ids).to(torch.long)
                self.token_id_is_inflection_mapping = torch.zeros(
                    self.extended_vocab_size, dtype=self.dtype
                )
                self.token_id_is_inflection_mapping[indices] = 1
                if inflection_token_counts is not None:
                    self.token_id_to_inflection_token_count = torch.zeros(
                        self.extended_vocab_size, dtype=self.dtype
                    )
                    self.token_id_to_inflection_token_count[indices] = torch.Tensor(
                        inflection_token_counts
                    ).to(self.dtype)

        def set_token_id_to_orig_first_id_mapping(self, mapping_dict: Dict[int, int]):
            """Set token mapping using tensors for efficient lookup."""
            indices = torch.Tensor(list(mapping_dict.keys())).to(torch.long)
            values = torch.Tensor(list(mapping_dict.values())).to(torch.long)
            self.token_id_to_orig_first_id_mapping = torch.arange(
                self.extended_vocab_size, dtype=torch.long
            )
            self.token_id_to_orig_first_id_mapping[indices] = values

        def set_seq_replacement_mapping(self, mapping_dict: Dict[int, List[int]]):
            return None

        def set_logits_mapper(
            self,
            base_token_indices,
            decomposition_map,
            transformation_int_to_name,
            types_loss_indices_map,
            untouched_indices=None,
        ):
            self._logits_mapper_recipe = {
                "base_token_indices": list(base_token_indices),
                "decomposition_map": {
                    str(k): [int(v[0]), list(v[1])] for k, v in decomposition_map.items()
                },
                "transformation_int_to_name": transformation_int_to_name,
                "types_loss_indices_map": types_loss_indices_map,
                "untouched_indices": list(untouched_indices)
                if untouched_indices is not None
                else None,
            }
            transform_groups = list()
            # List of dictionaries, each representing a transform group. Each dict contains:
            # - 'name': str, name of the transform group
            # - 'transforms': list of transform names in this group
            for k, v in types_loss_indices_map.items():
                transform_groups.append(
                    {
                        "name": k,
                        "transforms": [transformation_int_to_name[i] for i in range(v[0], v[1])],
                    }
                )

            inverse_decomposition_map = dict()
            for k, v in decomposition_map.items():
                inverse_decomposition_map[(v[0], tuple(v[1]))] = k
            self.logits_mapper = ExtendedLogitsMapper(
                base_token_indices,
                self.extended_vocab_size,
                transform_groups,
                inverse_decomposition_map,
                untouched_indices=untouched_indices,
                default_value=self.ignored_logits_val,
                device=self.device,
            )

        def set_train_input_types_only(self, value: bool = False):
            self.train_input_types_only = value

        def set_inference_with_input_types_only(self, value: bool = False):
            self.inference_with_input_types_only = value

        def set_distill_to_model_copy(self, value: bool = False):
            self.distill_to_model_copy = value

        def set_distill_to_self(self, value: bool = False):
            self.distill_to_self = value

        def set_distill_hidden_states(self, value: bool = False):
            self.distill_hidden_states = value

        def set_distill_hidden_layer(self, layer_idx: int):
            self.distill_hidden_layer = layer_idx

        def set_distill_hidden_alpha(self, alpha: float):
            self.distill_hidden_alpha = alpha

        def set_always_distill_to_original(self, value: bool = False):
            self.always_distill_to_original = value

        def set_use_lm_preds_for_type_labels(self, value: bool = False):
            self.use_lm_preds_for_type_labels = value

        def set_use_type_labels_for_lm_preds(self, value: bool = False):
            self.use_type_labels_for_lm_preds = value

        def split_types_to_groups(self, types_tensor):
            prefix_types = types_tensor[
                ..., self.types_loss_indices["prefix"][0] : self.types_loss_indices["prefix"][1]
            ]
            capitalization_types = types_tensor[
                ...,
                self.types_loss_indices["capitalization"][0] : self.types_loss_indices[
                    "capitalization"
                ][1],
            ]
            inflection_types = types_tensor[
                ...,
                self.types_loss_indices["inflections"][0] : self.types_loss_indices["inflections"][
                    1
                ],
            ]
            return {
                # "multilabel": multilabel_types,
                "capitalization": capitalization_types,
                "prefix": prefix_types,
                "inflections": inflection_types,
            }

        def get_split_type_ids(self):
            return {
                # "multilabel": list(range(self.types_num_multilabel)),
                "prefix": list(
                    range(
                        self.types_loss_indices["prefix"][0], self.types_loss_indices["prefix"][1]
                    )
                ),
                "capitalization": list(
                    range(
                        self.types_loss_indices["capitalization"][0],
                        self.types_loss_indices["capitalization"][1],
                    )
                ),
                "inflections": list(
                    range(
                        self.types_loss_indices["inflections"][0],
                        self.types_loss_indices["inflections"][1],
                    )
                ),
            }

        def map_input_ids_to_base_ids(self, tokens: torch.Tensor) -> torch.Tensor:
            if self.token_id_to_base_id_mapping is None:
                return tokens

            # Move mapping tensors to same device as input
            if self.token_id_to_base_id_mapping.device != tokens.device:
                self.token_id_to_base_id_mapping = self.token_id_to_base_id_mapping.to(
                    tokens.device
                )

            return self.token_id_to_base_id_mapping[tokens]

        def map_input_ids_to_is_inflection_token(self, tokens: torch.Tensor) -> torch.Tensor:
            if self.token_id_is_inflection_mapping is None:
                return torch.zeros_like(tokens)

            # Move mapping tensors to same device as input
            if self.token_id_is_inflection_mapping.device != tokens.device:
                self.token_id_is_inflection_mapping = self.token_id_is_inflection_mapping.to(
                    tokens.device
                )

            return self.token_id_is_inflection_mapping[tokens]

        def map_input_ids_to_inflection_token_count(self, tokens: torch.Tensor) -> torch.Tensor:
            if self.token_id_to_inflection_token_count is None:
                return torch.zeros_like(tokens)

            # Move mapping tensors to same device as input
            if self.token_id_to_inflection_token_count.device != tokens.device:
                self.token_id_to_inflection_token_count = (
                    self.token_id_to_inflection_token_count.to(tokens.device)
                )

            return self.token_id_to_inflection_token_count[tokens]

        def map_input_ids_to_is_base_or_inflection_token(
            self, tokens: torch.Tensor
        ) -> torch.Tensor:
            if self.token_id_is_base_or_inflection_mapping is None:
                return torch.ones_like(tokens)

            # Move mapping tensors to same device as input
            if self.token_id_is_base_or_inflection_mapping.device != tokens.device:
                self.token_id_is_base_or_inflection_mapping = (
                    self.token_id_is_base_or_inflection_mapping.to(tokens.device)
                )

            return self.token_id_is_base_or_inflection_mapping[tokens]

        def map_input_ids_to_orig_first_ids(self, tokens: torch.Tensor) -> torch.Tensor:
            if self.token_id_to_orig_first_id_mapping is None:
                return tokens

            # Move mapping tensors to same device as input
            if self.token_id_to_orig_first_id_mapping.device != tokens.device:
                self.token_id_to_orig_first_id_mapping = self.token_id_to_orig_first_id_mapping.to(
                    tokens.device
                )

            return self.token_id_to_orig_first_id_mapping[tokens]

        def freeze_rows_in_type_embeddings(self, rows_to_freeze):
            self.zero_out_negative_rows_in_output_type_embeddings(rows_to_freeze)
            rows_to_freeze = torch.Tensor(list(rows_to_freeze)).to(torch.long)

            def _mask_grads(grad):
                grad = grad.clone()
                grad[rows_to_freeze] = 0
                return grad

            self.embed_types.weight.register_hook(_mask_grads)
            self.types_lm_head.weight.register_hook(_mask_grads)

        def get_type_embeddings(self):
            return self.embed_types

        def set_input_type_embeddings(self, new_embeddings):
            with torch.no_grad():
                self.embed_types.weight.copy_(new_embeddings)

        def post_from_pretrained_init(self):
            with torch.no_grad():
                self.embed_types.weight.data.fill_(0)
                self.token_types_filter.weight.data.fill_(0)
                self.token_to_type_ids.weight.data.fill_(0)

                self.types_lm_proj_mat.weight.data.fill_(0)

                self.types_lm_head.weight.data.fill_(0)

            self.clone_last_layer()

        def get_output_type_embeddings(self):
            return self.types_lm_head

        def set_output_type_embeddings(self, new_embeddings, ignored_idx=None):
            if ignored_idx is not None:
                ignored_idx = torch.Tensor(list(ignored_idx)).to(torch.long)
            with torch.no_grad():
                if ignored_idx is not None:
                    new_embeddings[ignored_idx] = self.types_lm_head.weight[ignored_idx]
                self.types_lm_head.weight.copy_(new_embeddings)

        def zero_out_negative_rows_in_output_type_embeddings(self, ignored_idx):
            ignored_idx = torch.Tensor(list(ignored_idx)).to(torch.long)
            with torch.no_grad():
                weight = self.types_lm_head.weight.data

                if max(ignored_idx) >= weight.shape[0] or min(ignored_idx) < 0:
                    raise ValueError("Indices must be within the range [0, n_rows-1]")

                # Zero out the specified rows
                weight[ignored_idx] = 0
                # If the layer has bias, zero out corresponding bias terms as well
                if self.types_lm_head.bias is not None:
                    self.types_lm_head.bias.data[ignored_idx] = 0

        def set_output_probe_embeddings(
            self,
            new_embeddings,
            negative_type_ids=None,
            classes_to_merge=None,
        ):
            if self.num_probe_layers == 1:
                self.types_prediction_head.set_negative_classes(negative_type_ids)
                self.types_prediction_head.set_classes_to_merge(classes_to_merge)
                self.types_prediction_head.set_representatives(new_embeddings)
                self.types_prediction_head = nn.ModuleList(
                    [
                        FlexProbe(
                            self.config.hidden_size,
                            new_embeddings,
                            empty_class_value=0.0,
                            negative_class_ids=negative_type_ids,
                            classes_to_merge=classes_to_merge,
                            dtype=self.dtype,
                        )
                        for _ in range(self.num_probe_layers)
                    ]
                ).to(self.device)

        def set_token_types_filter(self, base_vocab_idx, allowed_types_per_base_word):
            with torch.no_grad():
                self.token_types_filter.weight.data.fill_(0)
                for token_id, type_ids in zip(base_vocab_idx, allowed_types_per_base_word):
                    self.token_types_filter.weight.data[token_id, type_ids] = 1

        def set_token_to_type_ids(self, final_decomposition_map, negative_type_ids=None):
            token_id_to_base_id_map = dict()
            token_id_to_types_list_map = dict()
            with torch.no_grad():
                self.token_to_type_ids.weight.data.fill_(0)
                if negative_type_ids is not None:
                    if isinstance(negative_type_ids, set):
                        negative_type_ids = list(negative_type_ids)
                    self.token_to_type_ids.weight.data[:, negative_type_ids] = 1
                for token_id, decomposition in final_decomposition_map.items():
                    base_token_id, type_ids = decomposition
                    token_id_to_base_id_map[token_id] = base_token_id
                    token_id_to_types_list_map[token_id] = type_ids
                    self.token_to_type_ids.weight.data[token_id, :] = 0
                    self.token_to_type_ids.weight.data[token_id, type_ids] = 1

                self.set_token_id_to_base_id_mapping(token_id_to_base_id_map)

        def set_negative_type_ids(self, values):
            self.types_negative_ids = torch.Tensor(list(values)).to(torch.long)

        def set_ignored_logits_idx(self, values):
            self.ignored_logits_idx = torch.Tensor(sorted(list(values))).to(torch.long)
            self.lm_head = EfficientMaskedLMHead(
                self.lm_head, self.ignored_logits_idx, self.ignored_logits_val
            )

            # If manual weight tying is enabled AND efficient embeddings are requested, wrap input embeddings and tie weights
            if self.tie_word_embeddings and self.use_efficient_tied_embeddings:
                self.model.embed_tokens = EfficientInputEmbedding(
                    self.model.embed_tokens, self.ignored_logits_idx
                )
                self._tie_or_clone_weights(
                    self.lm_head.shortened_lm_head, self.model.embed_tokens.shortened_embedding
                )
                print(
                    f"Enabled efficient tied embeddings: wrapped input embeddings and tied weights"
                )

        def _tie_or_clone_weights(self, output_embeddings, input_embeddings):
            """Tie or clone module weights depending of whether we are using TorchScript or not"""
            if self.config.torchscript:
                output_embeddings.weight = nn.Parameter(input_embeddings.weight.clone())
            else:
                output_embeddings.weight = input_embeddings.weight

            # Passing hooks over to the embeddings if needed
            # (currently limited to tensor parallel hooks and flags only)
            if hasattr(input_embeddings, "_is_hooked") and getattr(
                input_embeddings, "_hf_tp_plan", None
            ):
                output_embeddings._is_hooked = input_embeddings._is_hooked
                output_embeddings._hf_tp_plan = input_embeddings._hf_tp_plan
                output_embeddings._forward_hooks = input_embeddings._forward_hooks
                output_embeddings._forward_pre_hooks = input_embeddings._forward_pre_hooks
                output_embeddings.__repr__ = lambda: (
                    f"{output_embeddings.__repr__()}\nTP Plan: {output_embeddings._hf_tp_plan}"
                )

            if getattr(output_embeddings, "bias", None) is not None:
                output_embeddings.bias.data = nn.functional.pad(
                    output_embeddings.bias.data,
                    (
                        0,
                        output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0],
                    ),
                    "constant",
                    0,
                )
            if hasattr(output_embeddings, "out_features") and hasattr(
                input_embeddings, "num_embeddings"
            ):
                output_embeddings.out_features = input_embeddings.num_embeddings

        def tie_weights(self):
            """
            Override tie_weights to handle EfficientMaskedLMHead which has a weight property
            that conflicts with parameter assignment during weight tying.

            If tie_word_embeddings is enabled, manual tying is already handled in
            set_ignored_logits_idx via _tie_or_clone_weights.
            """
            # Check if lm_head is EfficientMaskedLMHead - if so, skip automatic weight tying
            # because the @property weight conflicts with PyTorch's parameter registration
            if isinstance(self.lm_head, EfficientMaskedLMHead):
                return
            # Otherwise proceed with normal weight tying
            super().tie_weights()

        def apply_lora_to_last_layer(
            self, target_modules: List = None, lora_r=64, lora_alpha=64, lora_dropout=0.0
        ):
            """
            Apply LoRA to the last k layers of the base model.

            Args:
                target_modules: List of module identifiers. Supports:
                    - Short names: "q", "k", "v", "o" for attention projections
                    - "mlp" for all MLP layers (gate_proj, up_proj, down_proj)
                    - Full names: "self_attn.q_proj", "mlp.gate_proj", etc.
                lora_r: LoRA rank (default: 64)
                lora_alpha: LoRA alpha parameter (default: 64)
                lora_dropout: LoRA dropout (default: 0.0)
            """
            from peft import LoraConfig, get_peft_model

            # Default: apply to all attention and MLP modules
            if target_modules is None:
                target_modules = ["q", "k", "v", "o", "mlp"]

            # Map short names to full module names
            module_mapping = {
                "q": "self_attn.q_proj",
                "k": "self_attn.k_proj",
                "v": "self_attn.v_proj",
                "o": "self_attn.o_proj",
                "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
            }

            # Convert short names to full module names
            expanded_modules = []
            for module in target_modules:
                if module in module_mapping:
                    mapped = module_mapping[module]
                    if isinstance(mapped, list):
                        expanded_modules.extend(mapped)
                    else:
                        expanded_modules.append(mapped)
                else:
                    # Assume it's already a full module name
                    expanded_modules.append(module)

            # Build final target modules with layer indices
            final_target_modules = []
            for layer_i in range(
                self.config.num_hidden_layers - self.k_last_layers, self.config.num_hidden_layers
            ):
                layer_name = f"layers.{layer_i}"
                for module_name in expanded_modules:
                    final_target_modules.append(f"{layer_name}.{module_name}")

            print(
                f"Applying LoRA to {len(final_target_modules)} modules across {self.k_last_layers} layer(s)"
            )
            print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, modules={expanded_modules}")

            # Configure LoRA
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=final_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # PEFT 0.8.0+ expects prepare_inputs_for_generation method
            # Add it to self.model if it doesn't exist (for compatibility)
            if not hasattr(self.model, "prepare_inputs_for_generation"):
                # Copy the method from the parent class (self)
                self.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)

            # Re-tie weights if needed (PEFT wrapping changes self.model reference)
            if (
                self.tie_word_embeddings
                and self.use_efficient_tied_embeddings
                and isinstance(self.lm_head, EfficientMaskedLMHead)
            ):
                # Re-tie to the PEFT-wrapped embedding layer
                self._tie_or_clone_weights(
                    self.lm_head.shortened_lm_head, self.model.embed_tokens.shortened_embedding
                )
                print("Re-tied weights after LoRA application")

            # Mark that LoRA has been applied
            self.lora_applied = True

            # Verify that only LoRA parameters are trainable
            trainable_params = 0
            all_params = 0
            for _, param in self.model.named_parameters():
                all_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()

            print(
                f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)"
            )

        def set_lora_trainable(self, trainable: bool):
            self.lora_trainable = trainable
            for name, param in self.named_parameters():
                if "lora" in name:
                    param.requires_grad = trainable
            if not self.lora_trained:
                # assume lora was fine-tuned and therefore needs to be used during inference
                self.lora_trained = True

        def selective_apply_linear_layer(self, hidden_states, routing_labels, layer):
            """
            Apply projection only to entries where routing_labels has 1s in non-negative type positions.

            Args:
                hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
                routing_labels: Tensor of shape (batch_size, n_types, seq_len) with 0s and 1s
                layer: nn.Linear layer to apply to hidden_states

            Returns:
                Tensor of shape (batch_size, seq_len, proj_dim) with selective projection applied
            """
            n_types = routing_labels.shape[1]

            # Create a mask of negative types (1 for negative types, 0 for others)
            negative_mask = torch.zeros(n_types, device=hidden_states.device)
            negative_mask[self.types_negative_ids] = 1.0

            # Reshape to make it broadcastable with routing_labels (1, n_types, 1)
            negative_mask = negative_mask.view(1, n_types, 1)

            # Create a mask that's 1 for positions that have any non-negative type activated
            # First, mask out all negative types by multiplying with (1 - negative_mask)
            non_negative_activations = routing_labels * (1 - negative_mask)

            # Then, check if any non-negative type is activated for each (batch, seq_len) position
            # Sum across the types dimension and check if > 0
            selector_mask = non_negative_activations.sum(dim=1) > 0  # Shape: (batch_size, seq_len)

            # Reshape the mask for broadcasting with hidden_states
            selector_mask = selector_mask.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)

            # Apply the projection to all hidden states
            projected = layer(hidden_states)

            # Use the mask to select which positions retain the projection
            result = projected * selector_mask

            return result

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            **kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            output_attentions = (
                output_attentions
                if output_attentions is not None
                else self.config.output_attentions
            )
            # Force output_hidden_states to True - both for probing and for extracting the last hidden state before normalization
            # When using adapter toggling or during inference, we don't need output_hidden_states for LoRA (more memory efficient)
            lora_needs_hidden_states = (
                self.lora_ft
                and self.lora_trained
                and self.training
                and not self.lora_adapter_toggling
            )
            need_hidden_for_distill = self.distill_hidden_states and self.train_input_types_only
            output_hidden_states = (
                True
                if (
                    (self.num_probe_layers > 1)
                    or lora_needs_hidden_states
                    or (self.base_prediction_layer < -1)
                    or need_hidden_for_distill
                )
                else output_hidden_states
            )

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # get input embeddings:
            # 1. get type ids of original token
            type_ids = self.token_to_type_ids(input_ids).transpose(1, 2)
            # 2. map token to base token in case it is modeled additively
            orig_input_ids = input_ids
            input_ids = self.map_input_ids_to_base_ids(input_ids)
            base_labels = input_ids
            types_embeds = torch.einsum(
                "nkm,kd->nmd", type_ids.to(self.dtype), self.embed_types.weight
            )
            inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds += types_embeds
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )

            all_hidden_states = outputs.hidden_states if output_hidden_states else None
            hidden_states = outputs[0]

            # Initialize teacher logits (may not be computed during inference)
            original_logits = None
            original_base_logits = None

            if self.distill_to_model_copy:
                # Stage 1: Input training with distillation to model using original embeddings
                with torch.no_grad():
                    orig_input_ids = self.map_input_ids_to_orig_first_ids(orig_input_ids)
                    # Output hidden states if we're doing hidden state distillation
                    need_hidden_states = self.distill_hidden_states and self.train_input_types_only
                    orig_outputs = self.model(
                        input_ids=orig_input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=None,  # Teacher needs its own cache
                        inputs_embeds=None,
                        use_cache=False,  # Don't cache teacher outputs
                        output_attentions=False,
                        output_hidden_states=need_hidden_states,
                        return_dict=return_dict,
                        cache_position=cache_position,
                        **kwargs,
                    )
                    original_logits = self.original_lm_head(
                        orig_outputs[0][:, -num_logits_to_keep:, :]
                    ).detach()
                    original_base_logits = original_logits.detach().clone()
                    original_hidden_states = (
                        orig_outputs.hidden_states if need_hidden_states else None
                    )
            elif self.lora_ft and self.lora_trained:
                # If LoRA should be applied, verify it actually is
                if self.lora_applied:
                    # Verify PEFT adapters are actually present
                    has_peft_adapters = (
                        hasattr(self.model, "peft_config")
                        and len(getattr(self.model, "peft_config", {})) > 0
                    )
                    if not has_peft_adapters:
                        raise RuntimeError(
                            "LoRA was marked as applied (lora_applied=True) but PEFT adapters not found on model. "
                            "This indicates a bug in LoRA application. Please check apply_lora_to_last_layer()."
                        )
                else:
                    # LoRA not yet applied - we're in pre-init stages, skip distillation
                    has_peft_adapters = False

                if self.lora_adapter_toggling and has_peft_adapters:
                    if self.training:
                        # Memory-efficient adapter toggling approach (training only)
                        # Forward pass 1: Disable adapters to get teacher outputs
                        with torch.no_grad():
                            # Disable adapter layers to get base model outputs
                            # PEFT models have disable_adapter_layers() method
                            self.model.disable_adapter_layers()

                            # Use original embeddings or base+type embeddings based on always_distill_to_original flag
                            if self.always_distill_to_original:
                                teacher_input_ids = self.map_input_ids_to_orig_first_ids(
                                    orig_input_ids
                                )
                                teacher_inputs_embeds = None
                            else:
                                teacher_input_ids = None
                                teacher_inputs_embeds = inputs_embeds

                            teacher_outputs = self.model(
                                input_ids=teacher_input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=None,  # Don't reuse cache for teacher pass
                                inputs_embeds=teacher_inputs_embeds,
                                use_cache=False,  # No cache for teacher pass
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=return_dict,
                                cache_position=cache_position,
                                **kwargs,
                            )
                            original_logits = self.original_lm_head(
                                teacher_outputs[0][:, -num_logits_to_keep:, :]
                            ).detach()
                            original_base_logits = original_logits.detach().clone()

                            # Re-enable adapter layers for student forward pass
                            self.model.enable_adapter_layers()
                elif self.training and has_peft_adapters:
                    # Original approach: use cloned layers (requires output_hidden_states)
                    # Only during training - skip during inference to save computation
                    # Also requires PEFT adapters to be applied (cloned layers are created during apply_lora_to_last_layer)

                    # If always_distill_to_original, do full forward pass with original embeddings instead of using cloned layers
                    if self.always_distill_to_original:
                        with torch.no_grad():
                            teacher_input_ids = self.map_input_ids_to_orig_first_ids(orig_input_ids)
                            # Note: We don't disable adapters here because we're doing a full forward pass
                            # The cloned layers don't have adapters anyway
                            teacher_outputs = self.model(
                                input_ids=teacher_input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=None,
                                inputs_embeds=None,
                                use_cache=False,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=return_dict,
                                cache_position=cache_position,
                                **kwargs,
                            )
                            # Use the cloned lm_head (without LoRA) for teacher
                            original_logits = self.original_lm_head(
                                teacher_outputs[0][:, -num_logits_to_keep:, :]
                            ).detach()
                            original_base_logits = original_logits.detach().clone()
                    else:
                        # Standard cloned layers approach with base+type embeddings
                        with torch.no_grad():
                            if use_cache and past_key_values is None:
                                from transformers import DynamicCache

                                past_key_values = DynamicCache()

                            if cache_position is None:
                                past_seen_tokens = (
                                    past_key_values.get_seq_length()
                                    if past_key_values is not None
                                    else 0
                                )
                                cache_position = torch.arange(
                                    past_seen_tokens,
                                    past_seen_tokens + inputs_embeds.shape[1],
                                    device=inputs_embeds.device,
                                )

                            if position_ids is None:
                                position_ids = cache_position.unsqueeze(0)

                            position_embeddings = self.model.rotary_emb(inputs_embeds, position_ids)

                            causal_mask = self.model._update_causal_mask(
                                attention_mask,
                                inputs_embeds,
                                cache_position,
                                past_key_values,
                                output_attentions,
                            )
                            hidden_states_copy = all_hidden_states[-(self.k_last_layers + 1)]
                            for layer in self.last_layer_copy:
                                last_layer_out = layer(
                                    hidden_states_copy,
                                    attention_mask=causal_mask,
                                    position_embeddings=position_embeddings,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    cache_position=cache_position,
                                    **kwargs,
                                )
                                hidden_states_copy = last_layer_out[0]

                            original_logits = self.original_lm_head(
                                self.model.norm(hidden_states_copy[:, -num_logits_to_keep:, :])
                            ).detach()  # [..., :self.model_vocab_size].detach()

                            original_base_logits = original_logits.detach().clone()
                else:
                    # Pre-init stages or when PEFT not applied: compute teacher logits from last hidden state
                    if self.training and self.always_distill_to_original:
                        # Use original embeddings for teacher
                        with torch.no_grad():
                            teacher_input_ids = self.map_input_ids_to_orig_first_ids(orig_input_ids)
                            teacher_outputs = self.model(
                                input_ids=teacher_input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=None,
                                inputs_embeds=None,
                                use_cache=False,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=return_dict,
                                cache_position=cache_position,
                                **kwargs,
                            )
                            original_logits = self.original_lm_head(
                                teacher_outputs[0][:, -num_logits_to_keep:, :]
                            ).detach()
                            original_base_logits = original_logits.detach().clone()
                    else:
                        # Use student's hidden states (base+type embeddings)
                        with torch.no_grad():
                            original_logits = self.original_lm_head(
                                outputs[0][:, -num_logits_to_keep:, :]
                            ).detach()
                            original_base_logits = original_logits.detach().clone()
            else:
                # Default fallback: use the last hidden state
                if self.training and self.always_distill_to_original:
                    # Use original embeddings for teacher
                    with torch.no_grad():
                        teacher_input_ids = self.map_input_ids_to_orig_first_ids(orig_input_ids)
                        teacher_outputs = self.model(
                            input_ids=teacher_input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            past_key_values=None,
                            inputs_embeds=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=return_dict,
                            cache_position=cache_position,
                            **kwargs,
                        )
                        original_logits = self.original_lm_head(
                            teacher_outputs[0][:, -num_logits_to_keep:, :]
                        ).detach()
                        original_base_logits = original_logits.detach().clone()
                else:
                    # Use student's hidden states (base+type embeddings)
                    with torch.no_grad():
                        original_logits = self.original_lm_head(
                            outputs[0][:, -num_logits_to_keep:, :]
                        ).detach()
                        original_base_logits = original_logits.detach().clone()

            # Compute base logits FIRST (before type prediction for conditioning)
            if self.train_input_types_only:
                logits = self.original_lm_head(hidden_states[:, -num_logits_to_keep:, :])
            elif self.base_prediction_layer < -1:
                base_hidden_states = all_hidden_states[self.base_prediction_layer]
                base_hidden_states = base_hidden_states[:, -num_logits_to_keep:, :]
                base_hidden_states = self.model.norm(base_hidden_states)
                logits = self.lm_head(base_hidden_states)
            else:
                logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            # A transformation logit is the dot product with its unembedding offset.
            type_logits = self.types_lm_head(hidden_states[:, -num_logits_to_keep:, :])
            type_logits_output = type_logits

            if original_base_logits is not None and self.ignored_logits_idx is not None:
                with torch.no_grad():
                    original_base_logits[..., self.ignored_logits_idx] = self.ignored_logits_val

            if (
                self.ignored_logits_idx is not None
                and (not self.train_input_types_only)
                and (not self.inference_with_input_types_only)
            ):  #  and self.clm_loss_alpha > 0.0 and labels is not None
                with torch.no_grad():
                    logits[..., self.ignored_logits_idx] = self.ignored_logits_val

            # Conditionally compute extended logits
            if self.logits_mapper is not None:
                extended_logits = self.logits_mapper.forward(logits, type_logits)

            loss = 0.0

            if self.clm_loss_alpha > 0.0 and labels is not None:
                if self.train_input_types_only:
                    orig_labels = self.map_input_ids_to_orig_first_ids(labels)

                    shift_labels = orig_labels[..., 1:].contiguous()

                    shift_labels = shift_labels.view(-1)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_logits = shift_logits.view(len(shift_labels), -1)

                    use_clm = not self.distill_to_model_copy
                    use_distill = self.distill_to_model_copy

                    if use_clm:
                        loss_clm = self.clm_loss_function(shift_logits, shift_labels)
                        loss += self.clm_loss_alpha * loss_clm

                    if use_distill:
                        masked_extended_logits = logits
                        masked_extended_target_logits = original_logits

                        loss_distill = (
                            self.distill_loss_function(
                                F.log_softmax(
                                    masked_extended_logits / self.distill_temperature, dim=-1
                                ),
                                F.softmax(
                                    masked_extended_target_logits / self.distill_temperature, dim=-1
                                ),
                            )
                            * (self.distill_temperature) ** 2
                        ) / (masked_extended_logits.size(0) * masked_extended_logits.size(1))
                        loss += 1 * self.clm_loss_alpha * loss_distill

                    # Hidden state distillation loss
                    if (
                        self.distill_hidden_states
                        and original_hidden_states is not None
                        and all_hidden_states is not None
                    ):
                        # Extract hidden states at the specified layer
                        student_hidden = all_hidden_states[
                            self.distill_hidden_layer
                        ]  # [batch, seq_len, hidden_dim]
                        teacher_hidden = original_hidden_states[
                            self.distill_hidden_layer
                        ].detach()  # [batch, seq_len, hidden_dim]

                        # Create mask for positions where types are actually used (non-zero type embeddings)
                        # type_ids shape: [batch, num_types, seq_len] from line 1990
                        # Sum across type dimension to get positions with any non-zero types
                        type_mask = (types_embeds.sum(dim=-1) > 0).float()  # [batch, seq_len]

                        # Compute MSE loss only at positions with types
                        hidden_diff = (
                            student_hidden - teacher_hidden
                        ) ** 2  # [batch, seq_len, hidden_dim]
                        hidden_diff_mean = hidden_diff.mean(dim=-1)  # [batch, seq_len]

                        # Apply mask and normalize by number of positions with types
                        masked_loss = (hidden_diff_mean * type_mask).sum()
                        num_type_positions = type_mask.sum()

                        if num_type_positions > 0:
                            loss_hidden_distill = masked_loss / num_type_positions
                            loss += self.distill_hidden_alpha * loss_hidden_distill

                # Compositional mode: Use separate base and type losses
                use_clm = not self.distill_to_self

                if use_clm:
                    shift_labels = labels[..., 1:].contiguous()
                    shift_labels = shift_labels.view(-1)
                    shift_logits = extended_logits[..., :-1, :].contiguous()
                    shift_logits = shift_logits.view(len(shift_labels), -1)

                    loss_clm = self.clm_loss_function(shift_logits, shift_labels)
                    loss += self.clm_loss_alpha * loss_clm
                else:
                    masked_extended_logits = extended_logits[..., : original_logits.size(-1)]
                    masked_extended_target_logits = original_logits

                    # Create a mask for valid positions (inverse of ignored_targets)
                    ignored_targets = (labels < 0) | (labels >= original_logits.size(-1))
                    valid_mask = ~ignored_targets

                    # Compute distillation loss with position masking
                    # Select only valid positions using boolean indexing for memory efficiency
                    valid_student_logits = masked_extended_logits[
                        valid_mask
                    ]  # Shape: (num_valid, vocab)
                    valid_teacher_logits = masked_extended_target_logits[
                        valid_mask
                    ]  # Shape: (num_valid, vocab)

                    # Compute KL divergence only on valid positions (reduction="sum" sums over all elements)
                    loss_distill = (
                        self.distill_loss_function(
                            F.log_softmax(valid_student_logits / self.distill_temperature, dim=-1),
                            F.softmax(valid_teacher_logits / self.distill_temperature, dim=-1),
                        )
                        * (self.distill_temperature**2)
                    ) / valid_mask.sum().clamp_min(1)
                    loss += self.clm_loss_alpha * loss_distill

            # Add base token prediction loss if enabled  (skip if compositional mode - already included)
            if (
                self.base_loss_alpha > 0.0
                and labels is not None
                and original_logits is not None
                and not self.train_input_types_only
            ):
                # Use ensure_top_m for memory-efficient computation with fixed number of elements per position
                _, target_mask, masked_target_logits = self.create_top_k_target_logits(
                    original_logits,
                    top_k=self.distill_topk,
                    use_base_form=True,
                    ensure_top_m=self.base_distill_top_m,
                )

                # Only compute loss if there are valid positions
                if target_mask.any():
                    batch_size, seq_len, vocab_size = logits.shape
                    num_positions = batch_size * seq_len
                    expected_elements = num_positions * self.base_distill_top_m
                    actual_elements = target_mask.sum().item()

                    # Check if we have exactly the expected number of elements for memory-efficient path
                    if actual_elements == expected_elements:
                        # Memory-efficient: use boolean indexing + reshape
                        # Each position has exactly base_distill_top_m valid elements, so we can reshape
                        student_topk = logits[
                            target_mask
                        ]  # Shape: (batch * seq_len * base_distill_top_m,)
                        teacher_topk = masked_target_logits[
                            target_mask
                        ]  # Shape: (batch * seq_len * base_distill_top_m,)

                        # Reshape to (batch * seq_len, base_distill_top_m) for proper softmax computation
                        student_topk = student_topk.view(num_positions, self.base_distill_top_m)
                        teacher_topk = teacher_topk.view(num_positions, self.base_distill_top_m)

                        # Compute softmax only over top_m elements (much more memory efficient!)
                        loss_distill = (
                            self.distill_loss_function(
                                F.log_softmax(student_topk / self.distill_temperature, dim=-1),
                                F.softmax(teacher_topk / self.distill_temperature, dim=-1),
                            )
                            * (self.distill_temperature) ** 2
                        ) / num_positions
                    else:
                        # Fallback: some positions have fewer than base_distill_top_m valid elements
                        # Use original torch.where approach (less memory efficient but works with variable counts)
                        masked_logits = torch.where(target_mask, logits, self.ignored_logits_val)

                        loss_distill = (
                            self.distill_loss_function(
                                F.log_softmax(masked_logits / self.distill_temperature, dim=-1),
                                F.softmax(masked_target_logits / self.distill_temperature, dim=-1),
                            )
                            * (self.distill_temperature) ** 2
                        ) / num_positions

                    loss += self.base_loss_alpha * loss_distill

                else:
                    print("WARNING: target_mask is empty")

            # Add type prediction loss if enabled (skip if compositional mode - already included)
            if (
                self.types_loss_alpha > 0.0
                and labels is not None
                and type_logits is not None
                and not self.train_input_types_only
            ):
                # In distillation setting, get type labels from teacher's top-1 predictions
                # Get teacher's top-1 token predictions
                teacher_top1_tokens = original_logits.argmax(dim=-1)  # [batch, seq_len]

                # Get type IDs for teacher's predicted tokens
                teacher_type_ids = self.token_to_type_ids(teacher_top1_tokens).transpose(
                    1, 2
                )  # [batch, num_types, seq_len]

                # Shift for next-token prediction
                shift_type_labels = teacher_type_ids[
                    ..., 1:
                ].contiguous()  # [batch, num_types, seq_len-1]
                shift_type_logits = type_logits[
                    ..., :-1, :
                ].contiguous()  # [batch, seq_len-1, num_types]

                # Create mask to only compute loss on inflected tokens
                # (tokens that have at least one non-negative type active)
                negative_mask = torch.zeros(self.types_vocab_size, device=type_ids.device)
                if self.types_negative_ids is not None:
                    negative_mask[self.types_negative_ids] = 1.0

                # Check which positions have any non-negative type
                non_negative_types = shift_type_labels * (1 - negative_mask.view(1, -1, 1))
                shift_types_ignore_mask = non_negative_types.sum(dim=1) > 0  # [batch, seq_len-1]

                # Apply mask and reshape
                shift_type_labels_masked = shift_type_labels.transpose(1, 2)[
                    shift_types_ignore_mask
                ]  # [num_valid, num_types]
                shift_type_logits_masked = shift_type_logits[
                    shift_types_ignore_mask
                ]  # [num_valid, num_types]

                # Compute type loss using existing method
                if shift_type_labels_masked.numel() > 0:
                    types_loss = self.compute_type_losses(
                        shift_type_labels_masked, shift_type_logits_masked
                    )
                    loss += types_loss

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            if self.inference_with_input_types_only:
                return AdditiveCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    base_logits=logits,
                    original_logits=original_logits,
                    type_logits=type_logits,
                    all_type_logits=type_logits_output if self.probe_types else type_logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return AdditiveCausalLMOutputWithPast(
                    loss=loss,
                    logits=logits if self.compositional_prediction else extended_logits,
                    base_logits=logits,
                    original_logits=original_logits,
                    type_logits=type_logits,
                    all_type_logits=type_logits_output if self.probe_types else type_logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

        def compute_agreement_with_own_pred_as_base_form(
            self, logits, original_logits, original_base_logits, shift_labels
        ):
            # get metric
            orig_top1_preds = original_logits.detach().argmax(-1)
            orig_base_top1_preds = self.map_input_ids_to_base_ids(orig_top1_preds)
            new_logits = logits.detach()
            with torch.no_grad():
                new_logits[..., self.ignored_logits_idx] = self.ignored_logits_val
            new_top1_preds = new_logits.argmax(-1)

            clm_ignore_mask = orig_top1_preds != orig_base_top1_preds

            shift_clm_ignore_mask = clm_ignore_mask[..., :-1].contiguous()
            shift_reduced_orig_top1_preds = (
                original_base_logits.detach()
                .argmax(-1)[..., :-1]
                .contiguous()[shift_clm_ignore_mask]
            )
            shift_new_top1_preds = new_top1_preds[..., :-1].contiguous()[shift_clm_ignore_mask]

            target_shift_labels = shift_labels[shift_clm_ignore_mask]

            if shift_clm_ignore_mask.sum().item():
                print(
                    f"Correct base form preds - reduced original: {(shift_reduced_orig_top1_preds == target_shift_labels).to(torch.float).mean().item()}, new: {(shift_new_top1_preds == target_shift_labels).to(torch.float).mean().item()}"
                )

        def compute_type_losses(self, shift_type_labels, shift_type_logits):
            loss_types = 0.0
            loss_types += self.types_prefix_loss_fct(
                shift_type_logits[
                    ..., self.types_loss_indices["prefix"][0] : self.types_loss_indices["prefix"][1]
                ],
                shift_type_labels[
                    ..., self.types_loss_indices["prefix"][0] : self.types_loss_indices["prefix"][1]
                ].argmax(-1),
            )
            loss_types += self.types_capital_loss_fct(
                shift_type_logits[
                    ...,
                    self.types_loss_indices["capitalization"][0] : self.types_loss_indices[
                        "capitalization"
                    ][1],
                ],
                shift_type_labels[
                    ...,
                    self.types_loss_indices["capitalization"][0] : self.types_loss_indices[
                        "capitalization"
                    ][1],
                ].argmax(-1),
            )
            loss_types += self.types_inflection_loss_fct(
                shift_type_logits[
                    ...,
                    self.types_loss_indices["inflections"][0] : self.types_loss_indices[
                        "inflections"
                    ][1],
                ],
                shift_type_labels[
                    ...,
                    self.types_loss_indices["inflections"][0] : self.types_loss_indices[
                        "inflections"
                    ][1],
                ].argmax(-1),
            )
            return self.types_loss_alpha * loss_types

        def create_top_k_target_logits(
            self,
            original_logits,
            top_k=100,
            use_base_form=True,
            use_original_first_token=False,
            ensure_top_m=None,
        ):
            """
            Create target logits for base token distillation from teacher's predictions.

            Args:
                original_logits: Teacher model logits [batch, seq_len, vocab_size]
                top_k: Number of top predictions to consider
                use_base_form: If True, maps inflected tokens to base forms
                use_original_first_token: Alternative mapping strategy
                ensure_top_m: If set, guarantees exactly m valid logits per position (for memory-efficient indexing)

            Returns:
                target_logits: Full logits with top-k mapped to base forms
                target_mask: Mask indicating valid positions
                masked_target_logits: Target logits with mask applied
            """
            batch_size, seq_len, vocab_size = original_logits.shape
            device = original_logits.device

            with torch.no_grad():
                # Get top-k predictions and their indices
                top_k_values, top_k_indices = torch.topk(
                    original_logits.detach(), k=top_k, dim=-1
                )  # [B, S, K]

                # Initialize target logits with ignored values
                target_logits = torch.full(
                    (batch_size, seq_len, vocab_size), self.ignored_logits_val, device=device
                ).to(original_logits.dtype)

                # Create mapping tensor for top-k tokens
                # Map each token to its base form
                if use_original_first_token:
                    base_indices = self.map_input_ids_to_orig_first_ids(top_k_indices)
                elif use_base_form:
                    base_indices = self.map_input_ids_to_base_ids(top_k_indices)
                else:
                    base_indices = top_k_indices

                # Aggregate logits for base tokens using the specified reduction strategy
                reduce_method = self.base_logits_reduce

                if reduce_method == "amax":
                    # Maximum: base inherits logit of most confident inflection
                    target_logits.scatter_reduce_(2, base_indices, top_k_values, reduce="amax")

                elif reduce_method == "sum":
                    # Sum: base accumulates logits from all inflections
                    # Note: This inflates values as logits don't add linearly
                    target_logits.scatter_reduce_(2, base_indices, top_k_values, reduce="sum")

                elif reduce_method == "logsumexp":
                    # LogSumExp: Proper probability aggregation
                    # Convert to probabilities, sum, convert back to logits
                    # This is numerically stable and probabilistically principled
                    probs = torch.exp(top_k_values).to(target_logits.dtype)
                    target_logits.scatter_reduce_(2, base_indices, probs, reduce="sum")
                    # Convert back to logits (avoid log(0) by using masked operations later)
                    target_logits = torch.where(
                        target_logits != self.ignored_logits_val,
                        torch.log(target_logits.clamp(min=1e-10)),  # Clamp to avoid log(0)
                        torch.tensor(
                            self.ignored_logits_val, device=device, dtype=target_logits.dtype
                        ),
                    )

                else:
                    raise ValueError(
                        f"Unknown base_logits_reduce method: {reduce_method}. "
                        f"Must be one of: 'amax', 'sum', 'logsumexp'"
                    )

                # If ensure_top_m is set, select exactly top-m from the result
                # This guarantees a fixed number of valid elements per position for memory-efficient indexing
                if ensure_top_m is not None:
                    top_m_values, top_m_indices = torch.topk(target_logits, k=ensure_top_m, dim=-1)
                    # Use top_m_values.dtype to ensure consistency (important for logsumexp)
                    target_logits = torch.full(
                        (batch_size, seq_len, vocab_size),
                        self.ignored_logits_val,
                        device=device,
                        dtype=top_m_values.dtype,
                    )
                    target_logits.scatter_(2, top_m_indices, top_m_values)

                # Ensure final dtype matches original_logits (same as student logits from LM head)
                # This is important for logsumexp which may have caused dtype drift
                if target_logits.dtype != original_logits.dtype:
                    target_logits = target_logits.to(original_logits.dtype)

                # Also create logits only for the topk
                target_mask = target_logits != self.ignored_logits_val
                masked_target_logits = torch.where(
                    target_mask, target_logits, self.ignored_logits_val
                )

            return target_logits, target_mask, masked_target_logits

    # Set the class name dynamically based on the base model class
    AdditiveFooForCausalLM.__name__ = f"Additive{base_model_class.__name__}"

    return AdditiveFooForCausalLM


from transformers import (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    Qwen3ForCausalLM,
    CohereForCausalLM,
    Phi3ForCausalLM,
    Olmo2ForCausalLM,
    Gemma2ForCausalLM,
    MistralForCausalLM,
)

# Create additive classes for various models
AdditiveLlamaForCausalLM = create_additive_model_class(LlamaForCausalLM)
AdditiveQwen2ForCausalLM = create_additive_model_class(Qwen2ForCausalLM)
AdditiveQwen3ForCausalLM = create_additive_model_class(Qwen3ForCausalLM)
AdditiveCohereForCausalLM = create_additive_model_class(CohereForCausalLM)
AdditivePhi3ForCausalLM = create_additive_model_class(Phi3ForCausalLM)
AdditiveOlmo2ForCausalLM = create_additive_model_class(Olmo2ForCausalLM)
AdditiveGemma2ForCausalLM = create_additive_model_class(Gemma2ForCausalLM)
AdditiveMistralForCausalLM = create_additive_model_class(MistralForCausalLM)
