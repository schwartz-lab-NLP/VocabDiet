from typing import Callable, List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
import copy
import os
from transformers import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.streamers import BaseStreamer
from transformers.generation import GenerationMixin, GenerationConfig
from transformers.generation.utils import (
    ModelOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput,
)
from dataclasses import dataclass
from collections import defaultdict
import types
from .probes import FlexProbe

from typing import Type
from transformers import PreTrainedModel


# Variable names used to hold the cache at generation time
ALL_CACHE_NAMES = [
    "past_key_values",  # default
    "cache_params",  # mamba-based models
    "state",  # rwkv
    "mems",  # xlnet
    "past_buckets_states",  # reformer
]


@dataclass
class AdditiveCausalLMOutputWithPast(CausalLMOutputWithPast):
    original_logits: torch.FloatTensor = None
    type_logits: Optional[Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]] = None


class AutoWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                           'none' | 'mean' | 'sum'
        """
        super(AutoWeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predicted logits [batch_size, num_classes, ...]
            targets (Tensor): Ground truth labels [batch_size, ...]

        Returns:
            Tensor: Calculated loss
        """
        # Ensure targets is 1D or 2D for counting
        targets_flat = targets.view(-1)

        # Calculate class frequencies
        unique_classes, counts = torch.unique(targets_flat, return_counts=True)
        num_samples = targets_flat.size(0)

        # Calculate weights: inverse frequency normalized so most frequent class gets weight 1
        frequencies = counts.float() / num_samples
        max_freq = frequencies.max()
        weights = max_freq / frequencies  # Inverse frequency weighting

        # Create weight tensor for all classes (fill missing classes with 0)
        num_classes = inputs.size(1)
        weight_tensor = torch.zeros(num_classes, device=inputs.device)
        weight_tensor[unique_classes] = weights

        # Compute weighted cross entropy loss
        loss = F.cross_entropy(inputs, targets, weight=weight_tensor, reduction=self.reduction)

        return loss


@dataclass
class TrieNode:
    children: Dict[int, "TrieNode"]
    replacement_token: Optional[int] = None


class TokenSequenceReplacer:
    def __init__(self, sequences: List[List[int]], replacements: List[int], device: torch.device):
        self.device = device
        self.max_length = max(len(seq) for seq in sequences)
        self.root = TrieNode(children={})

        for sequence, replacement in zip(sequences, replacements):
            current = self.root
            for token in reversed(sequence):
                if token not in current.children:
                    current.children[token] = TrieNode(children={})
                current = current.children[token]
            current.replacement_token = replacement

    def find_match(self, recent_tokens: torch.Tensor) -> Optional[Tuple[int, int]]:
        """
        Find longest matching sequence in recent tokens.
        Returns (sequence_length, replacement_token) if found, None otherwise.
        """
        if len(recent_tokens) == 0:
            return None

        current = self.root
        max_match = None

        # Search trie for longest match
        for i in range(
            len(recent_tokens) - 1, max(-1, len(recent_tokens) - self.max_length - 1), -1
        ):
            token = recent_tokens[i].item()  # Convert tensor to int
            if token not in current.children:
                break
            current = current.children[token]
            if current.replacement_token is not None:
                max_match = (len(recent_tokens) - i, current.replacement_token)

        return max_match

    def process_recent_tokens(self, tokens: torch.Tensor) -> (int, torch.Tensor):
        """
        Process recently generated token, returning replacement if sequence is matched.
        """
        match = self.find_match(tokens)
        if match is None:
            return None

        match_len, replacement = match
        return match_len, torch.Tensor([replacement]).to(self.device).to(torch.long)


class CustomGenerationMixin(GenerationMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_curr_model_kwargs_update = False
        self.sequence_buffer = None
        self.max_replaced_seq_len = 0
        self.replaced_sequences = []
        self.replacement_tokens = []
        self.seq_replacer = None
        self.history = None

    def register_sequence(self, sequence: List[int], replacement: int):
        self.max_replaced_seq_len = max(self.max_replaced_seq_len, len(sequence))
        self.replaced_sequences.append(sequence)
        self.replacement_tokens.append(replacement)

    def build_sequence_replacer(self):
        self.seq_replacer = TokenSequenceReplacer(
            self.replaced_sequences, self.replacement_tokens, self.device
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = input_ids.shape[0]
        self.history = input_ids.clone().detach()
        if self.replace_multi_token_seqs_with_additive and past_key_values is not None:
            # Initialize buffers for new batch items
            for batch_idx in range(batch_size):
                if batch_idx not in self.sequence_buffer:
                    self.sequence_buffer[batch_idx] = torch.empty(
                        0, dtype=input_ids.dtype, device=input_ids.device
                    )

            curr_tokens = input_ids[..., -1]  # Shape: [batch_size]
            # Process each item in batch
            assert batch_size == 1, (
                "A solution for multi-token replacement in batched generation is not yet implemented."
            )
            debug = False
            for batch_idx in range(batch_size):
                curr_token = curr_tokens[batch_idx : batch_idx + 1]
                buffer = self.sequence_buffer[batch_idx]

                # Update sequence buffer
                buffer_len = len(buffer)
                if buffer_len >= self.max_replaced_seq_len:
                    buffer = torch.cat([buffer[-(self.max_replaced_seq_len - 1) :], curr_token])
                else:
                    buffer = torch.cat([buffer, curr_token])
                # Check for matches
                if buffer_len > 1:
                    match = self.seq_replacer.process_recent_tokens(buffer)
                    if match is not None:
                        match_len, replacement_id = match
                        # update input_ids
                        # TODO add a solution for batched generation - currently this only works for 1 example

                        input_ids = torch.cat(
                            [input_ids[batch_idx, :-match_len], replacement_id]
                        ).unsqueeze(0)

                        past_key_values.crop(-match_len + 1)
                        if attention_mask is not None:
                            attention_mask = attention_mask[:, : -match_len + 1]
                        if cache_position is not None:
                            cache_position = cache_position - match_len + 1

                        # Clean buffer
                        buffer = torch.empty(0, dtype=input_ids.dtype, device=input_ids.device)

                        # Skip the model_kwargs update for this iteration,
                        # as it overwrites the attention mask and cache position
                        self.skip_curr_model_kwargs_update = True

                self.sequence_buffer[batch_idx] = buffer

        # Get base preparation from parent
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        # update past_key_values keeping its naming used in model code
        for possible_cache_name in ALL_CACHE_NAMES:
            if possible_cache_name in outputs:
                # TODO (joao): remove output/input mismatch when these old models (xlnet, reformer) are deprecated
                if possible_cache_name in ("past_buckets_states", "mems"):
                    cache_name = "past_key_values"
                else:
                    cache_name = possible_cache_name
                model_kwargs[cache_name] = getattr(outputs, possible_cache_name)
                break

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        if model_kwargs.get("use_cache", True):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1,
                past_positions[-1] + num_new_tokens + 1,
                dtype=past_positions.dtype,
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        if self.skip_curr_model_kwargs_update:
            self.skip_curr_model_kwargs_update = False
            return model_kwargs

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )

        return model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[Any, torch.LongTensor]:
        if inputs is not None:
            batch_size = inputs.shape[0]
        else:
            batch_size = kwargs["input_ids"].shape[0]
        self.sequence_buffer = {}

        raw_output = super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,
        )

        output = {"input_ids": raw_output}

        return output["input_ids"]


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
            extended_vocab_size=None,
            clm_loss_alpha=1.0,
            types_loss_alpha=1.0,
            probe_types=False,
            num_probe_layers=1,
            types_loss_indices=None,
            replace_multi_token_seqs_with_additive=False,
            types_use_token_preds_as_labels=True,
            types_probe_apply_relu=False,
            types_probe_use_lm_head_for_base=True,
            tokenizer=None,
        ):
            super().__init__(config)
            self.tokenizer = tokenizer
            self.types_vocab_size = types_vocab_size
            self.extended_vocab_size = extended_vocab_size
            self.embed_types = nn.Embedding(self.types_vocab_size, config.hidden_size)

            with torch.no_grad():
                self.embed_types.weight.data.fill_(0)

            self.token_types_filter = nn.Embedding(self.extended_vocab_size, self.types_vocab_size)
            self.token_to_type_ids = nn.Embedding(self.extended_vocab_size, self.types_vocab_size)
            with torch.no_grad():
                self.token_types_filter.weight.data.fill_(0)
                self.token_to_type_ids.weight.data.fill_(0)

            self.token_id_to_base_id_mapping = None
            self.token_id_to_orig_first_id_mapping = None
            self.token_id_is_base_or_inflection_mapping = None
            self.replace_multi_token_seqs_with_additive = replace_multi_token_seqs_with_additive

            self.clm_loss_alpha = clm_loss_alpha
            self.clm_loss_function = nn.CrossEntropyLoss()

            self.types_negative_ids = None
            self.ignored_logits_idx = None
            self.ignored_logits_val = -10

            self.probe_types = probe_types
            self.num_probe_layers = (
                min(num_probe_layers, config.num_hidden_layers) if num_probe_layers > 0 else 1
            )
            self.types_probe_apply_relu = types_probe_apply_relu
            self.types_loss_alpha = types_loss_alpha
            self.types_use_token_preds_as_labels = types_use_token_preds_as_labels
            self.types_probe_use_lm_head_for_base = types_probe_use_lm_head_for_base
            self.act_fn = nn.SiLU()

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

            self.types_loss_pos_weight = 2
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

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            # Extract the tokenizer if provided
            tokenizer = kwargs.pop("tokenizer", None)

            # Load the model normally
            model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

            # Add the tokenizer back after loading
            if tokenizer is not None:
                model.tokenizer = tokenizer

            return model

        def set_token_id_to_base_id_mapping(self, mapping_dict: Dict[int, int]):
            """Set token mapping using tensors for efficient lookup."""
            indices = torch.Tensor(list(mapping_dict.keys())).to(torch.long)
            values = torch.Tensor(list(mapping_dict.values())).to(torch.long)
            self.token_id_to_base_id_mapping = torch.arange(
                self.extended_vocab_size, dtype=torch.long
            )
            self.token_id_to_base_id_mapping[indices] = values

        def set_token_id_is_base_or_inflection_token(
            self, base_token_ids: List[int], inflection_token_ids: List[int] = None
        ):
            """Set token mapping using tensors for efficient lookup."""
            if inflection_token_ids is not None:
                base_token_ids = base_token_ids + inflection_token_ids
            indices = torch.Tensor(base_token_ids).to(torch.long)
            self.token_id_is_base_or_inflection_mapping = torch.zeros(
                self.extended_vocab_size, dtype=self.dtype
            )
            self.token_id_is_base_or_inflection_mapping[indices] = 1

        def set_token_id_to_orig_first_id_mapping(self, mapping_dict: Dict[int, int]):
            """Set token mapping using tensors for efficient lookup."""
            indices = torch.Tensor(list(mapping_dict.keys())).to(torch.long)
            values = torch.Tensor(list(mapping_dict.values())).to(torch.long)
            self.token_id_to_orig_first_id_mapping = torch.arange(
                self.extended_vocab_size, dtype=torch.long
            )
            self.token_id_to_orig_first_id_mapping[indices] = values

        def set_seq_replacement_mapping(self, mapping_dict: Dict[int, List[int]]):
            if self.replace_multi_token_seqs_with_additive:
                for replacement_token_id, replaced_token_ids in mapping_dict.items():
                    if len(replaced_token_ids) > 1:
                        self.register_sequence(replaced_token_ids, replacement_token_id)
                self.build_sequence_replacer()

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
                "capitalization": capitalization_types,
                "prefix": prefix_types,
                "inflections": inflection_types,
            }

        def get_split_type_ids(self):
            return {
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
            rows_to_freeze = torch.Tensor(list(rows_to_freeze)).to(torch.long)

            def _mask_grads(grad):
                grad = grad.clone()
                grad[rows_to_freeze] = 0
                return grad

            self.embed_types.weight.register_hook(_mask_grads)

        def get_type_embeddings(self):
            return self.embed_types

        def set_input_type_embeddings(self, new_embeddings):
            with torch.no_grad():
                self.embed_types.weight.copy_(new_embeddings)

        def set_output_type_embeddings(self, new_embeddings, ignored_idx=None):
            if ignored_idx is not None:
                ignored_idx = torch.Tensor(list(ignored_idx)).to(torch.long)
            with torch.no_grad():
                if self.num_probe_layers == 1:
                    if ignored_idx is not None:
                        new_embeddings[ignored_idx] = self.types_prob_head.weight[ignored_idx]
                    if self.types_probe_apply_relu:
                        new_embeddings = F.relu(new_embeddings)
                    self.types_prob_head.weight.copy_(new_embeddings)
                else:
                    for layer in self.types_prob_head:
                        if ignored_idx is not None:
                            new_embeddings[ignored_idx] = layer.weight[ignored_idx]
                        if self.types_probe_apply_relu:
                            new_embeddings = F.relu(new_embeddings)
                        layer.weight.copy_(new_embeddings)

        def set_output_probe_embeddings(self, new_embeddings):
            if self.num_probe_layers == 1:
                self.types_prob_head = FlexProbe(self.config.hidden_size, new_embeddings).to(
                    self.device
                )
            else:
                self.types_prob_head = nn.ModuleList(
                    [
                        FlexProbe(self.config.hidden_size, new_embeddings).to(self.device)
                        for _ in range(self.num_probe_layers)
                    ]
                )

        def set_token_types_filter(self, base_vocab_idx, allowed_types_per_base_word):
            with torch.no_grad():
                self.token_types_filter.weight.data.fill_(0)
                for token_id, type_ids in zip(base_vocab_idx, allowed_types_per_base_word):
                    self.token_types_filter.weight.data[token_id, type_ids] = 1

        def set_token_to_type_ids(self, final_decomposition_map, negative_type_ids=None):
            token_id_to_base_id_map = dict()
            with torch.no_grad():
                self.token_to_type_ids.weight.data.fill_(0)
                if negative_type_ids is not None:
                    if isinstance(negative_type_ids, set):
                        negative_type_ids = list(negative_type_ids)
                    self.token_to_type_ids.weight.data[:, negative_type_ids] = 1
                for token_id, decomposition in final_decomposition_map.items():
                    base_token_id, type_ids = decomposition
                    token_id_to_base_id_map[token_id] = base_token_id
                    self.token_to_type_ids.weight.data[token_id, :] = 0
                    self.token_to_type_ids.weight.data[token_id, type_ids] = 1

                self.set_token_id_to_base_id_mapping(token_id_to_base_id_map)

        def set_negative_type_ids(self, values):
            self.types_negative_ids = torch.Tensor(list(values)).to(torch.long)

        def set_ignored_logits_idx(self, values):
            self.ignored_logits_idx = torch.Tensor(list(values)).to(torch.long)

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
            # Force output_hidden_states to True if we're probing multiple layers
            if self.probe_types and self.num_probe_layers > 1:
                output_hidden_states = True
            else:
                output_hidden_states = (
                    output_hidden_states
                    if output_hidden_states is not None
                    else self.config.output_hidden_states
                )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # get input embeddings:
            # 1. get type ids of original token
            type_ids = self.token_to_type_ids(input_ids).transpose(1, 2)
            # 2. map token to base token in case it is modeled additively
            orig_input_ids = input_ids
            input_ids = self.map_input_ids_to_base_ids(input_ids)
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

            hidden_states = outputs[0]

            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
            logits_for_base_word_pred = None
            if self.types_probe_use_lm_head_for_base and self.ignored_logits_idx is not None:
                logits_for_base_word_pred = logits.clone().detach()
                with torch.no_grad():
                    logits_for_base_word_pred[..., self.ignored_logits_idx] = (
                        self.ignored_logits_val
                    )

            loss = 0.0
            # CLM loss
            if self.clm_loss_alpha > 0.0 and labels is not None:
                additive_labels = labels
                labels = self.map_input_ids_to_orig_first_ids(labels)

                shift_labels = labels[..., 1:].contiguous()
                shift_labels = shift_labels.view(-1)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_logits = shift_logits.view(len(shift_labels), -1)
                loss_clm = self.clm_loss_function(shift_logits, shift_labels)
                loss += self.clm_loss_alpha * loss_clm

            type_logits_output = None
            if self.probe_types:
                # Get type labels once (same for all layers)
                if self.types_use_token_preds_as_labels:
                    shift_types_attn = self.map_input_ids_to_is_base_or_inflection_token(
                        logits.argmax(-1)[..., :-1]
                    )
                    shift_type_labels = (
                        self.token_to_type_ids(logits.argmax(-1)[..., :-1])
                        .contiguous()
                        .to(torch.float)
                    )
                else:
                    type_labels = type_ids
                    shift_types_attn = self.map_input_ids_to_is_base_or_inflection_token(
                        shift_labels
                    )
                    shift_type_labels = (
                        type_labels.transpose(1, 2)[..., 1:, :].contiguous().to(torch.float)
                    )
                shift_types_ignore_mask = shift_types_attn == 1
                shift_type_labels = shift_type_labels[shift_types_ignore_mask]
                shift_type_labels = shift_type_labels.view(-1, shift_type_labels.shape[-1])

                if self.num_probe_layers == 1:
                    # Single layer probing
                    types_prob_input = hidden_states[:, -num_logits_to_keep:, :].detach()
                    if self.types_probe_apply_relu:
                        types_prob_input = F.relu(types_prob_input)
                    types_prob_input = types_prob_input.detach()

                    type_logits = self.types_prob_head(types_prob_input)
                    if logits_for_base_word_pred is not None:
                        type_logits = torch.concat(
                            [
                                type_logits[..., :-1],
                                logits_for_base_word_pred.max(-1, keepdim=True).values,
                            ],
                            -1,
                        )

                    # Store in a dict for consistent output format
                    type_logits_output = type_logits

                    if self.types_loss_alpha > 0.0:
                        # probe inflection types losses
                        shift_type_logits = type_logits[..., :-1, :].contiguous()
                        shift_type_logits = shift_type_logits[shift_types_ignore_mask]
                        shift_type_logits = shift_type_logits.view(-1, shift_type_labels.shape[-1])
                        types_loss = self.compute_type_losses(shift_type_labels, shift_type_logits)
                        loss += types_loss
                else:
                    # Multi-layer probing
                    all_hidden_states = outputs.hidden_states

                    # Get the last N layers based on num_probe_layers
                    layers_to_probe = all_hidden_states[-self.num_probe_layers :]

                    # Initialize dict to store logits from each layer
                    type_logits_output = {}

                    # Compute total type loss across all layers
                    total_type_loss = 0.0

                    # Process each layer
                    for layer_idx, layer_hidden_states in enumerate(layers_to_probe):
                        # Use the appropriate slice of hidden states
                        layer_prob_input = self.model.norm(
                            layer_hidden_states[:, -num_logits_to_keep:, :]
                        ).detach()
                        if self.types_probe_apply_relu:
                            layer_prob_input = F.relu(layer_prob_input)
                        # Apply the probe for this layer
                        layer_type_logits = self.types_prob_head[layer_idx](layer_prob_input)

                        # Store in dictionary with layer name
                        layer_name = (
                            f"layer_{len(all_hidden_states) - self.num_probe_layers + layer_idx}"
                        )
                        type_logits_output[layer_name] = layer_type_logits

                        # Compute loss for this layer if needed
                        if self.types_loss_alpha > 0.0:
                            shift_layer_type_logits = layer_type_logits[..., :-1, :].contiguous()
                            shift_layer_type_logits = shift_layer_type_logits[
                                shift_types_ignore_mask
                            ]
                            shift_layer_type_logits = shift_layer_type_logits.view(
                                -1, shift_type_labels.shape[-1]
                            )
                            layer_loss = self.compute_type_losses(
                                shift_type_labels, shift_layer_type_logits
                            )
                            total_type_loss += layer_loss

                    # Add the average loss to the total loss
                    if self.types_loss_alpha > 0.0:
                        loss += total_type_loss / self.num_probe_layers

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return AdditiveCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                original_logits=logits,
                type_logits=type_logits_output if self.probe_types else None,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
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

    # Set the class name dynamically based on the base model class
    AdditiveFooForCausalLM.__name__ = f"Additive{base_model_class.__name__}"

    return AdditiveFooForCausalLM


from transformers import (
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    CohereForCausalLM,
    Phi3ForCausalLM,
    Olmo2ForCausalLM,
    Gemma2ForCausalLM,
    MistralForCausalLM,
)

# Create additive classes for various models
InputAdditiveLlamaForCausalLM = create_additive_model_class(LlamaForCausalLM)
InputAdditiveQwen2ForCausalLM = create_additive_model_class(Qwen2ForCausalLM)
InputAdditiveCohereForCausalLM = create_additive_model_class(CohereForCausalLM)
InputAdditivePhi3ForCausalLM = create_additive_model_class(Phi3ForCausalLM)
InputAdditiveOlmo2ForCausalLM = create_additive_model_class(Olmo2ForCausalLM)
InputAdditiveGemma2ForCausalLM = create_additive_model_class(Gemma2ForCausalLM)
InputAdditiveMistralForCausalLM = create_additive_model_class(MistralForCausalLM)
