import torch
import tempfile
import json
import re
import os
import math
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import random

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, pipeline

from collections import defaultdict, Counter
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple
from transformers import PreTrainedTokenizer


class PatchscopesProcessor:
    def __init__(
        self,
        model,
        tokenizer,
        patchscopes_prompt: str = "{word} {word} {word} {word}",
        num_tokens_to_generate: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_input_ids, self.prompt_target_idx = self._build_prompt(
            patchscopes_prompt, "{word}"
        )
        self.num_tokens_to_generate = num_tokens_to_generate

    def _build_prompt(self, prompt: str, placeholder: str) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_input_ids = (
            [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        )
        target_idx = []

        prompt_parts = prompt.split(placeholder)
        for part_i, part in enumerate(prompt_parts):
            prompt_input_ids += self.tokenizer.encode(part, add_special_tokens=False)
            if part_i < len(prompt_parts) - 1:
                target_idx += [len(prompt_input_ids)]
                prompt_input_ids += [0]

        return (
            torch.tensor(prompt_input_ids, dtype=torch.long),
            torch.tensor(target_idx, dtype=torch.long),
        )

    def process_hidden_states_all_layers(
        self,
        hidden_states_all_layers: torch.Tensor,  # Shape: [num_layers, batch_size, hidden_size]
    ) -> List[List[str]]:
        """Process hidden states from all layers in parallel."""
        num_layers, batch_size, _ = hidden_states_all_layers.shape

        # Reshape to process all layers and batch items at once
        hidden_states_flat = hidden_states_all_layers.reshape(-1, hidden_states_all_layers.size(-1))

        # Prepare inputs for all states at once
        inputs_embeds = (
            self.model.get_input_embeddings()(self.prompt_input_ids.to(self.model.device))
            .unsqueeze(0)
            .to(hidden_states_flat.dtype)
        )
        batch_inputs = inputs_embeds.repeat(len(hidden_states_flat), 1, 1)
        batch_inputs[:, self.prompt_target_idx] = hidden_states_flat.unsqueeze(1)

        # Prepare attention mask
        attention_mask = torch.ones(
            (len(hidden_states_flat), len(self.prompt_input_ids)), device=self.model.device
        )

        # Generate for all states at once
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=batch_inputs,
                attention_mask=attention_mask,
                max_new_tokens=self.num_tokens_to_generate,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )

        # Decode all outputs
        decoded = self.tokenizer.batch_decode(outputs)

        # Reshape results to [num_layers, batch_size]
        results = [decoded[i : i + batch_size] for i in range(0, len(decoded), batch_size)]
        return results

    def process_hidden_states(self, hidden_states: torch.Tensor, batch_size: int = 32) -> List[str]:
        """Process hidden states through patchscopes in batches."""
        results = []
        for i in range(0, len(hidden_states), batch_size):
            batch = hidden_states[i : i + batch_size]

            # Prepare inputs
            inputs_embeds = self.model.get_input_embeddings()(
                self.prompt_input_ids.to(self.model.device)
            ).unsqueeze(0)
            batch_inputs = inputs_embeds.repeat(len(batch), 1, 1)
            batch_inputs[:, self.prompt_target_idx] = batch.unsqueeze(1)

            # Prepare attention mask
            attention_mask = torch.ones(
                (len(batch), len(self.prompt_input_ids)), device=self.model.device
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs_embeds=batch_inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=self.num_tokens_to_generate,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    num_beams=1,
                )

            decoded = self.tokenizer.batch_decode(outputs)
            results.extend(decoded)

        return results


def run_patchscopes_on_additive_hidden_states(
    model,
    tokenizer,
    base_tokens: Dict[str, int],
    decomposition_map: Dict[str, Dict[str, List[str]]],
    input_type_embeddings: torch.Tensor,
    type_names_to_int: Dict[str, int],
    patchscopes_prompt: str = "{word} {word} {word} {word}",
    last_layer: int = 10,
    batch_size: int = 32,
    max_words: int = None,
    space_prefix: str = "Ġ",
    single_type_only: bool = False,
    multi_type_only: bool = False,
    skip_single_token_words: bool = False,
    skip_multi_token_words: bool = False,
    compare_to_original: bool = True,
    ignored_types: List = None,
):
    """
    Run patchscopes analysis on base+type hidden states representations and optionally original words.
    Also runs analysis on base words without type embeddings.

    Returns:
        Tuple containing:
        - results: Dict mapping word to [additive_outputs, original_outputs] lists
        - results_by_type: Dict mapping type to dict of {word: [additive_outputs, original_outputs]}
        - word_pairs: List of processed (inflection, base_word, types) tuples
        - base_words: List of unique base words processed
    """
    device = model.device
    num_layers = last_layer + 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Initialize patchscopes processor
    patchscopes = PatchscopesProcessor(model, tokenizer, patchscopes_prompt=patchscopes_prompt)

    # Prepare word pairs
    word_pairs = []
    base_words_set = set()

    if ignored_types is None:
        ignored_types = []

    for base_word, inflections in tqdm(decomposition_map.items(), desc="Preparing word pairs"):
        base_words_set.add(base_word)
        for infl_word, transforms in inflections.items():
            filtered_transforms = [t for t in transforms if t not in ignored_types]
            if single_type_only and len(filtered_transforms) != 1:
                continue
            elif multi_type_only and len(filtered_transforms) == 1:
                continue

            if infl_word != base_word:
                infl_encoding = tokenizer.encode(
                    infl_word.replace(space_prefix, " ") if space_prefix else infl_word,
                    add_special_tokens=False,
                )
                if len(infl_encoding) == 1 and not skip_single_token_words:
                    word_pairs.append((infl_word, base_word, filtered_transforms))
                if len(infl_encoding) > 1 and not skip_multi_token_words:
                    word_pairs.append((infl_word, base_word, filtered_transforms))

    base_words = list(base_words_set)

    # limit words to run patchscopes on
    if max_words is not None:
        word_pairs = random.sample(word_pairs, max_words)
        base_words = random.sample(base_words, max_words)

    # Initialize results
    results = defaultdict(lambda: [[], []])  # word -> [additive_outputs, original_outputs]
    results_by_type = defaultdict(
        lambda: defaultdict(lambda: [[], []])
    )  # type -> word -> [additive_outputs, original_outputs]

    # Process base words
    for batch_start in tqdm(range(0, len(base_words), batch_size), desc="Processing base words"):
        batch_end = min(batch_start + batch_size, len(base_words))
        batch = base_words[batch_start:batch_end]
        current_batch_size = len(batch)

        with torch.no_grad():
            # Get base embeddings
            # Process original representations if requested
            if compare_to_original:
                original_inputs = tokenizer(
                    [word.replace(space_prefix, " ") if space_prefix else word for word in batch],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                original_outputs = model(
                    input_ids=original_inputs.input_ids, output_hidden_states=True, return_dict=True
                )

                # Stack hidden states from all layers for original representation
                original_hidden_states = torch.stack(
                    [states[:, -1] for states in original_outputs.hidden_states[:num_layers]]
                )

            # Process all layers
            original_layer_outputs = patchscopes.process_hidden_states_all_layers(
                original_hidden_states
            )

            # Store results
            for batch_idx, word in enumerate(batch):
                word_original_outputs = [
                    layer_output[batch_idx] for layer_output in original_layer_outputs
                ]

                results[word] = [word_original_outputs]
                results_by_type["base"][word] = [word_original_outputs]

    # Process in batches
    for batch_start in tqdm(range(0, len(word_pairs), batch_size), desc="Processing words"):
        batch_end = min(batch_start + batch_size, len(word_pairs))
        batch = word_pairs[batch_start:batch_end]
        current_batch_size = len(batch)

        # Prepare base inputs for additive representation
        base_inputs = (
            torch.tensor([base_tokens[base_word] for _, base_word, _ in batch])
            .unsqueeze(1)
            .to(device)
        )

        # Add BOS token if needed
        if tokenizer.bos_token_id is not None:
            base_inputs = torch.cat(
                [
                    torch.full((current_batch_size, 1), tokenizer.bos_token_id, device=device),
                    base_inputs,
                ],
                dim=1,
            )

        # Prepare type embeddings
        type_embeddings = torch.zeros((current_batch_size, model.config.hidden_size), device=device)
        for i, (_, _, transforms) in enumerate(batch):
            for transform in transforms:
                type_embeddings[i] += input_type_embeddings[type_names_to_int[transform]]
        # Process additive representations
        with torch.no_grad():
            # Get base embeddings
            base_embeds = model.get_input_embeddings()(base_inputs)

            # Add type embeddings to last position
            base_type_embeds = base_embeds.clone()
            base_type_embeds[:, -1] += type_embeddings
            # Forward pass for additive representation
            additive_outputs = model(
                inputs_embeds=base_type_embeds, output_hidden_states=True, return_dict=True
            )

            # Stack hidden states from all layers for additive representation
            additive_hidden_states = torch.stack(
                [states[:, -1] for states in additive_outputs.hidden_states[:num_layers]]
            )

            # Process original representations if requested
            if compare_to_original:
                # Prepare original word inputs
                original_inputs = tokenizer(
                    [
                        pair[0].replace(space_prefix, " ") if space_prefix else pair[0]
                        for pair in batch
                    ],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                # Forward pass for original words
                original_outputs = model(
                    input_ids=original_inputs.input_ids, output_hidden_states=True, return_dict=True
                )

                # Stack hidden states from all layers for original representation
                original_hidden_states = torch.stack(
                    [states[:, -1] for states in original_outputs.hidden_states[:num_layers]]
                )

            # Process all layers in parallel for additive representation
            additive_layer_outputs = patchscopes.process_hidden_states_all_layers(
                additive_hidden_states
            )

            # Process all layers in parallel for original representation if requested
            if compare_to_original:
                original_layer_outputs = patchscopes.process_hidden_states_all_layers(
                    original_hidden_states
                )

            # Store results
            for batch_idx, (infl_word, _, transforms) in enumerate(batch):
                # Get outputs for all layers for this word
                word_additive_outputs = [
                    layer_output[batch_idx] for layer_output in additive_layer_outputs
                ]
                if compare_to_original:
                    word_original_outputs = [
                        layer_output[batch_idx] for layer_output in original_layer_outputs
                    ]
                else:
                    word_original_outputs = []

                # Store in results dict
                results[infl_word] = [word_additive_outputs, word_original_outputs]

                # Store in results_by_type dict
                for transform in transforms:
                    results_by_type[transform][infl_word] = [
                        word_additive_outputs,
                        word_original_outputs,
                    ]

    return dict(results), dict(results_by_type), word_pairs


def build_word_pairs(
    tokenizer,
    decomposition_map: Dict[str, Dict[str, List[str]]],
    max_words: int = None,
    space_prefix: str = "Ġ",
    single_type_only: bool = False,
    multi_type_only: bool = False,
    skip_multi_token_words: bool = False,
    skip_single_token_words: bool = False,
    ignored_types: List = None,
):

    # Prepare word pairs
    word_pairs = []
    base_words_set = set()

    if ignored_types is None:
        ignored_types = []

    for base_word, inflections in tqdm(decomposition_map.items(), desc="Preparing word pairs"):
        base_words_set.add(base_word)
        for infl_word, transforms in inflections.items():
            filtered_transforms = [t for t in transforms if t not in ignored_types]
            if single_type_only and len(filtered_transforms) != 1:
                continue
            elif multi_type_only and len(filtered_transforms) == 1:
                continue

            if infl_word != base_word:
                infl_encoding = tokenizer.encode(
                    infl_word.replace(space_prefix, " ") if space_prefix else infl_word,
                    add_special_tokens=False,
                )
                if len(infl_encoding) == 1 and not skip_single_token_words:
                    word_pairs.append((infl_word, base_word, filtered_transforms))
                if len(infl_encoding) > 1 and not skip_multi_token_words:
                    word_pairs.append((infl_word, base_word, filtered_transforms))

    # limit words to run logit lens on
    if max_words is not None:
        word_pairs = word_pairs[:max_words]

    return word_pairs


def run_logit_lens_on_hidden_states(
    model,
    tokenizer,
    base_tokens: Dict[str, int],
    decomposition_map: Dict[str, Dict[str, List[str]]],
    input_type_embeddings: torch.Tensor,
    type_names_to_int: Dict[str, int],
    last_layer: int = 10,
    batch_size: int = 32,
    max_words: int = None,
    space_prefix: str = "Ġ",
    single_type_only: bool = False,
    multi_type_only: bool = False,
    skip_multi_token_words: bool = False,
    compare_to_original: bool = True,
    ignored_types: List = None,
    top_k: int = None,
):
    """
    Run logit lens analysis on base+type hidden states representations and optionally original words.
    Decodes hidden states by finding their nearest neighbor in the model's input embedding space.

    Returns:
        Tuple containing:
        - results: Dict mapping word to [additive_decoded, original_decoded] lists
        - results_by_type: Dict mapping type to dict of {word: [additive_decoded, original_decoded]}
        - word_pairs: List of processed (inflection, base_word, types) tuples
        - base_words: List of unique base words processed
    """
    device = model.device
    num_layers = last_layer + 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prepare word pairs
    word_pairs = []
    base_words_set = set()

    if ignored_types is None:
        ignored_types = []

    for base_word, inflections in tqdm(decomposition_map.items(), desc="Preparing word pairs"):
        base_words_set.add(base_word)
        for infl_word, transforms in inflections.items():
            filtered_transforms = [t for t in transforms if t not in ignored_types]
            if single_type_only and len(filtered_transforms) != 1:
                continue
            elif multi_type_only and len(filtered_transforms) == 1:
                continue

            if infl_word != base_word:
                if not skip_multi_token_words:
                    word_pairs.append((infl_word, base_word, filtered_transforms))
                else:
                    infl_encoding = tokenizer.encode(
                        infl_word.replace(space_prefix, " ") if space_prefix else infl_word,
                        add_special_tokens=False,
                    )
                    if len(infl_encoding) == 1:
                        word_pairs.append((infl_word, base_word, filtered_transforms))

    base_words = list(base_words_set)

    # limit words to run logit lens on
    if max_words is not None:
        word_pairs = random.sample(word_pairs, max_words)
        base_words = random.sample(base_words, max_words)

    # Get input embeddings matrix for decoding
    input_embeddings = model.get_input_embeddings().weight.data

    # Initialize results
    results = defaultdict(lambda: [[], []])  # word -> [additive_decoded, original_decoded]
    results_by_type = defaultdict(
        lambda: defaultdict(lambda: [[], []])
    )  # type -> word -> [additive_decoded, original_decoded]

    def decode_hidden_states(hidden_states):
        """
        Decode hidden states by finding their nearest neighbor in input embedding space.

        Args:
            hidden_states (torch.Tensor): Hidden states to decode

        Returns:
            List of decoded tokens (text) for each hidden state
        """
        # Compute inner product between hidden states and input embeddings
        similarities = torch.matmul(hidden_states, input_embeddings.T)

        if top_k is None or top_k <= 1:
            # Original behavior - just get the best match
            nearest_indices = similarities.argmax(dim=1)
            decoded_tokens = tokenizer.batch_decode(nearest_indices)
            return decoded_tokens
        else:
            # Get top-k matches
            top_k_values, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
            decoded_tokens_list = []

            for indices in top_k_indices:
                tokens = tokenizer.batch_decode(indices)
                decoded_tokens_list.append(tokens)

            return decoded_tokens_list

    # Process base words
    for batch_start in tqdm(range(0, len(base_words), batch_size), desc="Processing base words"):
        batch_end = min(batch_start + batch_size, len(base_words))
        batch = base_words[batch_start:batch_end]
        current_batch_size = len(batch)

        with torch.no_grad():
            # Get base embeddings
            # Process original representations if requested
            if compare_to_original:
                original_inputs = tokenizer(
                    [word.replace(space_prefix, " ") if space_prefix else word for word in batch],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                original_outputs = model(
                    input_ids=original_inputs.input_ids, output_hidden_states=True, return_dict=True
                )

                # Stack hidden states from all layers for original representation
                original_hidden_states = torch.stack(
                    [states[:, -1] for states in original_outputs.hidden_states[:num_layers]]
                ).to(model.dtype)

            # Decode original representations
            original_decoded = [
                decode_hidden_states(layer_states) for layer_states in original_hidden_states
            ]
            # Store results
            for batch_idx, word in enumerate(batch):
                if top_k is None or top_k <= 1:
                    # Original behavior
                    word_original_decoded = [
                        layer_decoded[batch_idx] for layer_decoded in original_decoded
                    ]
                else:
                    # Top-k behavior
                    word_original_decoded = [
                        layer_decoded[batch_idx] for layer_decoded in original_decoded
                    ]

                results[word] = [word_original_decoded]
                results_by_type["base"][word] = [word_original_decoded]

    # Process in batches
    for batch_start in tqdm(range(0, len(word_pairs), batch_size), desc="Processing words"):
        batch_end = min(batch_start + batch_size, len(word_pairs))
        batch = word_pairs[batch_start:batch_end]
        current_batch_size = len(batch)

        # Prepare base inputs for additive representation
        base_inputs = (
            torch.tensor([base_tokens[base_word] for _, base_word, _ in batch])
            .unsqueeze(1)
            .to(device)
        )

        # Add BOS token if needed
        if tokenizer.bos_token_id is not None:
            base_inputs = torch.cat(
                [
                    torch.full((current_batch_size, 1), tokenizer.bos_token_id, device=device),
                    base_inputs,
                ],
                dim=1,
            )

        # Prepare type embeddings
        type_embeddings = torch.zeros((current_batch_size, model.config.hidden_size), device=device)
        for i, (_, _, transforms) in enumerate(batch):
            for transform in transforms:
                type_embeddings[i] += input_type_embeddings[type_names_to_int[transform]]

        # Process additive representations
        with torch.no_grad():
            # Get base embeddings
            base_embeds = model.get_input_embeddings()(base_inputs)

            # Add type embeddings to last position
            base_type_embeds = base_embeds.clone()
            base_type_embeds[:, -1] += type_embeddings

            # Forward pass for additive representation
            additive_outputs = model(
                inputs_embeds=base_type_embeds, output_hidden_states=True, return_dict=True
            )

            # Stack hidden states from all layers for additive representation
            additive_hidden_states = torch.stack(
                [states[:, -1] for states in additive_outputs.hidden_states[:num_layers]]
            ).to(model.dtype)

            # Process original representations if requested
            if compare_to_original:
                # Prepare original word inputs
                original_inputs = tokenizer(
                    [
                        pair[0].replace(space_prefix, " ") if space_prefix else pair[0]
                        for pair in batch
                    ],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                # Forward pass for original words
                original_outputs = model(
                    input_ids=original_inputs.input_ids, output_hidden_states=True, return_dict=True
                )

                # Stack hidden states from all layers for original representation
                original_hidden_states = torch.stack(
                    [states[:, -1] for states in original_outputs.hidden_states[:num_layers]]
                ).to(model.dtype)

            # Decode additive representations
            additive_decoded = [
                decode_hidden_states(layer_states) for layer_states in additive_hidden_states
            ]

            # Decode original representations if requested
            if compare_to_original:
                original_decoded = [
                    decode_hidden_states(layer_states) for layer_states in original_hidden_states
                ]
            else:
                original_decoded = []

            # Store results
            for batch_idx, (infl_word, _, transforms) in enumerate(batch):
                # Get decoded tokens for all layers for this word
                if top_k is None or top_k <= 1:
                    # Original behavior
                    word_additive_decoded = [
                        layer_decoded[batch_idx] for layer_decoded in additive_decoded
                    ]
                    word_original_decoded = (
                        [layer_decoded[batch_idx] for layer_decoded in original_decoded]
                        if compare_to_original
                        else []
                    )
                else:
                    # Top-k behavior - each layer result is already a list of tokens
                    word_additive_decoded = [
                        layer_decoded[batch_idx] for layer_decoded in additive_decoded
                    ]
                    word_original_decoded = (
                        [layer_decoded[batch_idx] for layer_decoded in original_decoded]
                        if compare_to_original
                        else []
                    )

                # Store in results dict
                results[infl_word] = [word_additive_decoded, word_original_decoded]

                # Store in results_by_type dict
                for transform in transforms:
                    results_by_type[transform][infl_word] = [
                        word_additive_decoded,
                        word_original_decoded,
                    ]

    return dict(results), dict(results_by_type), word_pairs


def filter_proper_nouns_from_patchscopes_results(
    results: Dict[str, List[List[str]]],
    results_by_type: Dict[str, Dict[str, List[List[str]]]],
    ner_cache: Dict[str, bool],
) -> Tuple[Dict[str, List[List[str]]], Dict[str, Dict[str, List[List[str]]]]]:
    """
    Filter out proper nouns from patchscopes results using NER cache.

    This is useful when reusing patchscopes cache from a run without --dont_decompose_proper_nouns
    when running with that flag. It removes proper nouns identified via NER from the results.

    Args:
        results: Dict mapping words to [additive_outputs, original_outputs]
        results_by_type: Dict mapping types to words and their outputs
        ner_cache: Dictionary containing NER results (word -> is_proper_noun bool)

    Returns:
        Tuple of (filtered_results, filtered_results_by_type)
    """

    # Import the NER check function from final_decomposition_utils
    # We'll use the cache directly to avoid re-running NER
    def is_proper_noun_from_cache(word: str, ner_cache: Dict[str, bool]) -> bool:
        """Check if word is a proper noun using the NER cache."""
        if not word:
            return False
        # Strip space prefix for lookup
        word_clean = word.lstrip("Ġ ").strip()
        if not word_clean:
            return False
        # Check cache (default to False if not in cache)
        return ner_cache.get(word_clean, False)

    # Filter results dict - remove proper nouns
    filtered_results = {
        word: outputs
        for word, outputs in results.items()
        if not is_proper_noun_from_cache(word, ner_cache)
    }

    # Filter results_by_type dict
    filtered_results_by_type = {}
    for type_name, type_words in results_by_type.items():
        # Filter out proper nouns from this type
        filtered_type_words = {
            word: outputs
            for word, outputs in type_words.items()
            if not is_proper_noun_from_cache(word, ner_cache)
        }

        # Only include the type if it still has words after filtering
        if filtered_type_words:
            filtered_results_by_type[type_name] = filtered_type_words

    return filtered_results, filtered_results_by_type


def analyze_patchscopes_results(
    results: Dict[str, List[List[str]]],
    results_by_type: Dict[str, Dict[str, List[List[str]]]],
    num_layers: int,
    word_pairs: List = None,
    tokenizer=None,
    ignored_types: List = None,
    space_prefix=None,
    top_k: int = None,
):
    """
    Analyzes word identification patterns in patchscopes results, with support for top-k predictions.

    Args:
        results: Dict mapping words to their patchscopes outputs
        results_by_type: Dict mapping types to words and their outputs
        num_layers: Number of layers analyzed
        word_pairs: List of (inflection, base_word, types) tuples
        tokenizer: Tokenizer used for encoding words
        ignored_types: List of types to ignore
        space_prefix: Prefix for space tokens (e.g., "Ġ")
        top_k: Number of top predictions to consider

    Returns:
        Dict containing analysis results structured for DataFrame conversion with fields:
        - word: The word being analyzed
        - type: The transformation type (or 'BASE' for base words)
        - identified_additive: Whether word was identified as the top-1 prediction in additive representation
        - first_layer_additive: First layer where word was identified as top-1 (additive)
        - layers_identified_additive: List of layers where word was identified as top-1 (additive)
        - identified_original: Whether word was identified as the top-1 prediction in original representation
        - first_layer_original: First layer where word was identified as top-1 (original)
        - layers_identified_original: List of layers where word was identified as top-1 (original)
        - identified_additive_topk: Whether word was identified in top-k predictions in additive representation
        - first_layer_additive_topk: First layer where word was identified in top-k (additive)
        - layers_identified_additive_topk: List of layers where word was identified in top-k (additive)
        - identified_original_topk: Whether word was identified in top-k predictions in original representation
        - first_layer_original_topk: First layer where word was identified in top-k (original)
        - layers_identified_original_topk: List of layers where word was identified in top-k (original)
    """
    analysis_results = list()
    analysis_results_by_type = defaultdict(list)
    analysis_summary_by_type = dict()
    resolution_detailed = list()

    word_to_types = dict()
    if word_pairs is not None:
        for word, base_word, types_list in word_pairs:
            if ignored_types is not None:
                word_to_types[word] = [t for t in types_list if t not in ignored_types]
            else:
                word_to_types[word] = types_list

    def analyze_outputs(
        outputs: List[List[str]], word: str
    ) -> Tuple[bool, int, List[int], bool, int, List[int]]:
        """Helper function to analyze a set of outputs across layers.

        Returns:
            Tuple containing:
            - identified_top1: Whether word was identified as top-1 prediction
            - first_layer_top1: First layer where word was identified as top-1
            - layers_identified_top1: List of layers where word was identified as top-1
            - identified_topk: Whether word was identified in top-k predictions
            - first_layer_topk: First layer where word was identified in top-k
            - layers_identified_topk: List of layers where word was identified in top-k
        """
        if space_prefix:
            word = word.replace(space_prefix, " ")
        word_stripped = word.strip()

        # Top-1 analysis (compatible with original function)
        identified_top1 = False
        first_layer_top1 = num_layers  # Default to num_layers if never identified
        layers_identified_top1 = []

        # Top-k analysis
        identified_topk = False
        first_layer_topk = num_layers
        layers_identified_topk = []

        for layer_idx, layer_output in enumerate(outputs):
            if not layer_output:  # Skip empty outputs
                continue
            if layer_idx == num_layers:
                break

            # Handle both single string and list of strings (top-k) cases
            if isinstance(layer_output, str):
                # Top-1 case (original behavior)
                if layer_output.strip().startswith(word_stripped):
                    identified_top1 = True
                    identified_topk = True  # If it's in top-1, it's in top-k
                    layers_identified_top1.append(layer_idx)
                    layers_identified_topk.append(layer_idx)
                    if layer_idx < first_layer_top1:
                        first_layer_top1 = layer_idx
                    if layer_idx < first_layer_topk:
                        first_layer_topk = layer_idx
            else:
                # Top-k case (list of strings)
                # Check top-1 (first element)
                if layer_output[0].strip().startswith(word_stripped):
                    identified_top1 = True
                    layers_identified_top1.append(layer_idx)
                    if layer_idx < first_layer_top1:
                        first_layer_top1 = layer_idx

                # Check if word appears in any of the top-k predictions
                for pred in layer_output:
                    if pred.strip().startswith(word_stripped):
                        identified_topk = True
                        layers_identified_topk.append(layer_idx)
                        if layer_idx < first_layer_topk:
                            first_layer_topk = layer_idx
                        break  # Found in this layer, no need to check other predictions

        if not identified_top1:
            first_layer_top1 = -1
        if not identified_topk:
            first_layer_topk = -1

        return (
            identified_top1,
            first_layer_top1,
            layers_identified_top1,
            identified_topk,
            first_layer_topk,
            layers_identified_topk,
        )

    # Analyze by type
    for type_name, type_results in results_by_type.items():
        for word, patchscopes_outputs in type_results.items():
            if len(patchscopes_outputs) == 1:
                original_outputs = patchscopes_outputs[0]
                additive_outputs = None
            else:
                additive_outputs, original_outputs = patchscopes_outputs

            if additive_outputs:
                (
                    identified_additive,
                    first_layer_additive,
                    layers_identified_additive,
                    identified_additive_topk,
                    first_layer_additive_topk,
                    layers_identified_additive_topk,
                ) = analyze_outputs(additive_outputs, word)
            else:
                identified_additive, first_layer_additive, layers_identified_additive = (
                    None,
                    None,
                    None,
                )
                (
                    identified_additive_topk,
                    first_layer_additive_topk,
                    layers_identified_additive_topk,
                ) = None, None, None

            if original_outputs:  # Only analyze original if available
                (
                    identified_original,
                    first_layer_original,
                    layers_identified_original,
                    identified_original_topk,
                    first_layer_original_topk,
                    layers_identified_original_topk,
                ) = analyze_outputs(original_outputs, word)
            else:
                identified_original, first_layer_original, layers_identified_original = (
                    None,
                    None,
                    None,
                )
                (
                    identified_original_topk,
                    first_layer_original_topk,
                    layers_identified_original_topk,
                ) = None, None, None

            word_n_tokens = None
            if tokenizer is not None:
                if space_prefix:
                    word_n_tokens = len(
                        tokenizer.encode(word.replace(space_prefix, " "), add_special_tokens=False)
                    )
                else:
                    word_n_tokens = len(tokenizer.encode(word, add_special_tokens=False))

            # Base analysis (compatible with original)
            result_dict = {
                "word": word,
                "type": type_name,
                "identified_additive": identified_additive,
                "first_layer_additive": first_layer_additive,
                "layers_identified_additive": layers_identified_additive,
                "identified_original": identified_original,
                "first_layer_original": first_layer_original,
                "layers_identified_original": layers_identified_original,
                "word_n_tokens": word_n_tokens,
            }

            # Add top-k analysis fields only if top_k is specified
            if top_k is not None and top_k > 1:
                result_dict.update(
                    {
                        "identified_additive_topk": identified_additive_topk,
                        "first_layer_additive_topk": first_layer_additive_topk,
                        "layers_identified_additive_topk": layers_identified_additive_topk,
                        "identified_original_topk": identified_original_topk,
                        "first_layer_original_topk": first_layer_original_topk,
                        "layers_identified_original_topk": layers_identified_original_topk,
                    }
                )

            analysis_results.append(result_dict)

            type_result_dict = result_dict.copy()
            type_result_dict.update(
                {
                    "is_single_type": (len(word_to_types.get(word, [])) <= 1)
                    if word_to_types
                    else None,
                }
            )
            analysis_results_by_type[type_name].append(type_result_dict)

            resolution_dict = {
                "word": word,
                "type": type_name,
                "word_n_tokens": word_n_tokens,
                "is_single_type": (len(word_to_types.get(word, [])) <= 1)
                if word_to_types
                else None,
                "is_two_token": (word_n_tokens == 2) if (word_n_tokens is not None) else None,
                "is_multi_token": (word_n_tokens > 1) if (word_n_tokens is not None) else None,
                "first_layer_additive": first_layer_additive
                if identified_additive == True
                else None,
                "first_layer_original": first_layer_original
                if identified_original == True
                else None,
            }

            # Add top-k resolution fields
            if top_k is not None and top_k > 1:
                resolution_dict.update(
                    {
                        "first_layer_additive_topk": first_layer_additive_topk
                        if identified_additive_topk == True
                        else None,
                        "first_layer_original_topk": first_layer_original_topk
                        if identified_original_topk == True
                        else None,
                    }
                )

            resolution_detailed.append(resolution_dict)

        # Calculate summary statistics
        curr_type_df = pd.DataFrame(analysis_results_by_type[type_name])
        is_single_type_mask = curr_type_df["is_single_type"] == True
        is_multi_type_mask = curr_type_df["is_single_type"] == False
        is_multi_token = curr_type_df["word_n_tokens"] > 1
        is_two_token = curr_type_df["word_n_tokens"] == 2
        is_single_token = curr_type_df["word_n_tokens"] == 1
        curr_single_token_df = curr_type_df[is_single_token]
        curr_two_token_df = curr_type_df[is_two_token]
        curr_multi_token_df = curr_type_df[is_multi_token]
        curr_single_token_single_type_df = curr_type_df[is_single_token & is_single_type_mask]
        curr_single_token_multi_type_df = curr_type_df[is_single_token & is_multi_type_mask]
        curr_multi_token_single_type_df = curr_type_df[is_multi_token & is_single_type_mask]
        curr_multi_token_multi_type_df = curr_type_df[is_multi_token & is_multi_type_mask]

        # Base statistics
        summary_dict = {
            "additive_id": float(curr_type_df["identified_additive"].mean())
            if "identified_additive" in curr_type_df
            else None,
            "orig_id": float(curr_type_df["identified_original"].mean())
            if "identified_original" in curr_type_df
            else None,
            "N": len(curr_type_df),
            "additive_id_1_token": float(curr_single_token_df["identified_additive"].mean())
            if "identified_additive" in curr_single_token_df
            else None,
            "orig_id_1_token": float(curr_single_token_df["identified_original"].mean())
            if "identified_original" in curr_single_token_df
            else None,
            "N_1_token": (is_single_token).sum(),
            "additive_id_2_token": float(curr_two_token_df["identified_additive"].mean())
            if "identified_additive" in curr_two_token_df
            else None,
            "orig_id_2_token": float(curr_two_token_df["identified_original"].mean())
            if "identified_original" in curr_two_token_df
            else None,
            "N_2_token": (is_two_token).sum(),
            "additive_id_m_token": float(curr_multi_token_df["identified_additive"].mean())
            if "identified_additive" in curr_multi_token_df
            else None,
            "orig_id_m_token": float(curr_multi_token_df["identified_original"].mean())
            if "identified_original" in curr_multi_token_df
            else None,
            "N_m_token": (is_multi_token).sum(),
            "additive_id_1_token_1_type": float(
                curr_single_token_single_type_df["identified_additive"].mean()
            )
            if "identified_additive" in curr_single_token_single_type_df
            else None,
            "orig_id_1_token_1_type": float(
                curr_single_token_single_type_df["identified_original"].mean()
            )
            if "identified_original" in curr_single_token_single_type_df
            else None,
            "N_1_token_1_type": (is_single_token & is_single_type_mask).sum(),
            "additive_id_1_token_m_type": float(
                curr_single_token_multi_type_df["identified_additive"].mean()
            )
            if "identified_additive" in curr_single_token_multi_type_df
            else None,
            "orig_id_1_token_m_type": float(
                curr_single_token_multi_type_df["identified_original"].mean()
            )
            if "identified_original" in curr_single_token_multi_type_df
            else None,
            "N_1_token_m_type": (is_single_token & is_multi_type_mask).sum(),
            "additive_id_m_token_1_type": float(
                curr_multi_token_single_type_df["identified_additive"].mean()
            )
            if "identified_additive" in curr_multi_token_single_type_df
            else None,
            "orig_id_m_token_1_type": float(
                curr_multi_token_single_type_df["identified_original"].mean()
            )
            if "identified_original" in curr_multi_token_single_type_df
            else None,
            "N_m_token_1_type": (is_multi_token & is_single_type_mask).sum(),
            "additive_id_m_token_m_type": float(
                curr_multi_token_multi_type_df["identified_additive"].mean()
            )
            if "identified_additive" in curr_multi_token_multi_type_df
            else None,
            "orig_id_m_token_m_type": float(
                curr_multi_token_multi_type_df["identified_original"].mean()
            )
            if "identified_original" in curr_multi_token_multi_type_df
            else None,
            "N_m_token_m_type": (is_multi_token & is_multi_type_mask).sum(),
            # 'additive_id_multi_token': curr_multi_token_type_df[
            #     'identified_additive'].mean() if 'identified_additive' in curr_multi_token_type_df else None,
            # 'orig_id_multi_token': curr_multi_token_type_df[
            #     'identified_original'].mean() if 'identified_original' in curr_multi_token_type_df else None,
            # 'N_multi_token': is_multi_token.sum(),
        }

        analysis_summary_by_type[type_name] = summary_dict

    return (
        analysis_results,
        dict(analysis_results_by_type),
        analysis_summary_by_type,
        resolution_detailed,
    )
