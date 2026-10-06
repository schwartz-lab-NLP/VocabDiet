"""Multilingual UniMorph-based vocabulary decomposition."""

import os
import re
import nltk
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, DefaultDict
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tempfile
import json
from itertools import chain
from copy import deepcopy

pass
from .english_morphology import UniMorphAnalyzer


# Constants for transformation types
NO_PREFIX_TRANSFORM = "no_space_prefix"
REMOVE_SPACE_PREFIX_TRANSFORM = "remove_space_prefix"
NO_CAPITALIZATION_TRANSFORM = "no_capitalization"
ADD_CAPITALIZATION_TRANSFORM = "add_capitalization"
REMOVE_CAPITALIZATION_TRANSFORM = "remove_capitalization"
NO_POS_INFLECTION = "none"
NO_EMBEDDING_TYPES = [NO_PREFIX_TRANSFORM, NO_POS_INFLECTION, NO_CAPITALIZATION_TRANSFORM]
IGNORE_MULTI_TYPES_IN_INIT = []
FORBID_MULTI_TYPES_IN_INIT = [
    ADD_CAPITALIZATION_TRANSFORM,
]


# Load necessary resources


def get_unimorph_paths(language_code, unimorph_base_path=None):
    """
    Get UniMorph file paths for a given language code.

    Args:
        language_code: ISO 639-3 language code (e.g., 'eng', 'spa', 'fra')
        unimorph_base_path: Base path to UniMorph data directory

    Returns:
        tuple: (inflections_path, derivations_path) or (inflections_path, None) if no derivations
    """
    unimorph_base_path = unimorph_base_path or os.environ.get("UNIMORPH_ROOT", "resources/unimorph")
    lang_dir = os.path.join(unimorph_base_path, language_code)

    # Standard inflections file
    inflections_path = os.path.join(lang_dir, language_code)

    if not os.path.exists(inflections_path):
        raise FileNotFoundError(f"UniMorph inflections file not found: {inflections_path}")

    # Derivations file (not available for all languages)
    derivations_path = os.path.join(lang_dir, f"{language_code}.derivations.tsv")

    if not os.path.exists(derivations_path):
        derivations_path = os.path.join(lang_dir, f"{language_code}.derivations")
        if not os.path.exists(derivations_path):
            derivations_path = None

    return inflections_path, derivations_path


def get_multilingual_variation_transformations(
    word,
    base_form=None,
    transforms=None,
    space_prefix=" ",
    decompose_spaces=True,
    decompose_capitalization=True,
    prefix_transforms=None,
    suffix_transforms=None,
    prefix_transform_names=None,
    suffix_transform_names=None,
):
    """
    Get variations of a word with transformation tags for multilingual support.

    Args:
        word: The word to transform
        base_form: The original base form for comparison
        transforms: Existing transforms to append to
        space_prefix: The space prefix character
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation
        prefix_transforms: List of prefixes to add as transformations
        suffix_transforms: List of suffixes to add as transformations
        prefix_transform_names: Optional list of names for prefix transforms
        suffix_transform_names: Optional list of names for suffix transforms

    Returns:
        Dict mapping word variations to their transformation tags
    """
    result = {}
    has_space_prefix = word.startswith(space_prefix)
    word_without_prefix = word[len(space_prefix) :] if has_space_prefix else word
    is_capitalized = word_without_prefix and word_without_prefix[0].isupper()

    # Base form transformation tracking
    space_transform = NO_PREFIX_TRANSFORM
    cap_transform = NO_CAPITALIZATION_TRANSFORM

    # Set transformation types
    if base_form:
        base_has_space_prefix = base_form.startswith(space_prefix)
        base_word_without_prefix = (
            base_form[len(space_prefix) :] if base_has_space_prefix else base_form
        )
        base_is_capitalized = base_word_without_prefix and base_word_without_prefix[0].isupper()

        # Determine space transformation
        if decompose_spaces:
            if has_space_prefix != base_has_space_prefix:
                space_transform = (
                    NO_PREFIX_TRANSFORM if has_space_prefix else REMOVE_SPACE_PREFIX_TRANSFORM
                )

        # Determine capitalization transformation
        if decompose_capitalization:
            if is_capitalized != base_is_capitalized:
                cap_transform = (
                    ADD_CAPITALIZATION_TRANSFORM
                    if is_capitalized
                    else REMOVE_CAPITALIZATION_TRANSFORM
                )

    # Initialize transforms
    transforms = [] if transforms is None else transforms
    if not transforms:
        transforms.append(NO_POS_INFLECTION)

    base_transforms = [t for t in transforms]
    base_transforms.append(space_transform)
    base_transforms.append(cap_transform)

    result[word] = base_transforms

    # Generate basic variations (space and capitalization)
    if decompose_spaces and has_space_prefix:
        word_var = word_without_prefix
        var_transforms = [REMOVE_SPACE_PREFIX_TRANSFORM]
        var_transforms.append(cap_transform)
        result[word_var] = transforms + var_transforms

    if decompose_capitalization and not is_capitalized:
        word_var = (space_prefix if has_space_prefix else "") + word_without_prefix.capitalize()
        var_transforms = [ADD_CAPITALIZATION_TRANSFORM]
        var_transforms.append(space_transform)
        result[word_var] = transforms + var_transforms

    # Add other combinations for space and capitalization
    if decompose_spaces and decompose_capitalization:
        if has_space_prefix and not is_capitalized:
            word_var = word_without_prefix.capitalize()
            result[word_var] = transforms + [
                REMOVE_SPACE_PREFIX_TRANSFORM,
                ADD_CAPITALIZATION_TRANSFORM,
            ]
        elif not has_space_prefix and not is_capitalized:
            word_var = space_prefix + word_without_prefix.capitalize()
            result[word_var] = transforms + [NO_PREFIX_TRANSFORM, ADD_CAPITALIZATION_TRANSFORM]

    # Generate prefix and suffix variations
    if prefix_transforms or suffix_transforms:
        prefix_transforms = prefix_transforms or []
        suffix_transforms = suffix_transforms or []

        # Use custom names if provided, otherwise use the transform strings themselves
        prefix_names = (
            prefix_transform_names
            if prefix_transform_names
            else [f"PREFIX_{p}" for p in prefix_transforms]
        )
        suffix_names = (
            suffix_transform_names
            if suffix_transform_names
            else [f"SUFFIX_{s}" for s in suffix_transforms]
        )

        # Generate all current variations as base for prefix/suffix combinations
        current_variations = dict(result)

        for base_var, base_var_transforms in current_variations.items():
            var_has_space_prefix = base_var.startswith(space_prefix)
            var_without_space = base_var[len(space_prefix) :] if var_has_space_prefix else base_var

            # Add prefix variations
            for i, prefix in enumerate(prefix_transforms):
                prefix_name = prefix_names[i]

                # Add prefix after space if present
                if var_has_space_prefix:
                    prefixed_word = space_prefix + prefix + var_without_space
                else:
                    prefixed_word = prefix + base_var

                if prefixed_word not in result:
                    result[prefixed_word] = base_var_transforms + [prefix_name]

            # Add suffix variations
            for i, suffix in enumerate(suffix_transforms):
                suffix_name = suffix_names[i]
                suffixed_word = base_var + suffix

                if suffixed_word not in result:
                    result[suffixed_word] = base_var_transforms + [suffix_name]

            # Add prefix + suffix combinations
            for i, prefix in enumerate(prefix_transforms):
                for j, suffix in enumerate(suffix_transforms):
                    prefix_name = prefix_names[i]
                    suffix_name = suffix_names[j]

                    if var_has_space_prefix:
                        combined_word = space_prefix + prefix + var_without_space + suffix
                    else:
                        combined_word = prefix + base_var + suffix

                    if combined_word not in result:
                        result[combined_word] = base_var_transforms + [prefix_name, suffix_name]

    return result


def is_valid_multilingual_word(word, min_length=2, language_code="eng"):
    """
    Check if a word is valid for processing in the given language.
    This is a simplified version that doesn't rely on language-specific resources.

    Args:
        word: Word to check
        min_length: Minimum word length
        language_code: Language code for language-specific rules

    Returns:
        bool: True if word is valid for processing
    """
    # Basic length check
    if len(word) < min_length:
        return False

    # Language-specific checks can be added here
    # For now, we'll use a simple approach

    return True


def get_multilingual_vocabulary_decomposition(
    tokenizer,
    language_code="eng",
    unimorph_base_path=None,
    space_prefix=" ",
    include_base_only=False,
    include_all_caps=False,
    override_unimorph_with_inflections=True,
    decompose_spaces=False,
    decompose_capitalization=True,
    skip_if_no_space_prefix=True,
    include_derivations=False,
    use_type_str_as_type=False,
    min_derivation_count=1000,
    prefix_transforms=None,
    suffix_transforms=None,
    prefix_transform_names=None,
    suffix_transform_names=None,
    min_word_length=2,
    no_diacritics=False,
):
    """
    Create a comprehensive token decomposition map for any language using UniMorph data.

    Args:
        tokenizer: Hugging Face tokenizer
        language_code: ISO 639-3 language code (e.g., 'eng', 'spa', 'fra', 'ara', 'heb')
        unimorph_base_path: Base path to UniMorph data directory
        space_prefix: Character used for space prefix
        include_base_only: Whether to include words with no inflections
        include_all_caps: Whether to include all caps words
        override_unimorph_with_inflections: Whether to treat words as inflections even if they appear as base forms
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation
        include_derivations: Whether to include derivations
        prefix_transforms: List of prefixes to add as transformations (e.g., ['ב', 'ל', 'כ'] for Hebrew)
        suffix_transforms: List of suffixes to add as transformations
        prefix_transform_names: Optional custom names for prefix transforms
        suffix_transform_names: Optional custom names for suffix transforms
        min_word_length: Minimum word length to consider

    Returns:
        Tuple containing:
        - base_tokens: Dict mapping base forms to token IDs
        - decomposition_map: Dict mapping base forms to {inflection: [types]}
        - type_to_words: Dict mapping transformation types to lists of words
        - ambiguous_inflections: Dict of ambiguous inflections
        - inflection_baseform_conflicts: List of conflicts between base forms and inflections
        - filtered: Dict containing filtered items
    """
    # Get UniMorph file paths
    unimorph_inflections_path, unimorph_derivations_path = get_unimorph_paths(
        language_code, unimorph_base_path
    )

    # Initialize analyzers
    print(f"Loading UniMorph data for {language_code}...")
    unimorph = UniMorphAnalyzer(
        unimorph_inflections_path, unimorph_derivations_path, no_diacritics=no_diacritics
    )

    # Set common derivation types if needed
    if include_derivations and unimorph_derivations_path:
        unimorph.set_common_derivation_types(min_count=min_derivation_count)

    # Initialize result containers
    base_tokens = {}
    decomposition_map = {}
    processed_tokens = set()
    ambiguous_inflections = {}
    type_to_words = defaultdict(set)
    inflection_baseform_conflicts = []
    duplicate_tags_inflections = defaultdict(lambda: defaultdict(set))
    filtered_duplicate_inflections = []
    filtered_short_words = set()
    filtered_invalid_words = set()

    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    vocab_items = sorted(vocab.items(), key=lambda x: x[1])

    # Process vocabulary
    print("Processing vocabulary...")
    for token, token_id in tqdm(vocab_items):
        # Get string representation
        token_str = tokenizer.convert_tokens_to_string([token])
        token_repr = token_str.strip(space_prefix)
        token_has_space_prefix = token_str.startswith(space_prefix)
        token_is_capitalized = token_repr and token_repr[0].isupper()

        processed_tokens.add(token_str)

        # Skip if not valid word
        if not is_valid_multilingual_word(token_repr, min_word_length, language_code):
            if len(token_repr) < min_word_length:
                filtered_short_words.add(token_str)
            else:
                filtered_invalid_words.add(token_str)
            continue

        # Skip if we're decomposing spaces and this token doesn't have a space prefix
        if skip_if_no_space_prefix and not token_has_space_prefix:
            continue

        # Skip all caps if requested
        if not include_all_caps and token_repr.isupper():
            continue

        # Preserve original capitalization when decompose_capitalization is False
        original_token_repr = token_repr
        original_token_str = token_str

        # Check if we're dealing with a capitalized form when decompose_capitalization is False
        is_capitalized = token_repr != token_repr.lower() and not token_repr.isupper()
        should_preserve_caps = is_capitalized and not decompose_capitalization

        # Try with original capitalization first when needed, otherwise use lowercase
        token_repr_for_lookup = token_repr if should_preserve_caps else token_repr.lower()

        # Check if token represents a base word in UniMorph
        is_base_form = unimorph.is_lemma(token_repr_for_lookup)
        if should_preserve_caps and not is_base_form:
            token_repr_for_lookup = token_repr.lower()
            lowercase_is_base = unimorph.is_lemma(token_repr_for_lookup)
            if lowercase_is_base:
                is_base_form = True

        # Skip if not a base form
        if not is_base_form:
            continue

        # Skip if we're decomposing capitalization and this token is capitalized
        if decompose_capitalization and token_is_capitalized:
            continue

        # Skip if already in decomposition map
        if token_str in decomposition_map:
            continue

        # Start building decomposition for this token
        inflections_map = {}
        has_inflections = False

        if is_base_form:
            # Generate base form variations
            base_variations = get_multilingual_variation_transformations(
                token_str,
                None,
                None,
                space_prefix,
                decompose_spaces,
                decompose_capitalization,
                prefix_transforms,
                suffix_transforms,
                prefix_transform_names,
                suffix_transform_names,
            )

            for var, transforms in base_variations.items():
                if var not in inflections_map:
                    inflections_map[var] = transforms
                    has_inflections = True

            # Get inflections from UniMorph
            unimorph_inflections = unimorph.get_inflections(token_repr_for_lookup)

            # Function to apply capitalization pattern from original to target string
            def apply_capitalization_pattern(original, target):
                if original and len(original) > 0 and original[0].isupper() and len(target) > 0:
                    return target[0].upper() + target[1:]
                return target

            # Process UniMorph inflections
            for inflected_form, tag_sets in unimorph_inflections.items():
                if inflected_form == token_repr_for_lookup:
                    continue

                # Apply capitalization pattern if needed
                if should_preserve_caps:
                    inflected_form = apply_capitalization_pattern(
                        original_token_repr, inflected_form
                    )

                # Clean up problematic tags (language-specific rules can be added here)
                if language_code == "eng":
                    # English-specific tag cleaning
                    if ("V", "NFIN", "IMP+SBJV") in tag_sets and len(tag_sets) > 1:
                        tag_sets.remove(("V", "NFIN", "IMP+SBJV"))

                    if ("V", "V.PTCP", "PST") in tag_sets and ("V", "PST") in tag_sets:
                        tag_sets.remove(("V", "V.PTCP", "PST"))

                    # Handle N;PL+V;PRS;3;SG ambiguity
                    if len(tag_sets) == 1 and ";".join(list(tag_sets)[0]) == "V;PRS;3;SG":
                        # For multilingual version, we skip this wordnet-based check
                        pass

                combined_tags = []
                separate_tags = set()
                for tag_set in tag_sets:
                    tag_str = ";".join(tag_set)
                    combined_tags.append(tag_str)
                    separate_tags.update(set(tag_set))
                if len(tag_sets) > 1:
                    tag_str = "+".join(sorted(combined_tags))

                # Check for ambiguous inflections
                # Check for conflicts with other base forms
                if inflected_form.lower() in unimorph.lemma_to_forms:
                    inflection_baseform_conflicts.append(
                        (original_token_repr, inflected_form, tag_str)
                    )
                    if not override_unimorph_with_inflections:
                        continue

                # Store mapping for tracking duplicates
                for tag_set in tag_sets:
                    tag_str_single = ";".join(sorted(tag_set))
                    duplicate_tags_inflections[token_repr_for_lookup][tag_str_single].add(
                        inflected_form.lower()
                    )

                # Check for duplicate tag inflections
                is_duplicate = False
                for tag_set in tag_sets:
                    tag_str_single = ";".join(sorted(tag_set))
                    if len(duplicate_tags_inflections[token_repr_for_lookup][tag_str_single]) > 1:
                        is_duplicate = True
                        filtered_info = (
                            original_token_repr,
                            inflected_form,
                            tag_str_single,
                            list(duplicate_tags_inflections[token_repr_for_lookup][tag_str_single]),
                        )
                        filtered_duplicate_inflections.append(filtered_info)
                        break

                if is_duplicate:
                    continue

                # Add inflection with proper space prefix
                inflected_with_prefix = (
                    space_prefix + inflected_form if token_has_space_prefix else inflected_form
                )

                # Get variations of the inflected form
                inflected_variations = get_multilingual_variation_transformations(
                    inflected_with_prefix,
                    token_str,
                    [tag_str],
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                    prefix_transforms,
                    suffix_transforms,
                    prefix_transform_names,
                    suffix_transform_names,
                )

                # Add all variations
                for var, transforms in inflected_variations.items():
                    if var not in inflections_map:
                        inflections_map[var] = transforms
                        has_inflections = True

            # Get derivations if requested
            if include_derivations and unimorph_derivations_path:
                derivations = unimorph.get_root_derivations(token_repr_for_lookup)
                for derived_form, type_pos_sets in derivations.items():
                    if should_preserve_caps:
                        derived_form = apply_capitalization_pattern(
                            original_token_repr, derived_form
                        )

                    for deriv_type, pos in type_pos_sets:
                        if derived_form.lower() in unimorph.lemma_to_forms:
                            inflection_baseform_conflicts.append(
                                (original_token_repr, derived_form, deriv_type)
                            )
                            if not override_unimorph_with_inflections:
                                continue

                        derived_with_prefix = (
                            space_prefix + derived_form if token_has_space_prefix else derived_form
                        )

                        derived_variations = get_multilingual_variation_transformations(
                            derived_with_prefix,
                            token_str,
                            [deriv_type],
                            space_prefix,
                            decompose_spaces,
                            decompose_capitalization,
                            prefix_transforms,
                            suffix_transforms,
                            prefix_transform_names,
                            suffix_transform_names,
                        )

                        for var, transforms in derived_variations.items():
                            if var not in inflections_map:
                                inflections_map[var] = transforms
                                has_inflections = True

        # Check if we have inflections
        if not has_inflections and not include_base_only:
            continue

        # Add to decomposition map
        decomposition_map[token_str] = {}
        for form, transforms in inflections_map.items():
            # Skip problematic cases
            if language_code == "eng":
                if "N;SG" in transforms:
                    print(f"Skipping problematic case: {token_str} -> {form}")
                    continue

            types_list = transforms
            types_str = "+".join(sorted(transforms))

            # Check for ambiguous inflections
            if form in decomposition_map[token_str]:
                ambiguous_key = f"{token_str} -> {form}"
                if ambiguous_key not in ambiguous_inflections:
                    ambiguous_inflections[ambiguous_key] = []
                ambiguous_inflections[ambiguous_key].append((types_list, types_str))
                continue

            # Add to map
            if use_type_str_as_type:
                decomposition_map[token_str][form] = [types_str]
            else:
                decomposition_map[token_str][form] = types_list

            processed_tokens.add(form)

            # Populate type_to_words
            if use_type_str_as_type:
                type_to_words[types_str].add(form)
            else:
                for type_tag in types_list:
                    type_to_words[type_tag].add(form)

        # Add to base tokens
        base_tokens[token_str] = token_id

    # Clean decomposition map of inflections that are also base words
    for base_word, inflections in decomposition_map.items():
        valid_inflections = {}
        for inflected_form, transforms in inflections.items():
            if inflected_form.lower() != base_word and inflected_form.lower() in decomposition_map:
                continue  # Skip inflections that are also base words
            else:
                valid_inflections[inflected_form] = transforms
        decomposition_map[base_word] = valid_inflections

    print(f"Built decomposition map with {len(decomposition_map)} base tokens")
    print(f"Found {len(ambiguous_inflections)} ambiguous inflections")
    print(
        f"Found {len(inflection_baseform_conflicts)} conflicts between base forms and inflections"
    )

    # Report duplicate tag inflections
    duplicate_tag_count = 0
    for base, tag_inflections in duplicate_tags_inflections.items():
        for tag, inflections in tag_inflections.items():
            if len(inflections) > 1:
                duplicate_tag_count += 1

    print(f"Found {duplicate_tag_count} cases of duplicate tag inflections")
    print(f"Filtered {len(filtered_duplicate_inflections)} inflections due to duplicate tags")
    print(f"Filtered {len(filtered_short_words)} short words")
    print(f"Filtered {len(filtered_invalid_words)} invalid words")

    # Convert type_to_words to regular dict with lists
    type_to_words = {k: list(v) for k, v in type_to_words.items()}

    filtered = {
        "duplicates": filtered_duplicate_inflections,
        "short_words": list(filtered_short_words),
        "invalid_words": list(filtered_invalid_words),
    }

    return (
        base_tokens,
        decomposition_map,
        type_to_words,
        ambiguous_inflections,
        inflection_baseform_conflicts,
        filtered,
    )


# Example usage for different languages:
def get_language_config(language):
    """
    Examples of how to use the multilingual vocabulary decomposition for different languages.
    """

    # Hebrew example with common prefixes
    hebrew_prefixes = [
        "ב",
        "ל",
        "כ",
        "ה",
        "ו",
        "מ",
        "ש",
        "כש",
        "וב",
        "ול",
        "וכ",
        "וה",
        "ומ",
        "וש",
        "וכש",
        "וכשה",
    ]
    hebrew_prefix_names = [
        "BE",
        "LE",
        "KE",
        "HA",
        "VE",
        "ME",
        "SHE",
        "KSHE",
        "VEBE",
        "VELE",
        "VEKE",
        "VEHA",
        "VEME",
        "VESHE",
        "VEKSHE",
        "VEKSHEHA",
    ]

    # Arabic example with common prefixes and suffixes
    arabic_prefixes = ["ال", "و", "ب", "ل", "ك"]
    arabic_suffixes = ["ها", "هم", "هن", "ني", "ك", "ة"]

    # Spanish example (minimal affixes for demonstration)
    spanish_suffixes = ["mente"]  # adverb suffix

    configs = {
        "hebrew": {
            "language_code": "heb",
            "prefix_transforms": hebrew_prefixes,
            "prefix_transform_names": hebrew_prefix_names,
            "decompose_capitalization": False,  # Hebrew doesn't have capitalization
            "skip_if_no_space_prefix": False,
        },
        "arabic": {
            "language_code": "ara",
            "prefix_transforms": arabic_prefixes,
            "suffix_transforms": arabic_suffixes,
            "decompose_capitalization": False,  # Arabic doesn't have capitalization
        },
        "spanish": {
            "language_code": "spa",
            "suffix_transforms": spanish_suffixes,
            "decompose_capitalization": True,
        },
        "french": {
            "language_code": "fra",
            "decompose_capitalization": True,
        },
        "farsi": {
            "language_code": "fas",
            "decompose_capitalization": False,
        },
        "german": {
            "language_code": "deu",
            "decompose_capitalization": True,
        },
        "ukrainian": {
            "language_code": "ukr",
            "decompose_capitalization": True,
        },
        "indonesian": {
            "language_code": "ind",
            "decompose_capitalization": True,
        },
        "dutch": {
            "language_code": "nld",
            "decompose_capitalization": True,
        },
        "portuguese": {
            "language_code": "por",
            "decompose_capitalization": True,
        },
        "russian": {
            "language_code": "rus",
            "decompose_capitalization": True,
            "skip_if_no_space_prefix": False,
        },
        "turkish": {
            "language_code": "tur",
            "decompose_capitalization": True,
            "skip_if_no_space_prefix": False,
        },
    }

    result = configs[language]
    for k in [
        "prefix_transforms",
        "prefix_transform_names",
        "suffix_transforms",
        "suffix_transform_names",
    ]:
        if k not in result:
            result[k] = None
    for k in ["decompose_capitalization"]:
        if k not in result:
            result[k] = False
    for k in ["skip_if_no_space_prefix"]:
        if k not in result:
            result[k] = True

    return result
