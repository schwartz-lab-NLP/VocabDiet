import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
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

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, pipeline

# for clustering category groups
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

pass
from sklearn.metrics import silhouette_score
import nltk
import pyinflect
from nltk.corpus import wordnet as wn

_nlp = None
_dictionary = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "This decomposition utility requires the spaCy model en_core_web_sm. "
                "Install it with `python -m spacy download en_core_web_sm`."
            ) from exc
    return _nlp


def _get_dictionary():
    global _dictionary
    if _dictionary is None:
        try:
            import enchant

            _dictionary = enchant.Dict("en_US")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "English decomposition requires the system en_US Enchant dictionary."
            ) from exc
    return _dictionary


ADD_SPACE_PREFIX_TRANSFORM = "add_space_prefix"
REMOVE_SPACE_PREFIX_TRANSFORM = "remove_space_prefix"
NO_PREFIX_TRANSFORM = "0_space_prefix"
CAPITALIZED_TRANSFORM = "capitalized"


POS_TO_INFLECTIONS = {
    "NOUN": [("plural", "NNS")],
    "ADJ": [("comparative", "JJR"), ("superlative", "JJS")],
    "ADV": [("comparative", "RBR"), ("superlative", "RBS")],
    "VERB": [
        ("past", "VBD"),
        ("past_participle", "VBN"),
        ("gerund", "VBG"),
        ("present_singular", "VBZ"),
    ],
}
POS_TO_OTHER_POS = {
    "NOUN": ["VERB"],
}
INFLECTION_POS_LIST = [
    # ("NOUN", "plural", "NNS"),
    ("NOUN", "present_singular", "NNS"),
    ("VERB", "past", "VBD"),
    ("VERB", "past_participle", "VBN"),
    ("VERB", "gerund", "VBG"),
    ("VERB", "present_singular", "VBZ"),
    ("ADJ", "comparative", "JJR"),
    ("ADJ", "superlative", "JJS"),
    ("ADV", "comparative", "RBR"),
    ("ADV", "superlative", "RBS"),
]
PLURAL_OR_PRESENT_SINGULAR = "present_singular"
NO_POS_INFLECTION = "none"

IGNORE_MULTI_TYPES_IN_INIT = [
    ADD_SPACE_PREFIX_TRANSFORM,
    REMOVE_SPACE_PREFIX_TRANSFORM,
    CAPITALIZED_TRANSFORM,
]
NO_EMBEDDING_TYPES = [NO_PREFIX_TRANSFORM, NO_POS_INFLECTION]


def is_english_word_without_punctuation(s: str) -> bool:
    # Check if the string contains only English letters (a-z, A-Z)
    return bool(re.fullmatch(r"[a-zA-Z]+", s))


def get_pos_from_wordnet(word):
    pos_map = {wn.NOUN: "NOUN", wn.VERB: "VERB", wn.ADJ: "ADJ", wn.ADV: "ADV"}
    possible_pos = set()
    for synset in wn.synsets(word):
        pos = synset.pos()
        if pos in pos_map:
            possible_pos.add(pos_map[pos])
    return possible_pos


def check_base_form_and_pos(word):
    doc = _get_nlp()(word)
    lemma = doc[0].lemma_
    is_base_form = lemma == word
    pos_set = get_pos_from_wordnet(word) | set(doc[0].pos_)
    return is_base_form, pos_set


def get_basic_variations(
    word,
    transformations_list=None,
    space_prefix="Ġ",
    add_space_prefix_transforms=True,
    add_capitalized_transforms=True,
):
    result = dict()

    has_space_prefix = word.startswith(space_prefix) if space_prefix else False

    if transformations_list is None:
        transformations_list = []

    if len(transformations_list) > 0:
        result[word] = transformations_list + [NO_PREFIX_TRANSFORM]

    # space prefix
    if space_prefix:
        if add_space_prefix_transforms:
            if has_space_prefix:
                result[word[len(space_prefix) :]] = [
                    REMOVE_SPACE_PREFIX_TRANSFORM
                ] + transformations_list
            else:
                result[space_prefix + word] = [ADD_SPACE_PREFIX_TRANSFORM] + transformations_list

    # capitalization
    if add_capitalized_transforms:
        if not has_space_prefix and word[0].islower():
            result[word[0].upper() + word[1:]] = [
                REMOVE_SPACE_PREFIX_TRANSFORM,
                CAPITALIZED_TRANSFORM,
            ] + transformations_list
            if space_prefix and add_space_prefix_transforms:
                result[space_prefix + word[0].upper() + word[1:]] = [
                    NO_PREFIX_TRANSFORM,
                    CAPITALIZED_TRANSFORM,
                ] + transformations_list
        elif has_space_prefix and word[len(space_prefix)].islower():
            if add_space_prefix_transforms:
                result[word[len(space_prefix)].upper() + word[len(space_prefix) + 1 :]] = [
                    REMOVE_SPACE_PREFIX_TRANSFORM,
                    CAPITALIZED_TRANSFORM,
                ] + transformations_list
                if space_prefix:
                    result[
                        space_prefix
                        + word[len(space_prefix)].upper()
                        + word[len(space_prefix) + 1 :]
                    ] = [NO_PREFIX_TRANSFORM, CAPITALIZED_TRANSFORM] + transformations_list
            else:
                result[
                    space_prefix + word[len(space_prefix)].upper() + word[len(space_prefix) + 1 :]
                ] = [NO_PREFIX_TRANSFORM, CAPITALIZED_TRANSFORM] + transformations_list

    return result


def _get_inflection(base_form, pos_type):
    inflection = base_form._.inflect(pos_type)
    if inflection is None:
        return None
    if _get_dictionary().check(inflection):
        return inflection
    return None


def decapitalize(s):
    if not s or any(c.isupper() for c in s[1:]):
        return s
    return s[:1].lower() + s[1:]


def get_vocabulary_decomposition(
    tokenizer,
    space_prefix="Ġ",
    add_space_prefix_transforms=True,
    add_capitalized_transforms=True,
):
    base_tokens = dict()
    decomposition_map = dict()
    processed_tokens = set()
    vocab_size = len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))
    for token_id in tqdm(
        range(vocab_size),
        total=vocab_size,
        miniters=100,
        desc="Preparing vocabulary decomposition",
        unit="word",
    ):
        token = tokenizer._tokenizer.id_to_token(token_id)
        token_repr = token.strip(space_prefix)
        token_has_space_prefix = token.startswith(space_prefix)
        # check if token represents a base word
        if not (
            is_english_word_without_punctuation(token_repr) and _get_dictionary().check(token_repr)
        ):
            continue
        is_base_form, pos_set = check_base_form_and_pos(token_repr)
        if not is_base_form:
            continue

        # check if token is already accounted for in the decomposition map
        if add_space_prefix_transforms:
            if token in decomposition_map or token_repr in decomposition_map:
                continue
            if token_repr.islower() and not token_has_space_prefix:
                continue
            if (
                not token_has_space_prefix
            ):  # make the base form be the one with space prefix - also for capitalized words
                continue
            if add_capitalized_transforms:
                if (
                    decapitalize(token_repr) in decomposition_map
                    or f"{space_prefix}{decapitalize(token_repr)}" in decomposition_map
                ):
                    continue
                if (
                    token_repr.capitalize() in decomposition_map
                    or f"{space_prefix}{token_repr.capitalize()}" in decomposition_map
                ):
                    continue
        else:
            if token in decomposition_map:
                continue
            if add_capitalized_transforms:
                if token_has_space_prefix:
                    if f"{space_prefix}{decapitalize(token_repr)}" in decomposition_map:
                        continue
                    if f"{space_prefix}{token_repr.capitalize()}" in decomposition_map:
                        continue
                else:
                    if decapitalize(token_repr) in decomposition_map:
                        continue
                    if token_repr.capitalize() in decomposition_map:
                        continue

        # don't add pronouns, proper nouns, etc.
        accepted_pos = {"VERB", "NOUN", "ADV", "ADJ"}
        pos_set = pos_set & accepted_pos
        if len(pos_set) == 0:
            continue

        # spacy sometimes tags single letters as nouns.
        # 2 and 3 letters also appear to be noisy
        if len(token_repr) == 1:
            continue

        # map to properties
        base_tokens[token] = token_id
        processed_tokens.add(token)
        decomposition_map[token] = dict()
        decomposition_map[token].update(
            get_basic_variations(
                token,
                transformations_list=[NO_POS_INFLECTION],
                space_prefix=space_prefix,
                add_space_prefix_transforms=add_space_prefix_transforms,
                add_capitalized_transforms=add_capitalized_transforms,
            )
        )

        base_form = _get_nlp()(token_repr)[0]
        inflection_pos_list = INFLECTION_POS_LIST

        # if 1. the word could be both a noun and a verb,
        # and 2. the plural form and the present singular form match,
        # then add it to the decomposition map as special type
        if "NOUN" in pos_set and "VERB" in pos_set:
            plural_form = _get_inflection(base_form, "NNS")
            present_singular_form = _get_inflection(base_form, "VBZ")
            if plural_form and (plural_form == present_singular_form):
                inflection_form = plural_form
                inflection_name = PLURAL_OR_PRESENT_SINGULAR
                if token_has_space_prefix:
                    inflection_form = f"{space_prefix}{inflection_form}"
                if (
                    inflection_form not in decomposition_map[token]
                    and inflection_form not in processed_tokens
                ):
                    decomposition_map[token].update(
                        get_basic_variations(
                            inflection_form,
                            transformations_list=[inflection_name],
                            space_prefix=space_prefix,
                            add_space_prefix_transforms=add_space_prefix_transforms,
                            add_capitalized_transforms=add_capitalized_transforms,
                        )
                    )
                    processed_tokens.add(inflection_form)

        # handle all other cases
        for base_pos, inflection_name, inflection_pos in inflection_pos_list:
            if base_pos not in pos_set:
                continue
            inflection_form = _get_inflection(base_form, inflection_pos)
            if inflection_form:
                if token_has_space_prefix:
                    inflection_form = f"{space_prefix}{inflection_form}"
                if (
                    inflection_form not in decomposition_map[token]
                    and inflection_form not in processed_tokens
                ):
                    decomposition_map[token].update(
                        get_basic_variations(
                            inflection_form,
                            transformations_list=[inflection_name],
                            space_prefix=space_prefix,
                            add_space_prefix_transforms=add_space_prefix_transforms,
                            add_capitalized_transforms=add_capitalized_transforms,
                        )
                    )
                    processed_tokens.add(inflection_form)

    return base_tokens, decomposition_map


class PolysemyDetector:
    """
    # Replace 'your-model-name' with the actual identifier of your Llama or similar chat model.
    detector = PolysemyDetector(model_name="meta-llama/Llama-3.1-8B-Instruct")
    test_word = "set"  # Example word known for its multiple meanings.
    result = detector.is_polysemous(test_word)

    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initializes the detector by loading the model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        # Using a text-generation pipeline; adjust parameters as needed.
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def is_polysemous(self, word: str) -> bool:
        """
        Determines if the given word has multiple, context-dependent meanings.
        Returns True if yes, otherwise False.
        """
        # Ten prompt templates with examples:
        prompt_templates = [
            f"Some words have multiple meanings. For example, 'friends' can be a TV show or refer to companions, and 'lost' can mean missing or be the title of a TV series. Is the word '{word}' polysemous? Answer only with 'yes' or 'no'.",
            f"Consider words like 'set' which can mean a collection or a stage set, and 'bug' which can be an insect or a software error. Does '{word}' have more than one meaning? Answer yes or no.",
            f"Examples include 'overwatch' (a video game or the act of observing) and 'halo' (a ring of light or a game title). Does '{word}' have distinct meanings depending on context? Answer with yes or no.",
            f"Words such as 'house' (a building or a TV show title) and 'office' (a workplace or a series name) illustrate multiple meanings. Is '{word}' used in different contexts? Answer yes or no.",
            f"In different fields, words can have varying meanings. For instance, 'crash' can describe a collision or a computer failure, and 'patch' might mean a fabric piece or a software update. Is '{word}' polysemous? Answer only yes or no.",
            f"Some terms like 'theory' may have a strict scientific meaning versus everyday usage, while 'bond' can be a financial instrument or imply a personal connection. Does '{word}' exhibit polysemy? Answer with yes or no.",
            f"Consider 'lost' which might mean missing or refer to a TV series, and 'singularity' which can mean uniqueness or a point in a black hole. Is '{word}' a word with multiple meanings? Answer yes or no.",
            f"Words can carry distinct meanings in different domains. For example, 'charge' can mean an accusation or electrical energy, and 'liability' might be a legal duty or financial obligation. Does '{word}' have multiple meanings? Answer yes or no.",
            f"Some words, such as 'portal' (a doorway or a video game title) and 'doom' (fate or a game series), are used in various contexts. Is '{word}' used in multiple ways? Answer with yes or no.",
            f"For example, 'frozen' can describe a state of being or refer to the animated film, and 'overwatch' can mean vigilant observation or a video game. Does '{word}' have multiple context-dependent meanings? Answer yes or no.",
        ]

        answers = []
        for prompt in prompt_templates:
            # Generate a response with a modest max length to avoid extra text.
            response = self.generator(
                prompt, max_length=50, do_sample=False, num_return_sequences=1
            )[0]["generated_text"]
            # Normalize the response text
            answer = response.strip().lower()
            # Check for a clear 'yes' or 'no' answer
            if "yes" in answer:
                answers.append(True)
            elif "no" in answer:
                answers.append(False)

        # Use a simple majority vote among the responses.
        if answers:
            return answers.count(True) > answers.count(False)
        return False

    def batch_is_polysemous(self, words: List[str], batch_size: int = 10) -> List[bool]:
        """
        Processes a list of words in batches to determine if each is polysemous.

        Args:
            words: List of words to check
            batch_size: Number of words to process in each batch

        Returns:
            List of boolean values indicating if each word is polysemous
        """
        if not words:
            return []

        results = [False] * len(words)
        num_batches = math.ceil(len(words) / batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(words))
            batch_words = words[start_idx:end_idx]

            # Generate prompts for all words in the batch
            all_prompts = []
            prompt_templates = [
                f"Some words have multiple meanings. For example, 'friends' can be a TV show or refer to companions, and 'lost' can mean missing or be the title of a TV series. Is the word '{{word}}' polysemous? Answer only with 'yes' or 'no'.",
                f"Consider words like 'set' which can mean a collection or a stage set, and 'bug' which can be an insect or a software error. Does '{{word}}' have more than one meaning? Answer yes or no.",
                f"Examples include 'overwatch' (a video game or the act of observing) and 'halo' (a ring of light or a game title). Does '{{word}}' have distinct meanings depending on context? Answer with yes or no.",
                f"Words such as 'house' (a building or a TV show title) and 'office' (a workplace or a series name) illustrate multiple meanings. Is '{{word}}' used in different contexts? Answer yes or no.",
                f"In different fields, words can have varying meanings. For instance, 'crash' can describe a collision or a computer failure, and 'patch' might mean a fabric piece or a software update. Is '{{word}}' polysemous? Answer only yes or no.",
                f"Some terms like 'theory' may have a strict scientific meaning versus everyday usage, while 'bond' can be a financial instrument or imply a personal connection. Does '{{word}}' exhibit polysemy? Answer with yes or no.",
                f"Consider 'lost' which might mean missing or refer to a TV series, and 'singularity' which can mean uniqueness or a point in a black hole. Is '{{word}}' a word with multiple meanings? Answer yes or no.",
                f"Words can carry distinct meanings in different domains. For example, 'charge' can mean an accusation or electrical energy, and 'liability' might be a legal duty or financial obligation. Does '{{word}}' have multiple meanings? Answer yes or no.",
                f"Some words, such as 'portal' (a doorway or a video game title) and 'doom' (fate or a game series), are used in various contexts. Is '{{word}}' used in multiple ways? Answer yes or no.",
                f"For example, 'frozen' can describe a state of being or refer to the animated film, and 'overwatch' can mean vigilant observation or a video game. Does '{{word}}' have multiple context-dependent meanings? Answer yes or no.",
            ]

            for word in batch_words:
                word_prompts = [template.format(word=word) for template in prompt_templates]
                all_prompts.extend(word_prompts)

            # Process all prompts in the batch
            responses = self.generator(
                all_prompts, max_length=50, do_sample=False, num_return_sequences=1
            )

            # Process responses for each word
            prompts_per_word = len(prompt_templates)
            for i, word in enumerate(batch_words):
                word_responses = responses[i * prompts_per_word : (i + 1) * prompts_per_word]
                answers = []
                for response in word_responses:
                    answer = response["generated_text"].strip().lower()
                    if "yes" in answer:
                        answers.append(True)
                    elif "no" in answer:
                        answers.append(False)

                if answers:
                    results[start_idx + i] = answers.count(True) > answers.count(False)

        return results


def get_initial_type_embeddings_from_strings(
    base_tokens,
    decomposition_map,
    tokenizer,
    input_embeddings,
    output_embeddings,
    type_names_to_int,
    space_prefix="Ġ",
):
    """
    Compute initial type embeddings using string-based decomposition map.

    Args:
        base_tokens: Dict mapping base words to their token IDs
        decomposition_map: Dict mapping base words to their inflections and transformations
        tokenizer: Tokenizer object with vocabulary
        input_embeddings: Input embedding matrix
        output_embeddings: Output embedding matrix
        type_names_to_int: Dict mapping type names to integer IDs
        space_prefix: Prefix used for tokens that start with space

    Returns:
        tuple: (input_type_embeddings, output_type_embeddings) - Type embeddings for input/output spaces
    """
    input_type_embeddings = torch.zeros(
        (len(type_names_to_int), input_embeddings.shape[-1]), dtype=input_embeddings.dtype
    ).to(input_embeddings.device)
    output_type_embeddings = torch.zeros(
        (len(type_names_to_int), output_embeddings.shape[-1]), dtype=output_embeddings.dtype
    ).to(output_embeddings.device)
    type_counts = torch.zeros((len(type_names_to_int)), dtype=torch.long).to(
        input_embeddings.device
    )
    types_to_tokens = [[] for _ in type_names_to_int]
    vocab_size = len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))
    allowed_multi_types = set(
        [type_names_to_int[type_name] for type_name in IGNORE_MULTI_TYPES_IN_INIT]
    )
    ignored_types = set([type_names_to_int[type_name] for type_name in NO_EMBEDDING_TYPES])

    for base_word, inflections in decomposition_map.items():
        base_token_id = base_tokens[base_word]

        for inflection_word, transformations_list in inflections.items():
            # Skip if inflection is same as base word
            if inflection_word == base_word:
                continue

            # Get token ID for inflection word
            inflection_word_encoding = tokenizer.encode(
                inflection_word.replace(space_prefix, " ") if space_prefix else inflection_word,
                add_special_tokens=False,
            )

            # Skip multi-token words
            if len(inflection_word_encoding) > 1:
                continue

            inflection_token_id = inflection_word_encoding[0]

            # Skip tokens outside core vocabulary
            if inflection_token_id >= vocab_size:
                continue

            # Convert transformation names to type IDs
            type_ids = [
                type_names_to_int[t]
                for t in transformations_list
                if type_names_to_int[t] not in ignored_types
            ]

            # Process single vs multiple transformations
            if len(type_ids) > 1:
                for type_id in type_ids:
                    if type_id not in allowed_multi_types:
                        continue
                    input_type_embeddings[type_id] += (
                        input_embeddings[inflection_token_id] - input_embeddings[base_token_id]
                    )
                    output_type_embeddings[type_id] += (
                        output_embeddings[inflection_token_id] - output_embeddings[base_token_id]
                    )
                    type_counts[type_id] += 1
                    types_to_tokens[type_id].append(
                        tokenizer._tokenizer.id_to_token(inflection_token_id)
                    )
            elif len(type_ids) == 1:
                type_id = type_ids[0]
                input_type_embeddings[type_id] += (
                    input_embeddings[inflection_token_id] - input_embeddings[base_token_id]
                )
                output_type_embeddings[type_id] += (
                    output_embeddings[inflection_token_id] - output_embeddings[base_token_id]
                )
                type_counts[type_id] += 1
                types_to_tokens[type_id].append(
                    tokenizer._tokenizer.id_to_token(inflection_token_id)
                )

    # Average the embeddings
    type_counts = type_counts.unsqueeze(-1)
    input_type_embeddings = torch.where(
        type_counts != 0, input_type_embeddings / type_counts, input_type_embeddings
    )
    output_type_embeddings = torch.where(
        type_counts != 0, output_type_embeddings / type_counts, output_type_embeddings
    )
    return input_type_embeddings, output_type_embeddings


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
        hidden_states_flat = hidden_states_all_layers.reshape(
            -1, hidden_states_all_layers.size(-1)
        ).to(self.model.dtype)

        # Prepare inputs for all states at once
        inputs_embeds = self.model.get_input_embeddings()(
            self.prompt_input_ids.to(self.model.device)
        ).unsqueeze(0)
        batch_inputs = inputs_embeds.repeat(len(hidden_states_flat), 1, 1)
        batch_inputs[:, self.prompt_target_idx] = hidden_states_flat.unsqueeze(1)

        # Prepare attention mask
        attention_mask = torch.ones(
            (len(hidden_states_flat), len(self.prompt_input_ids)),
            device=self.model.device,
            dtype=self.model.dtype,
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
    last_layer: int = 9,
    batch_size: int = 32,
    max_words: int = None,
    space_prefix: str = "Ġ",
    single_type_only: bool = False,
    skip_multi_token_words: bool = False,
    compare_to_original: bool = True,
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

    # Initialize patchscopes processor
    patchscopes = PatchscopesProcessor(model, tokenizer)

    # Prepare word pairs
    word_pairs = []
    base_words_set = set()

    for base_word, inflections in tqdm(decomposition_map.items(), desc="Preparing word pairs"):
        base_words_set.add(base_word)
        for infl_word, transforms in inflections.items():
            filtered_transforms = [t for t in transforms if t != "NO_EMBEDDING"]
            if single_type_only and len(filtered_transforms) != 1:
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

    # limit words to run patchscopes on
    if max_words is not None:
        word_pairs = word_pairs[:max_words]
        base_words = base_words[:max_words]

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

                last_token_indices = original_inputs.attention_mask.sum(dim=1) - 1
                original_hidden_states = torch.stack(
                    [
                        states[torch.arange(current_batch_size), last_token_indices]
                        for states in original_outputs.hidden_states[:num_layers]
                    ]
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
        base_inputs = torch.tensor([base_tokens[base_word] for _, base_word, _ in batch]).to(device)

        # Add BOS token if needed
        if tokenizer.bos_token_id is not None:
            base_inputs = torch.cat(
                [
                    torch.full((current_batch_size, 1), tokenizer.bos_token_id, device=device),
                    base_inputs.unsqueeze(1),
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

                # Get last token indices
                last_token_indices = original_inputs.attention_mask.sum(dim=1) - 1

                # Stack hidden states from all layers for original representation
                original_hidden_states = torch.stack(
                    [
                        states[torch.arange(current_batch_size), last_token_indices]
                        for states in original_outputs.hidden_states[:num_layers]
                    ]
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


def analyze_patchscopes_results(
    results: Dict[str, List[List[str]]],
    results_by_type: Dict[str, Dict[str, List[List[str]]]],
    num_layers: int,
    space_prefix=None,
):
    """
    Analyzes word identification patterns in patchscopes results.

    Args:
        results: Dict mapping words to their patchscopes outputs
        results_by_type: Dict mapping types to words and their outputs
        num_layers: Number of layers analyzed

    Returns:
        Dict containing analysis results structured for DataFrame conversion with fields:
        - word: The word being analyzed
        - type: The transformation type (or 'BASE' for base words)
        - identified_additive: Whether word was identified in additive representation
        - first_layer_additive: First layer where word was identified (additive)
        - layers_identified_additive: List of layers where word was identified (additive)
        - identified_original: Whether word was identified in original representation
        - first_layer_original: First layer where word was identified (original)
        - layers_identified_original: List of layers where word was identified (original)
    """
    analysis_results = list()
    analysis_results_by_type = defaultdict(list)

    def analyze_outputs(outputs: List[List[str]], word: str) -> Tuple[bool, int, List[int]]:
        """Helper function to analyze a set of outputs across layers."""
        if space_prefix:
            word = word.replace(space_prefix, " ")
        identified = False
        first_layer = num_layers  # Default to num_layers if never identified
        layers_identified = []

        for layer_idx, layer_output in enumerate(outputs):
            if not layer_output:  # Skip empty outputs
                continue
            # Strip spaces and check if word is at start of output
            if layer_output.strip().startswith(word.strip()):
                identified = True
                layers_identified.append(layer_idx)
                if layer_idx < first_layer:
                    first_layer = layer_idx

        if not identified:
            first_layer = -1

        return identified, first_layer, layers_identified

    # Analyze by type
    for type_name, type_results in results_by_type.items():
        for word, patchscopes_outputs in type_results.items():
            if len(patchscopes_outputs) == 1:
                original_outputs = patchscopes_outputs[0]
                additive_outputs = None
            else:
                additive_outputs, original_outputs = patchscopes_outputs

            if additive_outputs:
                identified_additive, first_layer_additive, layers_identified_additive = (
                    analyze_outputs(additive_outputs, word)
                )
            else:
                identified_additive, first_layer_additive, layers_identified_additive = (
                    None,
                    None,
                    None,
                )

            if original_outputs:  # Only analyze original if available
                identified_original, first_layer_original, layers_identified_original = (
                    analyze_outputs(original_outputs, word)
                )
            else:
                identified_original, first_layer_original, layers_identified_original = (
                    None,
                    None,
                    None,
                )

            analysis_results.append(
                {
                    "word": word,
                    "type": type_name,
                    "identified_additive": identified_additive,
                    "first_layer_additive": first_layer_additive,
                    "layers_identified_additive": layers_identified_additive,
                    "identified_original": identified_original,
                    "first_layer_original": first_layer_original,
                    "layers_identified_original": layers_identified_original,
                }
            )
            analysis_results_by_type[type_name].append(
                {
                    "word": word,
                    "type": type_name,
                    "identified_additive": identified_additive,
                    "first_layer_additive": first_layer_additive,
                    "layers_identified_additive": layers_identified_additive,
                    "identified_original": identified_original,
                    "first_layer_original": first_layer_original,
                    "layers_identified_original": layers_identified_original,
                }
            )

    return analysis_results, dict(analysis_results_by_type)


def remove_outliers_from_decomposition_map(decomposition_map, outliers):
    clean_decomposition_map = deepcopy(decomposition_map)
    outliers = set(outliers)
    for base_word, inflection_decompositions in decomposition_map.items():
        for inflection_word in inflection_decompositions:
            if inflection_word in outliers:
                clean_decomposition_map[base_word].pop(inflection_word)
        if not clean_decomposition_map[base_word]:
            clean_decomposition_map.pop(base_word)
            break
    return clean_decomposition_map


def rms_norm(x, eps=1e-8):
    """
    Applies RMS (Root Mean Square) normalization to a 2D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features)
        eps (float): Small value added for numerical stability

    Returns:
        torch.Tensor: Normalized tensor of the same shape as input
    """
    # Ensure input is 2D
    assert x.dim() == 2, "Input tensor must be 2D (batch_size, features)"

    # Calculate RMS: sqrt(mean(x^2))
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

    # Normalize
    return rms


def add_words_to_core_vocab(tokenizer, new_tokens_to_ids, merges_list):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save tokenizer to the temporary directory
        tokenizer.save_pretrained(temp_dir)

        # Load the tokenizer.json file
        tokenizer_json_path = f"{temp_dir}/tokenizer.json"
        with open(tokenizer_json_path, "r") as f:
            tokenizer_json = json.load(f)

        # Make edits to the tokenizer.json
        for word, token_id in new_tokens_to_ids.items():
            tokenizer_json["model"]["vocab"][word] = token_id

        tokenizer_json["model"]["merges"] = tokenizer_json["model"]["merges"] + merges_list
        # Save the edited tokenizer.json as a temp file and reload the modified tokenizer
        with open(tokenizer_json_path, "w") as f:
            json.dump(tokenizer_json, f, indent=2)
        new_tokenizer = AutoTokenizer.from_pretrained(
            temp_dir, added_tokens_decoder=tokenizer.added_tokens_decoder.copy()
        )

        return new_tokenizer


def get_type_names_to_int_map():
    transformations_names = list(
        dict.fromkeys(
            [
                transformation_name
                for base_pos, transformation_name, transformation_pos in INFLECTION_POS_LIST
            ]
        )
    )
    if PLURAL_OR_PRESENT_SINGULAR not in transformations_names:
        transformations_names += [PLURAL_OR_PRESENT_SINGULAR]
    transformations_names += [NO_POS_INFLECTION]
    transformations_names = [
        CAPITALIZED_TRANSFORM,
        ADD_SPACE_PREFIX_TRANSFORM,
        REMOVE_SPACE_PREFIX_TRANSFORM,
        NO_PREFIX_TRANSFORM,
    ] + transformations_names
    transformations_to_int = dict(zip(transformations_names, range(len(transformations_names))))
    return transformations_to_int


def extend_tokenizer_by_decomposition_map(
    tokenizer,
    base_tokens,
    decomposition_map,
    space_prefix="Ġ",
    skip_multi_token_words=False,
    skip_three_token_words=False,
    transformations_to_int=None,
):
    vocab_size = len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))

    # TODO fix: is there a problem with adjectives like "Western"?
    # TODO fix: Adverbs are not treated as inflections of adjectives, but are their own base form
    if transformations_to_int is None:
        transformations_to_int = get_type_names_to_int_map()
    # build the new tokenizer and the final decomposition map for tokenization

    final_decomposition_map = dict()
    special_tokens_map = dict()
    base_token_to_valid_transformations = dict()
    decomposed_token_ids = list()
    decomposed_to_original_id = dict()
    decomposed_to_original_seq = dict()
    existing_inflection_token_ids = set()
    new_tokens = dict()
    scaffold_vocab = dict()
    scaffold_merges = list()

    # Add placeholders for existing special tokens:
    # This prevents the new tokens from taking their ID in the embedding table,
    # and this is important because we don't really add new input embeddings
    for special_token_id, special_token_str in tokenizer.added_tokens_decoder.items():
        special_tokens_map[special_token_id] = (special_token_id, [])
        decomposed_to_original_id[special_token_id] = special_token_id
        new_tokens[str(special_token_str)] = special_token_id

    newest_token_id = max(vocab_size, max(tokenizer.added_tokens_decoder.keys()) + 1)
    for base_word, inflections in decomposition_map.items():
        base_token_id = base_tokens[base_word]
        base_token_to_valid_transformations[base_token_id] = [
            transformations_to_int[transformation]
            for transformation in set(chain(*decomposition_map[base_word].values()))
        ]
        for inflection_word in inflections:
            if inflection_word == base_word:
                transformations_list = decomposition_map[base_word][inflection_word]
                final_decomposition_map[base_token_id] = (
                    base_token_id,
                    [
                        transformations_to_int[transformation]
                        for transformation in transformations_list
                    ],
                )
                decomposed_to_original_id[base_token_id] = base_token_id
                continue
            # get token id for inflection word
            inflection_word_encoding = (
                tokenizer.encode(
                    inflection_word.replace(space_prefix, " "), add_special_tokens=False
                )
                if space_prefix
                else tokenizer.encode(inflection_word, add_special_tokens=False)
            )

            # add scaffold tokens and merges if necessary
            if len(inflection_word_encoding) == 1:
                decomposed_token_id = inflection_word_encoding[0]
                existing_inflection_token_ids.add(decomposed_token_id)
            else:
                if skip_multi_token_words:
                    continue
                if skip_three_token_words and len(inflection_word_encoding) > 2:
                    continue

                curr_new_token = f"{tokenizer._tokenizer.id_to_token(inflection_word_encoding[0])}"

                for i in range(len(inflection_word_encoding) - 1):
                    curr_merge = [
                        f"{curr_new_token}",
                        f"{tokenizer._tokenizer.id_to_token(inflection_word_encoding[i + 1])}",
                    ]

                    curr_new_token = f"{curr_new_token}{tokenizer._tokenizer.id_to_token(inflection_word_encoding[i + 1])}"
                    if curr_new_token in new_tokens:
                        continue
                    scaffold_merges.append(curr_merge)
                    new_tokens[curr_new_token] = newest_token_id
                    if i < len(inflection_word_encoding) - 2:
                        scaffold_vocab[curr_new_token] = newest_token_id
                    newest_token_id += 1

                decomposed_token_id = newest_token_id - 1

            decomposed_token_ids.append(decomposed_token_id)
            decomposed_to_original_id[decomposed_token_id] = inflection_word_encoding[0]
            decomposed_to_original_seq[decomposed_token_id] = inflection_word_encoding
            transformations_list = decomposition_map[base_word][inflection_word]
            final_decomposition_map[decomposed_token_id] = (
                base_token_id,
                [transformations_to_int[transformation] for transformation in transformations_list],
            )

    new_tokenizer = add_words_to_core_vocab(tokenizer, new_tokens, scaffold_merges)
    existing_inflection_token_ids = list(existing_inflection_token_ids)
    negative_types = set([transformations_to_int[type_name] for type_name in NO_EMBEDDING_TYPES])
    return (
        new_tokenizer,
        final_decomposition_map,
        special_tokens_map,
        scaffold_vocab,
        base_token_to_valid_transformations,
        transformations_to_int,
        existing_inflection_token_ids,
        decomposed_to_original_id,
        decomposed_to_original_seq,
        negative_types,
    )
