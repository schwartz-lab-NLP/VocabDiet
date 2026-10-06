import os
import re
import nltk
import spacy
from nltk.corpus import wordnet as wn
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
from pathlib import Path
import unicodedata
from itertools import chain
from copy import deepcopy
from sklearn.cluster import KMeans

pass


# Constants for transformation types
NO_PREFIX_TRANSFORM = "no_space_prefix"
REMOVE_SPACE_PREFIX_TRANSFORM = "remove_space_prefix"
NO_CAPITALIZATION_TRANSFORM = "no_capitalization"
ADD_CAPITALIZATION_TRANSFORM = "add_capitalization"
REMOVE_CAPITALIZATION_TRANSFORM = "remove_capitalization"
NO_POS_INFLECTION = "none"
NO_EMBEDDING_TYPES = [NO_PREFIX_TRANSFORM, NO_POS_INFLECTION, NO_CAPITALIZATION_TRANSFORM, "N;SG"]
IGNORE_MULTI_TYPES_IN_INIT = []
FORBID_MULTI_TYPES_IN_INIT = [
    ADD_CAPITALIZATION_TRANSFORM,
]

# Load linguistic resources only when decomposition needs them. This lets CLI help,
# model math tests, and analysis that uses precomputed maps run without local NLP data.
_nlp = None
_dictionary = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise RuntimeError(
                "English vocabulary filtering requires the spaCy model en_core_web_sm. "
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
                "English vocabulary filtering requires the en_US Enchant dictionary. "
                "Install the system Enchant library and English dictionary."
            ) from exc
    return _dictionary


def get_english_unimorph_paths(unimorph_root=None):
    """Return English UniMorph inflection and derivation paths.

    `unimorph_root` should contain `eng/eng` and optionally
    `eng/eng.derivations.tsv`; it defaults to `$UNIMORPH_ROOT` or
    `resources/unimorph` relative to the current working directory.
    """
    root = Path(unimorph_root or os.environ.get("UNIMORPH_ROOT", "resources/unimorph"))
    language_dir = root / "eng"
    inflections = language_dir / "eng"
    derivations = language_dir / "eng.derivations.tsv"
    if not inflections.is_file():
        raise FileNotFoundError(
            f"English UniMorph inflections not found at {inflections}. "
            "Set --unimorph_root or UNIMORPH_ROOT to the UniMorph data directory."
        )
    return str(inflections), str(derivations) if derivations.is_file() else None


class OrderedSet:
    """Maintains insertion order while providing set-like functionality."""

    def __init__(self, iterable=None):
        self._items = []
        self._item_set = set()
        if iterable:
            for item in iterable:
                self.add(item)

    def add(self, item):
        if item not in self._item_set:
            self._items.append(item)
            self._item_set.add(item)

    def __contains__(self, item):
        return item in self._item_set

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return self._items == other._items
        return self._item_set == set(other)

    def __repr__(self):
        return f"OrderedSet({self._items})"

    def intersection(self, other):
        return OrderedSet(x for x in self if x in other)

    def union(self, other):
        result = OrderedSet(self)
        for item in other:
            result.add(item)
        return result

    def to_tuple(self):
        return tuple(self._items)


class UniMorphAnalyzer:
    """Processes and provides access to UniMorph data for inflections and derivations."""

    def __init__(self, inflections_filepath=None, derivations_filepath=None, no_diacritics=False):
        # Initialize data structures
        self.lemma_to_forms = defaultdict(lambda: defaultdict(set))
        self.form_to_lemmas = defaultdict(lambda: defaultdict(set))
        self.tag_combinations = Counter()
        self.data = None
        self.frequent_tag_combinations = set()
        self.tag_to_subtypes = defaultdict(set)

        # Derivation-related structures
        self.derivation_to_words = defaultdict(set)
        self.root_to_derivations = defaultdict(lambda: defaultdict(set))
        self.derived_to_root = defaultdict(set)
        self.derivation_chains = defaultdict(list)
        self.common_derivation_types = set()

        # Other parameters
        self.no_diacritics = no_diacritics

        # Load data if provided
        if inflections_filepath:
            self.load_inflections_data(inflections_filepath)

        if derivations_filepath:
            self.load_derivations_data(derivations_filepath)

    def remove_diacritics(self, text):
        """Remove diacritics using regex patterns."""
        # Remove Arabic diacritics (harakat) - Unicode range for Arabic diacritics
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

        # Remove Hebrew nikud (vowel points) - Unicode range for Hebrew points
        text = re.sub(r"[\u05B0-\u05BD\u05BF\u05C1-\u05C2\u05C4-\u05C5\u05C7]", "", text)

        return text

    def load_inflections_data(self, filepath):
        """Load UniMorph inflection data."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    parts = line.split("\t")
                    lemma, form, tags = parts[0], parts[1], parts[2]
                    if form == "countable" or form == "uncountable":
                        form = lemma
                    if self.no_diacritics:
                        lemma = self.remove_diacritics(lemma)
                        form = self.remove_diacritics(form)
                    tag_list = tags.split(";")
                    tag_tuple = tuple(tag_list)
                    # Add to data structures
                    self.lemma_to_forms[lemma][form].add(tag_tuple)
                    self.form_to_lemmas[form][lemma].add(tag_tuple)
                    self.tag_combinations[tag_tuple] += 1
                    for tag in tag_list:
                        self.tag_to_subtypes[tag].add(tag_tuple)
                except Exception as e:
                    print(f"Error processing line: {line}, Error: {str(e)}")

    def load_derivations_data(self, filepath):
        """Load UniMorph derivation data."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    parts = line.split("\t")
                    if len(parts) >= 4:
                        root, derived, pos, deriv_type = parts[0], parts[1], parts[2], parts[3]
                        if self.no_diacritics:
                            root = self.remove_diacritics(root)
                            derived = self.remove_diacritics(derived)
                        pos_pair = pos.split(":")
                        if len(pos_pair) != 2:
                            continue

                        # Map derivation type to derived words
                        self.derivation_to_words[deriv_type].add(derived)

                        # Map root to derivations with type and POS
                        self.root_to_derivations[root][derived].add((deriv_type, pos))

                        # Map derived word to root with type and POS
                        self.derived_to_root[derived].add((root, deriv_type, pos))

                        # Build derivation chains
                        self._build_derivation_chain(root, derived, deriv_type, pos_pair)
                except Exception as e:
                    print(f"Error processing derivation line: {line}, Error: {str(e)}")

    def _build_derivation_chain(self, root, derived, deriv_type, pos_pair):
        """Build chains of derivations, tracking the sequence."""
        # Check if the root itself is derived from something else
        if root in self.derived_to_root:
            for grand_root, prev_type, prev_pos in self.derived_to_root[root]:
                prev_pos_pair = prev_pos.split(":")
                chain = self.derivation_chains[root]
                if chain:
                    # Extend existing chain
                    new_types = chain[0][1] + [deriv_type]
                    new_pos = chain[0][2] + [pos_pair[1]]
                    self.derivation_chains[derived].append((grand_root, new_types, new_pos))
                else:
                    self.derivation_chains[derived].append(
                        (grand_root, [prev_type, deriv_type], [prev_pos_pair[1], pos_pair[1]])
                    )
        else:
            # No chain, just a single derivation
            self.derivation_chains[derived].append((root, [deriv_type], [pos_pair[1]]))

    def set_frequent_tag_combinations(self, top_n=None, min_count=None):
        """Set the frequent tag combinations based on count or top N."""
        if top_n is not None:
            self.frequent_tag_combinations = {
                tags for tags, _ in self.tag_combinations.most_common(top_n)
            }
        elif min_count is not None:
            self.frequent_tag_combinations = {
                tags for tags, count in self.tag_combinations.items() if count >= min_count
            }
        else:
            self.frequent_tag_combinations = set(self.tag_combinations.keys())

    def set_common_derivation_types(self, min_count=1000):
        """Set common derivation types based on their frequency."""
        derivation_counts = {
            deriv_type: len(words) for deriv_type, words in self.derivation_to_words.items()
        }
        self.common_derivation_types = {
            deriv_type
            for deriv_type, count in derivation_counts.items()
            if count >= min_count and deriv_type != "ing"
        }

        # Print out chosen derivation types
        common_types = sorted(
            [(t, derivation_counts[t]) for t in self.common_derivation_types],
            key=lambda x: x[1],
            reverse=True,
        )
        print(f"Selected {len(common_types)} common derivation types:")
        for deriv_type, count in common_types[:20]:
            print(f"  {deriv_type}: {count} instances")
        if len(common_types) > 20:
            print(f"  ... and {len(common_types) - 20} more")

    def get_inflections(self, lemma):
        """Get all inflected forms of a lemma with their tags."""
        return dict(self.lemma_to_forms.get(lemma, {}))

    def get_root_derivations(self, root):
        """Get all derivations of a root word."""
        if root not in self.root_to_derivations:
            return {}

        # Filter to only common derivation types
        result = {}
        for derived, type_pos_set in self.root_to_derivations[root].items():
            filtered_types = [
                (deriv_type, pos)
                for deriv_type, pos in type_pos_set
                if deriv_type in self.common_derivation_types
            ]
            if filtered_types:
                result[derived] = filtered_types

        return result

    def is_unimorph_word(self, word):
        is_lemma = word in self.lemma_to_forms
        is_form = word in self.form_to_lemmas
        return is_lemma or is_form

    def is_base_form(self, word):
        """Check if a word is a base form in UniMorph."""
        is_lemma = word in self.lemma_to_forms
        is_form = word in self.form_to_lemmas
        return is_lemma or (not (is_lemma or is_form))

    def is_lemma(self, word):
        """Check if a word is a lemma in UniMorph."""
        is_lemma = word in self.lemma_to_forms
        return is_lemma


def get_pos_from_wordnet(word):
    """Get possible parts of speech for a word from WordNet."""
    pos_map = {wn.NOUN: "NOUN", wn.VERB: "VERB", wn.ADJ: "ADJ", wn.ADV: "ADV"}
    possible_pos = set()
    for synset in wn.synsets(word):
        pos = synset.pos()
        if pos in pos_map:
            possible_pos.add(pos_map[pos])
    # Add spacy POS
    doc = _get_nlp()(word)
    possible_pos = possible_pos | set([doc[0].pos_])
    return possible_pos


def _get_inflection_from_pyinflect(base_form, pos_type):
    """Get an inflection using PyInflect."""
    inflection = base_form._.inflect(pos_type)
    if inflection is None:
        return None
    if _get_dictionary().check(inflection):
        synsets = wn.synsets(inflection)
        if any(base_form.text in [lemma.name() for lemma in syn.lemmas()] for syn in synsets):
            return inflection
    return None


def get_pyinflect_variations(word, pos_set):
    """Get inflections using PyInflect."""
    result = {}
    doc = _get_nlp()(word)
    base_form = doc[0]

    # Define inflection types to check
    pos_to_inflections = {
        "NOUN": [("NNS", "N;PL")],
        "ADJ": [("JJR", "ADJ;CMPR"), ("JJS", "ADJ;SPRL")],
        "ADV": [("RBR", "ADJ;CMPR"), ("RBS", "ADJ;SPRL")],
        # "ADV": [("RBR", "ADV;CMPR"), ("RBS", "ADV;SPRL")],
        "VERB": [
            ("VBD", "V;PST"),
            # ("VBN", "V;V.PTCP;PST"),
            ("VBG", "V;V.PTCP;PRS"),
            ("VBZ", "V;PRS;3;SG"),
        ],
    }

    for pos, inflection_types in pos_to_inflections.items():
        if pos in pos_set:
            for pos_type, unimorph_tag in inflection_types:
                inflection = _get_inflection_from_pyinflect(base_form, pos_type)
                if inflection:
                    if inflection in result and result[inflection] != unimorph_tag:
                        result[inflection] = result[inflection] + "+" + unimorph_tag
                    else:
                        result[inflection] = unimorph_tag

    return result


def vocab_special_cases_filter(base_word, inflection):
    if base_word == "numb" and inflection == "number":
        return True
    return False


def vocab_base_words_filter(base_word):
    if base_word.lower() in [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    ]:
        return True
    return False


def is_english_word(word, unimorph=None):
    """Check if a word is valid English and contains only letters."""
    result = bool(re.fullmatch(r"[a-zA-Z]+", word))
    if unimorph is not None:
        result = result and (_get_dictionary().check(word) or unimorph.is_unimorph_word(word))
    else:
        result = result and _get_dictionary().check(word)
    return result


def get_variation_transformations(
    word,
    base_form=None,
    transforms=None,
    space_prefix=" ",
    decompose_spaces=True,
    decompose_capitalization=True,
):
    """
    Get variations of a word with transformation tags.

    Args:
        word: The word to transform
        base_form: The original base form for comparison
        space_prefix: The space prefix character
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation

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
    # Add current word with transformations
    transforms = [] if transforms is None else transforms
    # If it's the base form without transformations, add NO_POS_INFLECTION
    if not transforms:
        transforms.append(NO_POS_INFLECTION)

    base_transforms = [t for t in transforms]
    base_transforms.append(space_transform)
    base_transforms.append(cap_transform)

    result[word] = base_transforms

    # Generate variations if requested
    if decompose_spaces:
        # Add version without space prefix
        if has_space_prefix:
            word_var = word_without_prefix
            var_transforms = [REMOVE_SPACE_PREFIX_TRANSFORM]
            var_transforms.append(cap_transform)
            result[word_var] = transforms + var_transforms

    if decompose_capitalization:
        # Add capitalized/lowercase versions
        if is_capitalized:
            word_var = (space_prefix if has_space_prefix else "") + word_without_prefix.lower()
            var_transforms = [REMOVE_CAPITALIZATION_TRANSFORM]
            var_transforms.append(space_transform)
            result[word_var] = transforms + var_transforms
        elif not is_capitalized:
            word_var = (space_prefix if has_space_prefix else "") + word_without_prefix.capitalize()
            var_transforms = [ADD_CAPITALIZATION_TRANSFORM]
            var_transforms.append(space_transform)
            result[word_var] = transforms + var_transforms

    # Add other combinations
    if decompose_spaces and decompose_capitalization:
        if has_space_prefix and not is_capitalized:
            # No space, capitalized
            word_var = word_without_prefix.capitalize()
            result[word_var] = transforms + [
                REMOVE_SPACE_PREFIX_TRANSFORM,
                ADD_CAPITALIZATION_TRANSFORM,
            ]
        elif not has_space_prefix and not is_capitalized:
            # With space, capitalized
            word_var = space_prefix + word_without_prefix.capitalize()
            result[word_var] = transforms + [NO_PREFIX_TRANSFORM, ADD_CAPITALIZATION_TRANSFORM]

    return result


def load_ner_cache(cache_path):
    """Load NER cache from pickle file, or return empty dict if not exists."""
    import pickle
    import os

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load NER cache from {cache_path}: {e}")
            return {}
    return {}


def save_ner_cache(cache, cache_path):
    """Save NER cache to pickle file."""
    import pickle
    import os

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Warning: Failed to save NER cache to {cache_path}: {e}")


def is_proper_noun_with_cache(word, ner_cache):
    """
    Check if a word is a proper noun/named entity using spaCy NER.
    Uses cache to avoid repeated spaCy processing.

    Only filters words that are true proper nouns - recognized as entities even in lowercase.
    This avoids filtering common words like "bash", "peter", "meg" that happen to be names
    when capitalized but are primarily common words.

    Conservative entity types: PERSON, ORG, GPE (not WORK_OF_ART, PRODUCT, etc.)

    Args:
        word: The word to check (can have space prefix like 'Ġjohn')
        ner_cache: Dictionary to cache NER results

    Returns:
        Boolean indicating if word is a proper noun/named entity
    """
    # Strip space prefix for processing
    word_clean = word.lstrip("Ġ ").strip()

    if not word_clean:
        return False

    # Check cache first
    if word_clean in ner_cache:
        return ner_cache[word_clean]

    # Conservative entity types that are clearly proper nouns
    PROPER_NOUN_ENTITY_TYPES = {"PERSON", "ORG", "GPE"}

    try:
        # Check lowercase form - if it's recognized as an entity even in lowercase,
        # it's a true proper noun (like "john", "microsoft", "paris")
        lowercase_word = word_clean.lower()
        doc_lower = _get_nlp()(lowercase_word)

        is_entity = False
        if len(doc_lower) > 0:
            lower_entity_type = doc_lower[0].ent_type_
            # Filter if lowercase form is recognized as a proper noun entity
            is_entity = lower_entity_type in PROPER_NOUN_ENTITY_TYPES
    except Exception as e:
        print(f"Warning: NER failed for word '{word_clean}': {e}")
        is_entity = False

    # Cache the result
    ner_cache[word_clean] = is_entity

    return is_entity


def get_vocabulary_decomposition(
    tokenizer,
    unimorph_inflections_path=None,
    unimorph_derivations_path=None,
    space_prefix=" ",
    include_base_only=False,
    include_all_caps=False,
    override_unimorph_with_inflections=True,
    decompose_spaces=True,
    decompose_capitalization=True,
    skip_if_no_space_prefix=False,
    include_derivations=False,
    use_type_str_as_type=False,
    no_diacritics=False,
    filter_propernouns_and_aux=True,
    merge_plural_and_present_singular=False,
    include_non_resource_latin_tokens=False,
    min_derivation_count=2500,
    dont_decompose_proper_nouns=False,
    ner_cache_dir="./cache",
    regenerate_ner_cache=False,
    dont_use_capitalized_base_words=False,
):
    """
    Create a comprehensive token decomposition map using UniMorph data and other resources.

    Args:
        tokenizer: Hugging Face tokenizer
        unimorph_inflections_path: Path to UniMorph inflections data
        unimorph_derivations_path: Path to UniMorph derivations data
        space_prefix: Character used for space prefix
        include_base_only: Whether to include words with no inflections
        include_all_caps: Whether to include all caps words
        override_unimorph_with_inflections: Whether to treat words as inflections even if they appear as base forms
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation
        include_derivations: Whether to include derivations
        min_derivation_count: Minimum count for common derivation types

    Returns:
        Tuple containing:
        - base_tokens: Dict mapping base forms to token IDs
        - decomposition_map: Dict mapping base forms to {inflection: {types: [], types_str: str}}
        - ambiguous_inflections: Dict of ambiguous inflections
        - inflection_baseform_conflicts: List of conflicts between base forms and inflections
    """
    # Initialize analyzers
    if unimorph_inflections_path is None:
        unimorph_inflections_path, default_derivations = get_english_unimorph_paths()
        if unimorph_derivations_path is None:
            unimorph_derivations_path = default_derivations
    print("Loading UniMorph data...")
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
    inflection_baseform_conflicts = []  # For tracking tokens that appear as both base forms and inflections
    duplicate_tags_inflections = defaultdict(
        lambda: defaultdict(set)
    )  # For tracking base forms with multiple inflected forms for the same tag
    filtered_duplicate_inflections = []  # For tracking inflections filtered due to duplicate tags
    filtered_propernouns = set()
    filtered_aux = set()
    filtered_propernouns_ner = set()  # For tracking proper nouns filtered via NER

    # Load NER cache if proper noun filtering is enabled
    ner_cache = {}
    if dont_decompose_proper_nouns:
        import os

        ner_cache_path = os.path.join(ner_cache_dir, "ner_cache.pkl")

        # Delete cache if regeneration requested
        if regenerate_ner_cache and os.path.exists(ner_cache_path):
            print(f"Deleting existing NER cache at {ner_cache_path}")
            os.remove(ner_cache_path)
            ner_cache = {}
        else:
            ner_cache = load_ner_cache(ner_cache_path)
            print(f"Loaded NER cache with {len(ner_cache)} entries from {ner_cache_path}")

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

        if token_str in processed_tokens:
            continue
        processed_tokens.add(token_str)

        # Skip characters and very short words
        if len(token_repr) < 4:
            continue

        # Skip if we're decomposing spaces and this token doesn't have a space prefix
        if decompose_spaces and not token_has_space_prefix:
            continue

        # Skip if not English or contains punctuation
        if not is_english_word(token_repr, unimorph) and not include_non_resource_latin_tokens:
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
        is_base_form = unimorph.is_base_form(token_repr_for_lookup)

        # For non-UniMorph Latin tokens, check if we should include them
        is_latin_lowercase_token = False
        if include_non_resource_latin_tokens and not is_base_form:
            # Check if token is Latin and if capitalizing would produce a different form
            # Check if token is lowercase and capitalizing would create a different token
            if (
                token_repr_for_lookup == token_repr_for_lookup.lower()
                and token_repr_for_lookup.capitalize() != token_repr_for_lookup
            ):
                is_latin_lowercase_token = True

        other_positive_conditions = is_latin_lowercase_token
        if should_preserve_caps and not is_base_form:
            # Try lowercase version to see if this is a capitalized variant of a base word
            lowercase_form = token_repr.lower()
            lowercase_is_base = unimorph.is_base_form(lowercase_form)

            if lowercase_is_base:
                # The lowercase form is a base, so treat this capitalized form as a base too
                # (e.g., " War" is a base because "war" is a base)
                is_base_form = True
                token_repr_for_lookup = lowercase_form
                # We'll use lowercase for lookup but preserve capitalization pattern for results
            else:
                # Neither the capitalized nor lowercase form is a base
                # This is likely an inflected form (e.g., " Wars" where "wars" is not a base)
                # Skip it - it will be added as an inflection when we process its base
                continue

        # Skip if not a base form
        if not is_base_form and not other_positive_conditions:
            continue

        # Skip if skip_if_no_space_prefix is True and token doesn't have space prefix
        if skip_if_no_space_prefix and not token_has_space_prefix:
            continue

        # Skip if already in decomposition map
        if token_str in decomposition_map:
            continue

        # Get possible parts of speech
        pos_set = get_pos_from_wordnet(token_repr_for_lookup)

        # For capitalized words when not decomposing capitalization, also check lowercase version
        # This prevents filtering out valid base words like "War" that have lowercase forms in resources
        if not pos_set and should_preserve_caps:
            lowercase_pos_set = get_pos_from_wordnet(token_repr.lower())
            if lowercase_pos_set:
                pos_set = lowercase_pos_set

        # # HACK: Let's not skip - instead, use this token id as the base form
        # Skip if we're decomposing capitalization and this token doesn't start with an uppercase letter
        if decompose_capitalization and token_is_capitalized:
            if not filter_propernouns_and_aux and (
                is_proper_noun_with_cache(token_repr_for_lookup, ner_cache) or (pos_set & {"PROPN"})
            ):
                if token_str.lower() in processed_tokens:
                    continue
            else:
                continue

        if not pos_set and not other_positive_conditions:
            continue

        if (
            filter_propernouns_and_aux
            and (pos_set & {"AUX", "DET"} or (pos_set & {"PROPN"}))
            and not other_positive_conditions
        ):
            if "PROPN" in pos_set:
                filtered_propernouns.add(token_str)
            else:
                filtered_aux.add(token_str)
            continue

        # Filter proper nouns using NER
        # Logic depends on decompose_capitalization setting:
        # - If decomposing capitalization: filter lowercase bases that are proper nouns (e.g., " john" if "John" is a name)
        # - If not decomposing: only filter capitalized tokens (e.g., " John") to avoid filtering common words
        should_check_ner = False
        if dont_decompose_proper_nouns or filter_propernouns_and_aux:
            should_check_ner = True

        if should_check_ner and is_proper_noun_with_cache(token_repr_for_lookup, ner_cache):
            filtered_propernouns_ner.add(token_str)
            continue

        # Filter special cases (e.g., stop words)
        if vocab_base_words_filter(token_repr_for_lookup):
            continue

        # Start building decomposition for this token
        inflections_map = {}

        has_inflections = False

        if is_base_form:
            base_variations = get_variation_transformations(
                token_str, None, None, space_prefix, decompose_spaces, decompose_capitalization
            )

            for var, transforms in base_variations.items():
                if var not in inflections_map:
                    inflections_map[var] = transforms
                    has_inflections = True

            # Get inflections from UniMorph
            unimorph_inflections = unimorph.get_inflections(
                token_repr_for_lookup
            ) | unimorph.get_inflections(token_repr_for_lookup.lower())

            # Function to apply capitalization pattern from original to target string
            def apply_capitalization_pattern(original, target):
                if original[0].isupper() and len(target) > 0:
                    return target[0].upper() + target[1:]
                return target

            # Process UniMorph inflections
            for inflected_form, tag_sets in unimorph_inflections.items():
                # Apply capitalization pattern if needed
                if should_preserve_caps:
                    inflected_form = apply_capitalization_pattern(
                        original_token_repr, inflected_form
                    )

                # Skip the base form itself (compare against the form we looked up)
                if inflected_form == token_repr_for_lookup:
                    continue

                if vocab_special_cases_filter(token_repr_for_lookup, inflected_form):
                    continue

                # remove ('V', 'NFIN', 'IMP+SBJV') tag -
                if ("V", "NFIN", "IMP+SBJV") in tag_sets and len(tag_sets) > 1:
                    tag_sets.remove(
                        ("V", "NFIN", "IMP+SBJV")
                    )  # skip if it's the only one (the word always appears twice, the other with a regular tag - actually this is wrong)

                # If there are both ('V', 'PST') and ('V', 'V.PTCP', 'PST') -
                # keep only ('V', 'PST'), leave ('V', 'V.PTCP', 'PST') for irregular verbs with distinct forms
                if ("V", "V.PTCP", "PST") in tag_sets:
                    if ("V", "PST") in tag_sets:
                        tag_sets.remove(("V", "V.PTCP", "PST"))

                # There are some N;PL+V;PRS;3;SG words that are only labeled as verbs in UniMorph
                # So a hack to relabel them using wordnet POS labels:
                if len(tag_sets) == 1 and ";".join(list(tag_sets)[0]) == "V;PRS;3;SG":
                    inflected_pos_set = get_pos_from_wordnet(inflected_form)
                    if "NOUN" in inflected_pos_set:
                        tag_sets.add(("N", "PL"))

                # Check for ambiguous inflections
                if len(tag_sets) > 1:
                    # Combine tags maintaining a consistent order
                    combined_tags = []
                    for tag_set in tag_sets:
                        tag_str = ";".join(tag_set)
                        combined_tags.append(tag_str)

                    # Sort for consistent order across words
                    tag_str = "+".join(sorted(combined_tags))
                else:
                    tag_str = ";".join(list(tag_sets)[0])

                # Check for conflicts with other base forms
                if inflected_form.lower() in unimorph.lemma_to_forms:
                    inflection_baseform_conflicts.append(
                        (original_token_repr, inflected_form, tag_str)
                    )
                    # Skip if we're not overriding
                    if not override_unimorph_with_inflections:
                        continue

                # Store mapping of base word -> tag -> inflected forms - for tracking duplicates
                # Use lowercase for internal tracking to avoid duplicate capitalization issues
                for tag_set in tag_sets:
                    tag_str_single = ";".join(sorted(tag_set))
                    duplicate_tags_inflections[token_repr_for_lookup][tag_str_single].add(
                        inflected_form.lower()
                    )

                # Check if this is a duplicate tag inflection
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

                # Skip this inflection if it has a duplicate tag
                if is_duplicate:
                    continue

                # Add inflection with proper space prefix
                inflected_with_prefix = (
                    space_prefix + inflected_form if token_has_space_prefix else inflected_form
                )

                inflected_variations = get_variation_transformations(
                    inflected_with_prefix,
                    token_str,
                    [tag_str],
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                )

                # Add the inflection type to all variations
                for var, transforms in inflected_variations.items():
                    if var not in inflections_map:
                        inflections_map[var] = transforms
                        has_inflections = True

            # Get additional inflections from PyInflect if they're not in UniMorph
            pyinflect_variations = get_pyinflect_variations(token_repr_for_lookup, pos_set)
            for inflected_form, tag_str in pyinflect_variations.items():
                # Apply capitalization pattern if needed
                if should_preserve_caps:
                    inflected_form = apply_capitalization_pattern(
                        original_token_repr, inflected_form
                    )

                # Skip if already in inflections map
                inflected_with_prefix = (
                    space_prefix + inflected_form if token_has_space_prefix else inflected_form
                )
                if any(var.lower() == inflected_with_prefix.lower() for var in inflections_map):
                    continue

                # Check if it's a valid English word
                if not is_english_word(inflected_form):
                    continue

                # Check for conflicts with base forms
                if inflected_form.lower() in unimorph.lemma_to_forms:
                    inflection_baseform_conflicts.append(
                        (original_token_repr, inflected_form, tag_str)
                    )
                    # Skip if we're not overriding
                    if not override_unimorph_with_inflections:
                        continue

                inflected_variations = get_variation_transformations(
                    inflected_with_prefix,
                    token_str,
                    [tag_str],
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                )

                # Add the inflection type to all variations
                for var, transforms in inflected_variations.items():
                    if var not in inflections_map:
                        inflections_map[var] = transforms
                        has_inflections = True

            # Get derivations if requested
            if include_derivations and unimorph_derivations_path:
                derivations = unimorph.get_root_derivations(token_repr_for_lookup)
                for derived_form, type_pos_sets in derivations.items():
                    # Apply capitalization pattern if needed
                    if should_preserve_caps:
                        derived_form = apply_capitalization_pattern(
                            original_token_repr, derived_form
                        )

                    # Get the derivation type
                    for deriv_type, pos in type_pos_sets:
                        # Check for conflicts with base forms
                        if derived_form.lower() in unimorph.lemma_to_forms:
                            inflection_baseform_conflicts.append(
                                (original_token_repr, derived_form, deriv_type)
                            )
                            # Skip if we're not overriding
                            if not override_unimorph_with_inflections:
                                continue

                        # Add derivation with proper space prefix
                        derived_with_prefix = (
                            space_prefix + derived_form if token_has_space_prefix else derived_form
                        )

                        derived_variations = get_variation_transformations(
                            derived_with_prefix,
                            token_str,
                            [deriv_type],
                            space_prefix,
                            decompose_spaces,
                            decompose_capitalization,
                        )

                        # Add the derivation type to all variations
                        for var, transforms in derived_variations.items():
                            if var not in inflections_map:
                                inflections_map[var] = transforms
                                has_inflections = True
        elif other_positive_conditions:
            # For Latin tokens that aren't in UniMorph, just add capitalization variations
            if is_latin_lowercase_token:
                # For Latin tokens, we only need to add capitalization variations
                capitalized_form = token_repr_for_lookup.capitalize()

                # Add with proper space prefix
                if token_has_space_prefix:
                    capitalized_with_prefix = space_prefix + capitalized_form
                else:
                    capitalized_with_prefix = capitalized_form

                # Only add if it's different from the original
                if capitalized_with_prefix != token_str:
                    # Get variations with capitalization transformation
                    capitalized_variations = get_variation_transformations(
                        capitalized_with_prefix,
                        token_str,
                        None,
                        space_prefix,
                        decompose_spaces,
                        decompose_capitalization,
                    )

                    # Add all variations to inflections map
                    for var, transforms in capitalized_variations.items():
                        if var not in inflections_map:
                            inflections_map[var] = transforms
                            has_inflections = True
                print(token_repr_for_lookup)
                print(inflections_map)

        # Check if we have inflections
        if not has_inflections and not include_base_only:
            continue

        # Add to decomposition map
        decomposition_map[token_str] = {}
        for form, transforms in inflections_map.items():
            if "N;SG" in transforms:
                # HACK: cases where the base form is capitalized - skip these for now
                print(token_str, form)
                continue
            # Format types as list and string
            if merge_plural_and_present_singular:
                transforms = list(
                    map(
                        lambda x: {"N;PL": "N;PL+V;PRS;3;SG", "V;PRS;3;SG": "N;PL+V;PRS;3;SG"}.get(
                            x, x
                        ),
                        transforms,
                    )
                )
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
            for form, types in decomposition_map[token_str].items():
                # Use types_str if use_type_str_as_type is True
                if use_type_str_as_type:
                    type_to_words[types[0]].add(form)
                else:
                    for type_tag in types:
                        type_to_words[type_tag].add(form)

        # Add to base tokens
        base_tokens[token_str] = token_id
    # clean decomposition map of inflections that are also base words
    for base_word, inflections in decomposition_map.items():
        replace_inflections = False
        valid_inflections = dict()
        for inflected_form, transforms in inflections.items():
            if inflected_form.lower() != base_word and inflected_form.lower() in decomposition_map:
                replace_inflections = True
            else:
                valid_inflections[inflected_form] = transforms
        if replace_inflections:
            decomposition_map[base_word] = valid_inflections

    # Filter out capitalized base words if requested
    filtered_capitalized_bases = set()
    if dont_use_capitalized_base_words:
        bases_to_remove = []
        for base_token in decomposition_map.keys():
            # Remove space prefix to check actual word
            token_clean = base_token.lstrip("Ġ ").strip()
            # Check if first character is capitalized
            if token_clean and token_clean[0].isupper():
                bases_to_remove.append(base_token)
                filtered_capitalized_bases.add(base_token)

        # Remove capitalized bases from decomposition_map and base_tokens
        for base_token in bases_to_remove:
            del decomposition_map[base_token]
            if base_token in base_tokens:
                del base_tokens[base_token]

        print(f"Filtered {len(filtered_capitalized_bases)} capitalized base words")

    print(f"Built decomposition map with {len(decomposition_map)} base tokens")
    print(f"Found {len(ambiguous_inflections)} ambiguous inflections")
    print(
        f"Found {len(inflection_baseform_conflicts)} conflicts between base forms and inflections"
    )

    # Collect and report all duplicate tag inflections
    print("\nSummary of duplicate tag inflections:")
    duplicate_tag_count = 0
    for base, tag_inflections in duplicate_tags_inflections.items():
        for tag, inflections in tag_inflections.items():
            if len(inflections) > 1:
                duplicate_tag_count += 1
                print(f"  {base} -> {tag}: {', '.join(inflections)}")

    print(f"Found {duplicate_tag_count} cases of duplicate tag inflections")
    print(f"Filtered {len(filtered_duplicate_inflections)} inflections due to duplicate tags")
    print(f"Filtered {len(filtered_propernouns)} propernoun tokens")
    print(f"Filtered {len(filtered_aux)} AUX tokens")
    if dont_decompose_proper_nouns:
        print(f"Filtered {len(filtered_propernouns_ner)} proper noun tokens via NER")

    type_to_words = {k: list(v) for k, v in type_to_words.items()}

    # Save NER cache if proper noun filtering was enabled
    if dont_decompose_proper_nouns and ner_cache:
        import os

        ner_cache_path = os.path.join(ner_cache_dir, "ner_cache.pkl")
        save_ner_cache(ner_cache, ner_cache_path)
        print(f"Saved NER cache with {len(ner_cache)} entries to {ner_cache_path}")

    filtered = {
        "duplicates": filtered_duplicate_inflections,
        "propernouns": list(filtered_propernouns),
        "aux": list(filtered_aux),
        "propernouns_ner": list(filtered_propernouns_ner) if dont_decompose_proper_nouns else [],
        "capitalized_bases": list(filtered_capitalized_bases)
        if dont_use_capitalized_base_words
        else [],
    }
    return (
        base_tokens,
        decomposition_map,
        type_to_words,
        ambiguous_inflections,
        inflection_baseform_conflicts,
        filtered,
    )


def get_label_maps_from_decomposition_map(decomposition_map):
    """
    Extract unique transformation types from the decomposition map.

    Args:
        decomposition_map (dict): Decomposition map created by get_vocabulary_decomposition

    Returns:
        dict: Mapping of transformation types to unique integer indices
    """
    # Collect all unique transformation types
    transformation_types = set()

    # Special transforms we want to include by default
    default_transforms = [
        NO_POS_INFLECTION,
        REMOVE_SPACE_PREFIX_TRANSFORM,
        NO_PREFIX_TRANSFORM,
        ADD_CAPITALIZATION_TRANSFORM,
        REMOVE_CAPITALIZATION_TRANSFORM,
        NO_CAPITALIZATION_TRANSFORM,
    ]

    # Collect all transformation types from decomposition map
    for base_token, inflections in decomposition_map.items():
        for inflection_form, inflection_info in inflections.items():
            # Extract individual types from types_str which is concatenated with '+'
            types = inflection_info
            transformation_types.update(types)

    # Convert to sorted list to ensure consistent ordering
    transformation_types = transformation_types - set(default_transforms)
    transformations_names = sorted(
        list(transformation_types), key=lambda s: (0 if "-" in s else 1, s)
    )

    # Add default transforms last
    transformations_names = transformations_names + default_transforms

    # Create mapping to integers
    transformations_to_int = dict(zip(transformations_names, range(len(transformations_names))))

    # create map to indices for losses
    loss_indices = {
        "inflections": (0, transformations_names.index(NO_POS_INFLECTION) + 1),
        "prefix": (
            transformations_names.index(REMOVE_SPACE_PREFIX_TRANSFORM),
            transformations_names.index(NO_PREFIX_TRANSFORM) + 1,
        ),
        "capitalization": (
            transformations_names.index(ADD_CAPITALIZATION_TRANSFORM),
            transformations_names.index(NO_CAPITALIZATION_TRANSFORM) + 1,
        ),
    }

    return transformations_to_int, loss_indices


def get_type_group_idx_from_decomposition_map(decomposition_map):
    """
    Extract unique transformation types from the decomposition map.

    Args:
        decomposition_map (dict): Decomposition map created by get_vocabulary_decomposition

    Returns:
        dict: Mapping of transformation types to unique integer indices
    """
    # Collect all unique transformation types
    transformation_types = set()

    # Special transforms we want to include by default
    default_transforms = [
        NO_POS_INFLECTION,
        REMOVE_SPACE_PREFIX_TRANSFORM,
        NO_PREFIX_TRANSFORM,
        ADD_CAPITALIZATION_TRANSFORM,
        REMOVE_CAPITALIZATION_TRANSFORM,
        NO_CAPITALIZATION_TRANSFORM,
    ]

    # Collect all transformation types from decomposition map
    for base_token, inflections in decomposition_map.items():
        for inflection_form, inflection_info in inflections.items():
            # Extract individual types from types_str which is concatenated with '+'
            types = inflection_info
            transformation_types.update(types)

    # Convert to sorted list to ensure consistent ordering
    transformation_types = transformation_types - set(default_transforms)
    transformations_names = sorted(
        list(transformation_types), key=lambda s: (0 if "-" in s else 1, s)
    )

    # Add default transforms last
    transformations_names = transformations_names + default_transforms

    # Create mapping to integers
    transformations_to_int = dict(zip(transformations_names, range(len(transformations_names))))

    last_derivation_idx = max(
        (i for i, s in enumerate(transformations_names) if "-" in s), default=-1
    )
    # create map to indices by type group
    type_indices = {
        "derivations": (0, last_derivation_idx + 1),
        "inflections": (last_derivation_idx, transformations_names.index(NO_POS_INFLECTION) + 1),
        "prefix": (
            transformations_names.index(REMOVE_SPACE_PREFIX_TRANSFORM),
            transformations_names.index(NO_PREFIX_TRANSFORM) + 1,
        ),
        "capitalization": (
            transformations_names.index(ADD_CAPITALIZATION_TRANSFORM),
            transformations_names.index(NO_CAPITALIZATION_TRANSFORM) + 1,
        ),
    }

    return type_indices


def extend_tokenizer_by_decomposition_map(
    tokenizer,
    base_tokens,
    decomposition_map,
    space_prefix=None,
    skip_multi_token_words=False,
    skip_three_token_words=False,
    transformations_to_int=None,
    decompose_spaces=True,
    decompose_capitalization=True,
    predict_negative_types=False,
):
    vocab_size = len(tokenizer._tokenizer.get_vocab(with_added_tokens=False))

    # TODO fix: is there a problem with adjectives like "Western"?
    # TODO fix: Adverbs are not treated as inflections of adjectives, but are their own base form
    if transformations_to_int is None:
        raise ValueError("transformations_to_int missing")
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
        try:
            base_token_to_valid_transformations[base_token_id] = [
                transformations_to_int[transformation]
                for transformation in set(chain(*decomposition_map[base_word].values()))
            ]
        except KeyError as exc:
            raise ValueError(f"Unknown transformation for base {base_word!r}") from exc
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

    # Filter negative types based on settings
    negative_type_names = filter_active_negative_types(
        decompose_spaces, decompose_capitalization, predict_negative_types
    )
    negative_types = set(
        [
            transformations_to_int[type_name]
            for type_name in negative_type_names
            if type_name in transformations_to_int
        ]
    )

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


import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def add_words_to_core_vocab(tokenizer, new_tokens_to_ids, merges_list):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save tokenizer to the temporary directory
        tokenizer.save_pretrained(temp_dir)

        if os.path.isfile(os.path.join(temp_dir, "tokenizer.model")):
            # sentencepiece tokenizer

            model_file = "tokenizer.model"

            model_path = os.path.join(temp_dir, model_file)

            # Load sentence piece model
            model = sp_pb2_model.ModelProto()
            with open(model_path, "rb") as f:
                model.ParseFromString(f.read())

            # Load the tokenizer.json file
            tokenizer_json_path = f"{temp_dir}/tokenizer.json"
            with open(tokenizer_json_path, "r") as f:
                orig_tokenizer_json = json.load(f)
            tokenizer_json = deepcopy(orig_tokenizer_json)
            # Make edits to the tokenizer.json
            for word, token_id in new_tokens_to_ids.items():
                tokenizer_json["model"]["vocab"][word] = token_id

            tokenizer_json["model"]["merges"] = tokenizer_json["model"]["merges"] + merges_list

            # Save the edited tokenizer.json as a temp file and reload the modified tokenizer
            with open(tokenizer_json_path, "w") as f:
                json.dump(tokenizer_json, f, indent=2)

            # Make edits to the sp model
            # First, create a mapping of existing tokens to avoid duplicates
            existing_tokens = {piece.piece: i for i, piece in enumerate(model.pieces)}

            for word, token_id in new_tokens_to_ids.items():
                if word in existing_tokens:
                    print(f"SentencePiece editing - Token '{word}' already exists, skipping")
                    continue

                new_piece = sp_pb2_model.ModelProto.SentencePiece()
                new_piece.piece = word
                new_piece.score = -10.0  # Lower score for added tokens
                new_piece.type = sp_pb2_model.ModelProto.SentencePiece.NORMAL

                model.pieces.append(new_piece)

            # Save modified model
            with open(model_path, "wb") as f:
                f.write(model.SerializeToString())

            # Load new tokenizer
            new_tokenizer = AutoTokenizer.from_pretrained(
                temp_dir, added_tokens_decoder=tokenizer.added_tokens_decoder.copy()
            )

        else:
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


def get_ignored_types():
    return NO_EMBEDDING_TYPES


def filter_active_negative_types(
    decompose_spaces, decompose_capitalization, predict_negative_types
):
    """
    Determine which negative types should be treated as zero (not predicted).

    When predict_negative_types=True, we want to PREDICT the active identity types,
    so we exclude them from the negative_types set.

    Args:
        decompose_spaces: Whether space decomposition is active
        decompose_capitalization: Whether capitalization decomposition is active
        predict_negative_types: Whether to enable prediction of identity types

    Returns:
        Set of type names that should remain as negative (zero) types
    """
    if not predict_negative_types:
        # Original behavior: all NO_EMBEDDING_TYPES are negative (zero)
        return set(NO_EMBEDDING_TYPES)

    # When predicting negative types, only keep as "negative" the ones that aren't active
    negative_types = set()

    # NO_POS_INFLECTION ('none') - always active, so make it predictable
    # (don't add to negative_types)

    # NO_PREFIX_TRANSFORM ('no_space_prefix') - only relevant if decompose_spaces
    if not decompose_spaces:
        negative_types.add(NO_PREFIX_TRANSFORM)

    # NO_CAPITALIZATION_TRANSFORM ('no_capitalization') - only relevant if decompose_capitalization
    if not decompose_capitalization:
        negative_types.add(NO_CAPITALIZATION_TRANSFORM)

    # 'N;SG' - not a true identity type, just a UniMorph artifact - always keep as negative
    negative_types.add("N;SG")

    return negative_types


def apply_linear_rep_hypothesis_formulation(embeddings):
    gamma = embeddings.to(torch.float32)
    W, d = gamma.shape
    gamma_bar = torch.mean(gamma, dim=0)
    centered_gamma = gamma - gamma_bar

    ### compute Cov(gamma) and tranform gamma to g ###
    Cov_gamma = centered_gamma.T @ centered_gamma / W
    eigenvalues, eigenvectors = torch.linalg.eigh(Cov_gamma)
    inv_sqrt_Cov_gamma = eigenvectors @ torch.diag(1 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    g = gamma @ inv_sqrt_Cov_gamma
    g = g.to(embeddings.dtype)
    return g


def get_initial_type_embeddings_from_strings(
    base_tokens,
    decomposition_map,
    tokenizer,
    input_embeddings,
    output_embeddings,
    type_names_to_int,
    space_prefix="Ġ",
    enforce_allowed_multitypes: bool = True,
    project_to_decomposition_space: bool = False,
    projection_use_svd: bool = False,
    remove_non_decomposition_directions: bool = False,
    return_decomp_token_ids: bool = False,
    linear_rep_hypothesis_formulation: bool = False,
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
        enforce_allowed_multitypes: Whether to enforce allowed multi-types
        project_to_decomposition_space: Whether to project the transformation vectors into the space
                                        spanned by the embeddings of tokens in the decomposition map
        projection_use_svd: Whether to use SVD for projection (more stable) or regular projection
        remove_non_decomposition_directions: Whether to remove directions of tokens not in the decomposition map
                                            (i.e., project onto the orthogonal complement of non-decomposition tokens)

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
    forbidden_multi_types = set(
        [type_names_to_int[type_name] for type_name in FORBID_MULTI_TYPES_IN_INIT]
    )
    ignored_types = set(
        [
            type_names_to_int[type_name]
            for type_name in NO_EMBEDDING_TYPES
            if type_name in type_names_to_int
        ]
    )

    # Choose if to apply the reformulation from the linear representation hypothesis paper
    if linear_rep_hypothesis_formulation:
        input_embeddings = apply_linear_rep_hypothesis_formulation(input_embeddings)
        output_embeddings = apply_linear_rep_hypothesis_formulation(output_embeddings)

    # Collect token IDs from decomposition map for projection
    decomposition_token_ids = set()

    for base_word, inflections in decomposition_map.items():
        base_token_id = base_tokens[base_word]
        decomposition_token_ids.add(base_token_id)

        for inflection_word in inflections.keys():
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

            decomposition_token_ids.add(inflection_token_id)

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
                    if enforce_allowed_multitypes and (type_id not in allowed_multi_types):
                        continue
                    if type_id in forbidden_multi_types:
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

    # Project to decomposition space if requested
    if project_to_decomposition_space and decomposition_token_ids:
        # Convert set to list for indexing
        decomposition_token_ids = list(decomposition_token_ids)

        # Extract embeddings for tokens in decomposition map
        input_decomp_embeddings = input_embeddings[decomposition_token_ids]
        output_decomp_embeddings = output_embeddings[decomposition_token_ids]

        # Project input type embeddings
        input_type_embeddings = project_to_subspace(
            input_type_embeddings, input_decomp_embeddings, use_svd=projection_use_svd
        )

        # Project output type embeddings
        output_type_embeddings = project_to_subspace(
            output_type_embeddings, output_decomp_embeddings, use_svd=projection_use_svd
        )

    # Remove non-decomposition directions if requested
    if remove_non_decomposition_directions and decomposition_token_ids:
        # Find token IDs not in decomposition
        all_token_ids = set(range(vocab_size))
        non_decomp_token_ids = list(all_token_ids - set(decomposition_token_ids))

        if non_decomp_token_ids:
            # Extract embeddings for tokens not in decomposition map
            input_non_decomp_embeddings = input_embeddings[non_decomp_token_ids]
            output_non_decomp_embeddings = output_embeddings[non_decomp_token_ids]

            # Remove components in the space of non-decomposition tokens
            input_type_embeddings = remove_subspace_components(
                input_type_embeddings, input_non_decomp_embeddings, use_svd=projection_use_svd
            )

            output_type_embeddings = remove_subspace_components(
                output_type_embeddings, output_non_decomp_embeddings, use_svd=projection_use_svd
            )

    if return_decomp_token_ids:
        decomp_token_ids = list(decomposition_token_ids)
        all_token_ids = set(range(vocab_size))
        non_decomp_token_ids = list(all_token_ids - set(decomposition_token_ids))
        return input_type_embeddings, output_type_embeddings, decomp_token_ids, non_decomp_token_ids
    else:
        return input_type_embeddings, output_type_embeddings


def project_to_subspace(vectors, basis_vectors, use_svd=True):
    """
    Project vectors to the subspace spanned by basis_vectors.

    Args:
        vectors: Tensor of shape (n_vectors, embedding_dim) to be projected
        basis_vectors: Tensor of shape (n_basis, embedding_dim) defining the subspace
        use_svd: Whether to use SVD for orthogonalization (more stable) or regular projection

    Returns:
        Tensor of shape (n_vectors, embedding_dim) - The projected vectors
    """
    if use_svd:
        vectors = vectors.to(torch.float32)
        basis_vectors = basis_vectors.to(torch.float32)
        # Orthogonalize the basis vectors using SVD
        # This handles the case where basis vectors might be linearly dependent
        U, S, Vh = torch.linalg.svd(basis_vectors, full_matrices=False)

        # Keep only components with significant singular values
        # Helps avoid numerical instability
        eps = 1e-8
        mask = S > eps
        U = U[:, mask]

        # Project vectors onto the orthogonal basis
        projections = torch.matmul(vectors, U.transpose(-2, -1))
        projected_vectors = torch.matmul(projections, U)
    else:
        vectors = vectors.to(torch.float32)
        basis_vectors = basis_vectors.to(torch.float32)

        # B: (n_basis, d)
        B = basis_vectors  # (n_basis, d)
        d = B.shape[1]

        # Compute (B^T B)^(-1)
        BtB = B.T @ B  # (d, d)
        reg = 1e-8 * torch.eye(d, device=B.device)
        BtB_inv = torch.linalg.inv(BtB + reg)  # (d, d)

        # Precompute B^T: (d, n_basis)
        Bt = B.T

        # For each input vector v, project: B @ BtB_inv @ Bt @ v
        # vectors: (n_vectors, d)
        projections = []
        for v in vectors:
            coeffs = (BtB_inv @ v) @ Bt  # (d,)
            proj = coeffs @ B  # (d,)
            projections.append(proj)

        projected_vectors = torch.stack(projections, dim=0)  # (n_vectors, d)

    return projected_vectors


def remove_subspace_components(vectors, basis_vectors, use_svd=True):
    """
    Remove components that lie in the subspace spanned by basis_vectors.
    This projects vectors onto the orthogonal complement of the subspace.

    Args:
        vectors: Tensor of shape (n_vectors, embedding_dim) to be projected
        basis_vectors: Tensor of shape (n_basis, embedding_dim) defining the subspace to remove
        use_svd: Whether to use SVD for orthogonalization (more stable) or regular projection

    Returns:
        Tensor of shape (n_vectors, embedding_dim) - The vectors with subspace components removed
    """
    # Project vectors onto the basis subspace
    projection = project_to_subspace(vectors, basis_vectors, use_svd=use_svd)

    # Subtract the projection to get the orthogonal complement
    return vectors - projection
