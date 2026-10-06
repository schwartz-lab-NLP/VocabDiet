import os
import re
import nltk
import enchant
import spacy
import pyinflect
from nltk.corpus import wordnet as wn
from typing import Dict, List, Tuple, Set, Optional, DefaultDict
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
import tempfile
import json
import unicodedata
from itertools import chain
from copy import deepcopy

_SPACY_NLP = None
_ENGLISH_DICTIONARY = None


def _get_spacy_nlp():
    """Load English NLP resources only when decomposition actually needs them."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except (OSError, ImportError) as exc:
            raise RuntimeError(
                "English decomposition requires spaCy and the en_core_web_sm model. "
                "Install the model with `python -m spacy download en_core_web_sm`."
            ) from exc
    return _SPACY_NLP


def _get_english_dictionary():
    """Load the system English dictionary only when word checks need it."""
    global _ENGLISH_DICTIONARY
    if _ENGLISH_DICTIONARY is None:
        try:
            _ENGLISH_DICTIONARY = enchant.Dict("en_US")
        except (enchant.DictNotFoundError, OSError, ImportError) as exc:
            raise RuntimeError(
                "English decomposition requires the PyEnchant package and an en_US dictionary."
            ) from exc
    return _ENGLISH_DICTIONARY


def _wordnet_synsets(word):
    try:
        return wn.synsets(word)
    except LookupError as exc:
        raise RuntimeError(
            "English morphology requires the NLTK WordNet corpus. Install it with "
            "`python -m nltk.downloader wordnet`."
        ) from exc


# Constants for transformation types
# Space prefix transformations
WITH_SPACE_PREFIX_TRANSFORM = "with_space_prefix"
NO_SPACE_PREFIX_TRANSFORM = "no_space_prefix"
REMOVE_SPACE_PREFIX_TRANSFORM = "remove_space_prefix"
NA_SPACE_PREFIX_TRANSFORM = "NA_space_prefix"

# Base capitalization transformations (existing, renamed for clarity)
NO_BASE_CAPITALIZATION_TRANSFORM = "no_capitalization"
ADD_BASE_CAPITALIZATION_TRANSFORM = "add_capitalization"
ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM = "add_all_caps"
REMOVE_BASE_CAPITALIZATION_TRANSFORM = "remove_capitalization"
NA_BASE_CAPITALIZATION_TRANSFORM = "NA_capitalization"

# Legacy names for backwards compatibility
NO_CAPITALIZATION_TRANSFORM = NO_BASE_CAPITALIZATION_TRANSFORM
ADD_CAPITALIZATION_TRANSFORM = ADD_BASE_CAPITALIZATION_TRANSFORM
ADD_ALL_CAPS_CAPITALIZATION_TRANSFORM = ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
REMOVE_CAPITALIZATION_TRANSFORM = REMOVE_BASE_CAPITALIZATION_TRANSFORM
NA_CAPITALIZATION_TRANSFORM = NA_BASE_CAPITALIZATION_TRANSFORM

# Inflection and derivation transformations
NO_INFLECTION = "none"
NA_INFLECTION = "NA_inflection"
NO_DERIVATION = "no_derivation"
NA_DERIVATION = "NA_derivation"

# Prefix punctuation transformations (combinations of opening marks)
NO_PREFIX_PUNCTUATION = "no_prefix_punct"
NA_PREFIX_PUNCTUATION = "NA_prefix_punct"
# Single marks
PREFIX_PUNCT_SINGLE_QUOTE = "punct_prefix_'"
PREFIX_PUNCT_DOUBLE_QUOTE = 'punct_prefix_"'
PREFIX_PUNCT_BACKTICK = "punct_prefix_`"
PREFIX_PUNCT_PAREN = "punct_prefix_("
PREFIX_PUNCT_SQUARE = "punct_prefix_["
PREFIX_PUNCT_CURLY = "punct_prefix_{"
PREFIX_PUNCT_HYPHEN = "punct_prefix_-"
# Two-mark combinations (most common)
PREFIX_PUNCT_SINGLE_PAREN = "punct_prefix_'("
PREFIX_PUNCT_DOUBLE_PAREN = 'punct_prefix_"('
PREFIX_PUNCT_SINGLE_SQUARE = "punct_prefix_'["
PREFIX_PUNCT_DOUBLE_SQUARE = 'punct_prefix_"['
PREFIX_PUNCT_PAREN_SINGLE = "punct_prefix_('\""  # "(' special case
PREFIX_PUNCT_HYPHEN_PAREN = "punct_prefix_-("
PREFIX_PUNCT_HYPHEN_SQUARE = "punct_prefix_-["

# Suffix punctuation transformations (combinations of closing marks)
NO_SUFFIX_PUNCTUATION = "no_suffix_punct"
NA_SUFFIX_PUNCTUATION = "NA_suffix_punct"
# Single marks - quotes
SUFFIX_PUNCT_SINGLE_QUOTE = "punct_suffix_'"
SUFFIX_PUNCT_DOUBLE_QUOTE = 'punct_suffix_"'
SUFFIX_PUNCT_BACKTICK = "punct_suffix_`"
# Single marks - brackets
SUFFIX_PUNCT_PAREN = "punct_suffix_)"
SUFFIX_PUNCT_SQUARE = "punct_suffix_]"
SUFFIX_PUNCT_CURLY = "punct_suffix_}"
# Single marks - terminal
SUFFIX_PUNCT_PERIOD = "punct_suffix_."
SUFFIX_PUNCT_EXCLAIM = "punct_suffix_!"
SUFFIX_PUNCT_QUESTION = "punct_suffix_?"
SUFFIX_PUNCT_COMMA = "punct_suffix_,"
SUFFIX_PUNCT_SEMICOLON = "punct_suffix_;"
SUFFIX_PUNCT_COLON = "punct_suffix_:"
# Single marks - possessive and hyphen
SUFFIX_PUNCT_POSSESSIVE_S = "punct_suffix_'s"
SUFFIX_PUNCT_POSSESSIVE_PLURAL = (
    "punct_suffix_'"  # Same as single quote but different semantic meaning
)
SUFFIX_PUNCT_HYPHEN = "punct_suffix_-"
# Two-mark combinations (most common)
SUFFIX_PUNCT_PAREN_SINGLE = "punct_suffix_)'"
SUFFIX_PUNCT_PAREN_DOUBLE = 'punct_suffix_)"'
SUFFIX_PUNCT_SQUARE_SINGLE = "punct_suffix_]'"
SUFFIX_PUNCT_SQUARE_DOUBLE = 'punct_suffix_]"'
SUFFIX_PUNCT_PERIOD_SINGLE = "punct_suffix_.'"
SUFFIX_PUNCT_PERIOD_DOUBLE = 'punct_suffix_."'
SUFFIX_PUNCT_EXCLAIM_SINGLE = "punct_suffix_!'"
SUFFIX_PUNCT_EXCLAIM_DOUBLE = 'punct_suffix_!"'
SUFFIX_PUNCT_QUESTION_SINGLE = "punct_suffix_?'"
SUFFIX_PUNCT_QUESTION_DOUBLE = 'punct_suffix_?"'
SUFFIX_PUNCT_COMMA_SINGLE = "punct_suffix_,'"
SUFFIX_PUNCT_COMMA_DOUBLE = 'punct_suffix_,"'
SUFFIX_PUNCT_PERIOD_PAREN = "punct_suffix_.)"
SUFFIX_PUNCT_EXCLAIM_PAREN = "punct_suffix_!)"
SUFFIX_PUNCT_QUESTION_PAREN = "punct_suffix_?)"
SUFFIX_PUNCT_COMMA_PAREN = "punct_suffix_,)"

# Pronoun possessive forms that should not be treated as inflections.
POSSESSIVE_PRONOUN_SKIP = {
    ("it", "its"),
}
SUFFIX_PUNCT_PAREN_HYPHEN = "punct_suffix_)-"
SUFFIX_PUNCT_SQUARE_HYPHEN = "punct_suffix_]-"

# Article transformations
NO_ARTICLE = "no_article"
NA_ARTICLE = "NA_article"
ARTICLE_THE = "article_the"
ARTICLE_A = "article_a"
ARTICLE_AN = "article_an"

# Preposition transformations
NO_PREPOSITION = "no_prep"
NA_PREPOSITION = "NA_prep"
# Most frequent prepositions (default list)
PREP_BY = "prep_by"
PREP_AT = "prep_at"
PREP_OF = "prep_of"
PREP_TO = "prep_to"
PREP_IN = "prep_in"
PREP_ON = "prep_on"
PREP_WITH = "prep_with"
PREP_FOR = "prep_for"
PREP_FROM = "prep_from"

# Preposition capitalization transformations
NO_PREP_CAPITALIZATION = "no_prep_cap"
ADD_PREP_CAPITALIZATION = "add_prep_cap"
NA_PREP_CAPITALIZATION = "NA_prep_cap"

# Spanish enclitic pronouns (for disambiguating UniMorph duplicate tags)
SPANISH_CLITICS = [
    "los",
    "las",
    "les",
    "lo",
    "la",
    "le",
    "me",
    "te",
    "se",
    "nos",
    "os",
]

MORPHOLOGICAL_GROUPS = [
    "pos",  # V, N, ADJ, ADV, PRON, DET
    "tense",  # PRS, PST, FUT, IMPF
    "mood",  # IND, SBJV, COND, IMP
    "aspect",  # PFV, IPFV, PRF, PROG
    "person",  # 1, 2, 3
    "number",  # SG, PL, DU
    "gender",  # MASC, FEM, NEUT
    "case",  # NOM, ACC, GEN, DAT, ABL, LOC, INS, VOC
    "formality",  # FORM, INFM
    "voice",  # ACT, PASS, MID
    "polarity",  # POS, NEG
    "definiteness",  # DEF, INDF, SPEC
    "finiteness",  # FIN, NFIN, INF, PTCP, CVB
    "animacy",  # ANIM, INAN
    "clitic1_surface",  # me, te, se, lo, la, le, etc.
    "clitic2_surface",  # second clitic surface (if present)
    "clitic_person",  # 1, 2, 3
    "clitic_number",  # SG, PL
    "clitic_gender",  # MASC, FEM
    "clitic_case",  # ACC, DAT, GEN
    "clitic_type",  # PRO, REFL
    "clitic_reflexive",  # REFL
    "degree",  # CMPR, SPRL
    "lgspec",  # LGSPEC1-10
]

MORPHOLOGICAL_GROUP_NO_VALUES = {group: f"no_{group}" for group in MORPHOLOGICAL_GROUPS}
MORPHOLOGICAL_GROUP_NA_VALUES = {group: f"NA_{group}" for group in MORPHOLOGICAL_GROUPS}


def _group_value_sort_key(value: str):
    if value.isdigit():
        return (0, int(value))
    match = re.match(r"^([A-Za-z_]+)(\\d+)$", value)
    if match:
        return (1, match.group(1), int(match.group(2)))
    return (2, value)


def _normalize_group_value(value: str) -> str:
    if value == "none" or "+" not in value:
        return value
    parts = [part for part in value.split("+") if part]
    unique = sorted(set(parts), key=_group_value_sort_key)
    return "+".join(unique)


# UniMorph tag to morphological group category mapping.
UNIMORPH_TAG_CATEGORY_MAP = {
    # Tense
    "PRS": "tense",
    "PST": "tense",
    "FUT": "tense",
    "IMPF": "tense",
    # Mood
    "IND": "mood",
    "SBJV": "mood",
    "COND": "mood",
    "IMP": "mood",
    "OPT": "mood",
    # Aspect
    "PFV": "aspect",
    "IPFV": "aspect",
    "PRF": "aspect",
    "PROG": "aspect",
    # Person
    "1": "person",
    "2": "person",
    "3": "person",
    "0": "person",
    # Number
    "SG": "number",
    "PL": "number",
    "DU": "number",
    "PAU": "number",
    # Gender
    "MASC": "gender",
    "FEM": "gender",
    "NEUT": "gender",
    # Case
    "NOM": "case",
    "ACC": "case",
    "GEN": "case",
    "DAT": "case",
    "ABL": "case",
    "LOC": "case",
    "INS": "case",
    "VOC": "case",
    "ESS": "case",
    "TRANS": "case",
    "COM": "case",
    # Formality
    "FORM": "formality",
    "INFM": "formality",
    # Voice
    "ACT": "voice",
    "PASS": "voice",
    "MID": "voice",
    "CAUS": "voice",
    # Polarity
    "POS": "polarity",
    "NEG": "polarity",
    # Definiteness
    "DEF": "definiteness",
    "INDF": "definiteness",
    "SPEC": "definiteness",
    # Finiteness
    "FIN": "finiteness",
    "NFIN": "finiteness",
    "INF": "finiteness",
    # Animacy
    "ANIM": "animacy",
    "INAN": "animacy",
    "HUM": "animacy",
    # Degree
    "CMPR": "degree",
    "SPRL": "degree",
    # Clitic marker (handled explicitly when surface tags are injected)
}

# Article capitalization transformations
NO_ARTICLE_CAPITALIZATION = "no_article_cap"
ADD_ARTICLE_CAPITALIZATION = "add_article_cap"
NA_ARTICLE_CAPITALIZATION = "NA_article_cap"

# Article/preposition space-prefix transformations (optional)
NO_ARTICLE_SPACE_PREFIX = "no_article_space_prefix"
ADD_ARTICLE_SPACE_PREFIX = "add_article_space_prefix"
NA_ARTICLE_SPACE_PREFIX = "NA_article_space_prefix"
NO_PREP_SPACE_PREFIX = "no_prep_space_prefix"
ADD_PREP_SPACE_PREFIX = "add_prep_space_prefix"
NA_PREP_SPACE_PREFIX = "NA_prep_space_prefix"

# Update NO_EMBEDDING_TYPES and NA_EMBEDDING_TYPES to include new groups
BASE_NO_EMBEDDING_TYPES = [
    NO_INFLECTION,
    NO_DERIVATION,
    NO_SPACE_PREFIX_TRANSFORM,
    NO_BASE_CAPITALIZATION_TRANSFORM,
    NO_PREFIX_PUNCTUATION,
    NO_SUFFIX_PUNCTUATION,
    NO_ARTICLE,
    NO_PREPOSITION,
    NO_PREP_CAPITALIZATION,
    NO_ARTICLE_CAPITALIZATION,
    NO_ARTICLE_SPACE_PREFIX,
    NO_PREP_SPACE_PREFIX,
]

BASE_NA_EMBEDDING_TYPES = [
    NA_INFLECTION,
    NA_DERIVATION,
    NA_SPACE_PREFIX_TRANSFORM,
    NA_BASE_CAPITALIZATION_TRANSFORM,
    NA_PREFIX_PUNCTUATION,
    NA_SUFFIX_PUNCTUATION,
    NA_ARTICLE,
    NA_PREPOSITION,
    NA_PREP_CAPITALIZATION,
    NA_ARTICLE_CAPITALIZATION,
    NA_ARTICLE_SPACE_PREFIX,
    NA_PREP_SPACE_PREFIX,
]
NO_EMBEDDING_TYPES = BASE_NO_EMBEDDING_TYPES + list(MORPHOLOGICAL_GROUP_NO_VALUES.values())
NA_EMBEDDING_TYPES = BASE_NA_EMBEDDING_TYPES + list(MORPHOLOGICAL_GROUP_NA_VALUES.values())
IGNORE_MULTI_TYPES_IN_INIT = []
FORBID_MULTI_TYPES_IN_INIT = [ADD_CAPITALIZATION_TRANSFORM, ADD_ALL_CAPS_CAPITALIZATION_TRANSFORM]

# Unified transformation groups for dual-stream tokenization
# This defines the canonical order of groups in the 2D modifier array
# Each position in the modifier array corresponds to one group
UNIFIED_TRANSFORM_GROUPS = [
    "space_prefix",  # NO_SPACE, ADD_SPACE
    "base_capitalization",  # NO_CAP, FIRST_CAP, ALL_CAPS (for base word)
    "inflection",  # NONE, GERUND, PAST, PLURAL, etc.
    "derivation",  # NONE, AGENT_ER, ABLE, etc.
    "articles",  # NONE, THE, A, AN
    "article_capitalization",  # NO_CAP, CAP
    "prepositions",  # NONE, BY, AT, OF, TO, IN, ON, WITH, FOR, FROM
    "prep_capitalization",  # NO_CAP, CAP
    "prefix_punctuation",  # NONE, OPEN_PAREN, OPEN_BRACKET, QUOTE, etc.
    "suffix_punctuation",  # NONE, PERIOD, COMMA, EXCLAIM, CLOSE_PAREN, APOSTROPHE_S, etc.
]

# Mapping from group names to their NO_* constants (index 0 in each group)
UNIFIED_GROUP_NO_VALUES = {
    "space_prefix": NO_SPACE_PREFIX_TRANSFORM,
    "base_capitalization": NO_BASE_CAPITALIZATION_TRANSFORM,
    "inflection": NO_INFLECTION,
    "derivation": NO_DERIVATION,
    "articles": NO_ARTICLE,
    "article_capitalization": NO_ARTICLE_CAPITALIZATION,
    "prepositions": NO_PREPOSITION,
    "prep_capitalization": NO_PREP_CAPITALIZATION,
    "article_space_prefix": NO_ARTICLE_SPACE_PREFIX,
    "prep_space_prefix": NO_PREP_SPACE_PREFIX,
    "prefix_punctuation": NO_PREFIX_PUNCTUATION,
    "suffix_punctuation": NO_SUFFIX_PUNCTUATION,
}

# Mapping from group names to their NA_* constants (for inapplicable cases)
UNIFIED_GROUP_NA_VALUES = {
    "space_prefix": NA_SPACE_PREFIX_TRANSFORM,
    "base_capitalization": NA_BASE_CAPITALIZATION_TRANSFORM,
    "inflection": NA_INFLECTION,
    "derivation": NA_DERIVATION,
    "articles": NA_ARTICLE,
    "article_capitalization": NA_ARTICLE_CAPITALIZATION,
    "prepositions": NA_PREPOSITION,
    "prep_capitalization": NA_PREP_CAPITALIZATION,
    "article_space_prefix": NA_ARTICLE_SPACE_PREFIX,
    "prep_space_prefix": NA_PREP_SPACE_PREFIX,
    "prefix_punctuation": NA_PREFIX_PUNCTUATION,
    "suffix_punctuation": NA_SUFFIX_PUNCTUATION,
}


class UnifiedModifierArray:
    """Manages the 2D modifier array for dual-stream tokenization.

    The modifier array has shape (seq_len, num_groups) where each position
    contains a group-relative index for that transformation group.

    Example:
        For a token with article "the" and past tense inflection:
        modifier_array[i] = [0, 0, 2, 0, 1, 0, 0, 0, 0, 0]
    """

    def __init__(
        self,
        groups: Optional[List[str]] = None,
        types_loss_indices_map: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        """Initialize the unified modifier array manager.

        Args:
            groups: List of transformation group names to use.
                   Defaults to UNIFIED_TRANSFORM_GROUPS.
            types_loss_indices_map: Mapping from group names to (start_idx, end_idx)
                                   in the global transformation space.
        """
        self.groups = groups or UNIFIED_TRANSFORM_GROUPS.copy()
        self.num_groups = len(self.groups)
        self.group_to_idx = {name: i for i, name in enumerate(self.groups)}

        # Global transformation indices mapping
        self.types_loss_indices_map = types_loss_indices_map or {}

        # Group sizes (number of possible values per group)
        self.group_sizes = {}
        for group in self.groups:
            if group in self.types_loss_indices_map:
                start, end = self.types_loss_indices_map[group]
                self.group_sizes[group] = end - start
            else:
                self.group_sizes[group] = 1  # Unknown group, assume single value

    def create_empty_modifier(self) -> List[int]:
        """Create an empty modifier tuple (all zeros = no modifications)."""
        return [0] * self.num_groups

    def max_group_relative_index(self) -> int:
        """Maximum group-relative value across all active transformation groups."""
        max_idx = 0
        for group in self.groups:
            group_size = int(self.group_sizes.get(group, 1))
            if group_size > 0:
                max_idx = max(max_idx, group_size - 1)
        return max_idx

    def recommended_modifier_dtype_name(self) -> str:
        """Smallest unsigned dtype name that can represent all modifier values."""
        max_idx = self.max_group_relative_index()
        if max_idx <= 0xFF:
            return "uint8"
        if max_idx <= 0xFFFF:
            return "uint16"
        if max_idx <= 0xFFFFFFFF:
            return "uint32"
        return "uint64"

    def set_group_value(self, modifier: List[int], group_name: str, value: int) -> List[int]:
        """Set the value for a specific group in a modifier tuple.

        Args:
            modifier: The modifier tuple to modify (list of group-relative indices).
            group_name: Name of the group to set.
            value: Group-relative index for that group.

        Returns:
            Modified modifier tuple.
        """
        if group_name in self.group_to_idx:
            modifier[self.group_to_idx[group_name]] = value
        return modifier

    def get_group_value(self, modifier: List[int], group_name: str) -> int:
        """Get the value for a specific group from a modifier tuple."""
        if group_name in self.group_to_idx:
            return modifier[self.group_to_idx[group_name]]
        return 0

    def global_to_group_relative(self, group_name: str, global_idx: int) -> int:
        """Convert a global transformation index to a group-relative index.

        Args:
            group_name: Name of the transformation group.
            global_idx: Index in the global transformation space.

        Returns:
            Group-relative index (0, 1, 2, ... within the group).
        """
        if group_name not in self.types_loss_indices_map:
            return 0
        start, _ = self.types_loss_indices_map[group_name]
        return global_idx - start

    def group_relative_to_global(self, group_name: str, relative_idx: int) -> int:
        """Convert a group-relative index to a global transformation index.

        Args:
            group_name: Name of the transformation group.
            relative_idx: Index within the group (0, 1, 2, ...).

        Returns:
            Global transformation index.
        """
        if group_name not in self.types_loss_indices_map:
            return 0
        start, _ = self.types_loss_indices_map[group_name]
        return start + relative_idx

    def to_one_hot(self, modifier: List[int], total_transform_dim: int) -> List[float]:
        """Convert group-relative modifier to one-hot representation.

        Args:
            modifier: List of group-relative indices.
            total_transform_dim: Total size of the one-hot vector.

        Returns:
            One-hot encoded transformation vector.
        """
        one_hot = [0.0] * total_transform_dim
        for group_idx, group_name in enumerate(self.groups):
            if group_name in self.types_loss_indices_map:
                global_idx = self.group_relative_to_global(group_name, modifier[group_idx])
                if global_idx < total_transform_dim:
                    one_hot[global_idx] = 1.0
        return one_hot

    def from_one_hot(self, one_hot: List[float]) -> List[int]:
        """Convert one-hot representation back to group-relative modifier.

        Args:
            one_hot: One-hot encoded transformation vector.

        Returns:
            List of group-relative indices.
        """
        modifier = self.create_empty_modifier()
        for group_idx, group_name in enumerate(self.groups):
            if group_name in self.types_loss_indices_map:
                start, end = self.types_loss_indices_map[group_name]
                # Find which index in this group is set
                for i in range(start, end):
                    if i < len(one_hot) and one_hot[i] > 0.5:
                        modifier[group_idx] = i - start
                        break
        return modifier

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            "groups": self.groups,
            "types_loss_indices_map": self.types_loss_indices_map,
            "group_sizes": self.group_sizes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnifiedModifierArray":
        """Deserialize from dictionary."""
        instance = cls(
            groups=data.get("groups", UNIFIED_TRANSFORM_GROUPS),
            types_loss_indices_map=data.get("types_loss_indices_map", {}),
        )
        if "group_sizes" in data:
            instance.group_sizes = data["group_sizes"]
        return instance

    def __repr__(self):
        return f"UnifiedModifierArray(groups={self.groups}, sizes={self.group_sizes})"


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


def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    d = {k: v for k, v in d.items() if v}
    for key, value in d.items():
        if isinstance(value, defaultdict):
            d[key] = defaultdict_to_dict(value)
    return d


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
        self.invalid_derivation_rows_count = 0
        self.invalid_derivation_rows_samples = []

        # Other parameters
        self.no_diacritics = no_diacritics
        self.filtered_possessive_inflections = []

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
                    lemma_lower = lemma.lower()
                    form_lower = form.lower()
                    if (lemma_lower, form_lower) in POSSESSIVE_PRONOUN_SKIP:
                        self.filtered_possessive_inflections.append((lemma, form, tags))
                        continue
                    if tags in ["ADJ", "N;SG", "V;NFIN;IMP+SBJV"]:
                        # Keep lemma presence for base detection, but skip adding inflection tags.
                        if form == lemma:
                            self.lemma_to_forms[lemma][form].add(("LEMMA_ONLY",))
                        continue
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

        self.lemma_to_forms = defaultdict_to_dict(self.lemma_to_forms)
        self.form_to_lemmas = defaultdict_to_dict(self.form_to_lemmas)
        self.tag_to_subtypes = defaultdict_to_dict(self.tag_to_subtypes)

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
                        root, derived, pos, deriv_type = (
                            parts[0].strip(),
                            parts[1].strip(),
                            parts[2].strip(),
                            parts[3].strip(),
                        )
                        if (not root) or (not derived):
                            self.invalid_derivation_rows_count += 1
                            if len(self.invalid_derivation_rows_samples) < 10:
                                self.invalid_derivation_rows_samples.append(line)
                            continue
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

        self.derivation_to_words = defaultdict_to_dict(self.derivation_to_words)
        self.root_to_derivations = defaultdict_to_dict(self.root_to_derivations)
        self.derived_to_root = defaultdict_to_dict(self.derived_to_root)
        self.derivation_chains = defaultdict_to_dict(self.derivation_chains)
        if self.invalid_derivation_rows_count:
            print(
                "Filtered "
                f"{self.invalid_derivation_rows_count} invalid derivation rows "
                "(empty root/derived)"
            )

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

    def set_common_derivation_types(self, min_count=0):
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
        is_lemma = (word in self.lemma_to_forms) or (word.lower() in self.lemma_to_forms)
        is_form = (word in self.form_to_lemmas) or (word.lower() in self.form_to_lemmas)
        is_derived = (word in self.derived_to_root) or (word.lower() in self.derived_to_root)
        is_root = (word in self.root_to_derivations) or (word.lower() in self.root_to_derivations)
        return is_lemma or is_form or is_derived or is_root

    def is_base_form(self, word):
        """Check if a word is a base form in UniMorph."""
        is_lemma = word in self.lemma_to_forms
        is_form = word in self.form_to_lemmas
        return is_lemma  # or (not (is_lemma or is_form))

    def is_lemma(self, word):
        """Check if a word is a lemma in UniMorph."""
        is_lemma = word in self.lemma_to_forms
        return is_lemma


class ModifierMap:
    """Maps modifier_id to per-group transformation indices for new transformation groups.

    This class manages the compression of new transformation groups (articles, prepositions,
    punctuation) into a single modifier ID. Each modifier ID maps to a tuple of transformation
    indices, one per group.

    modifier_id = 0 is reserved for "no modifiers" (all groups set to their NA/NO values).
    IDs are assigned lazily during tokenization based on observed combinations.

    Example:
        modifier_id 1 → (ARTICLE_THE_idx, NO_PREP_idx, NO_PREFIX_PUNCT_idx, NO_SUFFIX_PUNCT_idx, ...)
        modifier_id 2 → (NO_ARTICLE_idx, PREP_BY_idx, NO_PREFIX_PUNCT_idx, NO_SUFFIX_PUNCT_idx, ...)
    """

    def __init__(self, type_groups: Optional[List[str]] = None):
        """Initialize the modifier map.

        Args:
            type_groups: List of transformation group names that use modifier IDs.
                        Typically: ['articles', 'prepositions', 'article_capitalization',
                                   'prep_capitalization', 'prefix_punctuation', 'suffix_punctuation']
        """
        self.type_groups = type_groups or []
        self.modifier_to_transforms = {}  # {modifier_id: (idx1, idx2, ...)}
        self.transforms_to_modifier = {}  # {(idx1, idx2, ...): modifier_id}
        self.next_id = 1  # 0 reserved for "no modifiers"

    def get_modifier_id(self, transform_tuple: Tuple[int, ...]) -> int:
        """Get or create modifier ID for a transformation combination.

        Args:
            transform_tuple: Tuple of transformation indices, one per group.
                           Order must match self.type_groups.

        Returns:
            modifier_id: Integer ID representing this combination (0 = no modifiers, >0 = has modifiers)
        """
        # Check if this is the "no modifiers" case (all zeros or empty)
        if not transform_tuple or all(idx == 0 for idx in transform_tuple):
            return 0

        # Check if we've seen this combination before
        if transform_tuple in self.transforms_to_modifier:
            return self.transforms_to_modifier[transform_tuple]

        # Create new modifier ID
        new_id = self.next_id
        self.next_id += 1
        self.modifier_to_transforms[new_id] = transform_tuple
        self.transforms_to_modifier[transform_tuple] = new_id
        return new_id

    def get_transforms(self, modifier_id: int) -> Optional[Tuple[int, ...]]:
        """Lookup transformation indices for a modifier ID.

        Args:
            modifier_id: The modifier ID to look up.

        Returns:
            Tuple of transformation indices, or None if modifier_id = 0 (no modifiers).
        """
        if modifier_id == 0:
            return None  # No modifiers
        return self.modifier_to_transforms.get(modifier_id)

    def __len__(self):
        """Return the number of unique modifiers (excluding 0)."""
        return len(self.modifier_to_transforms)

    def __contains__(self, modifier_id: int):
        """Check if a modifier ID exists."""
        return modifier_id == 0 or modifier_id in self.modifier_to_transforms

    def to_dict(self) -> dict:
        """Serialize to dictionary for bundle storage."""
        return {
            "type_groups": self.type_groups,
            "modifier_to_transforms": {str(k): v for k, v in self.modifier_to_transforms.items()},
            "transforms_to_modifier": {str(k): v for k, v in self.transforms_to_modifier.items()},
            "next_id": self.next_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModifierMap":
        """Deserialize from dictionary."""
        modifier_map = cls(type_groups=data.get("type_groups", []))

        # Restore modifier_to_transforms with int keys and tuple values
        modifier_map.modifier_to_transforms = {
            int(k): tuple(v) for k, v in data.get("modifier_to_transforms", {}).items()
        }

        # Restore transforms_to_modifier with tuple keys and int values
        modifier_map.transforms_to_modifier = {
            tuple(eval(k)): int(v) for k, v in data.get("transforms_to_modifier", {}).items()
        }

        modifier_map.next_id = data.get("next_id", 1)
        return modifier_map

    def __repr__(self):
        return (
            f"ModifierMap(groups={self.type_groups}, "
            f"num_modifiers={len(self)}, next_id={self.next_id})"
        )


def get_pos_from_wordnet(word):
    """Get possible parts of speech for a word from WordNet."""
    word = (word or "").strip()
    if not word:
        return set()

    pos_map = {wn.NOUN: "NOUN", wn.VERB: "VERB", wn.ADJ: "ADJ", wn.ADV: "ADV"}
    possible_pos = set()
    for synset in _wordnet_synsets(word):
        pos = synset.pos()
        if pos in pos_map:
            possible_pos.add(pos_map[pos])
    # Add spacy POS
    try:
        doc = _get_spacy_nlp()(word)
        if len(doc) > 0 and doc[0].pos_:
            possible_pos.add(doc[0].pos_)
    except Exception:
        pass
    return possible_pos


def _get_inflection_from_pyinflect(base_form, pos_type):
    """Get an inflection using PyInflect."""
    inflection = base_form._.inflect(pos_type)
    if inflection is None:
        return None
    if _get_english_dictionary().check(inflection):
        synsets = _wordnet_synsets(inflection)
        if any(base_form.text in [lemma.name() for lemma in syn.lemmas()] for syn in synsets):
            return inflection
    return None


def get_pyinflect_variations(word, pos_set):
    """Get inflections using PyInflect."""
    word = (word or "").strip()
    if not word:
        return {}

    result = {}
    try:
        doc = _get_spacy_nlp()(word)
    except Exception:
        return result
    if len(doc) == 0:
        return result
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


def is_english_alphabet(word):
    return bool(re.fullmatch(r"[a-zA-Z]+", word))


def _has_whitespace(word: str) -> bool:
    return bool(re.search(r"\s", word))


def _tag_set_has(tag_set, required_tags: List[str]) -> bool:
    return all(tag in tag_set for tag in required_tags)


def is_english_word(word, unimorph=None):
    """Check if a word is valid English and contains only letters."""
    result = is_english_alphabet(word)
    if unimorph is not None:
        result = result and (
            _get_english_dictionary().check(word) or unimorph.is_unimorph_word(word)
        )
    else:
        result = result and _get_english_dictionary().check(word)
    return result


def extract_spanish_clitic_sequence(word):
    """Return ordered Spanish enclitic sequence if present, else None."""
    if not word:
        return None
    lower = word.lower()
    clitics = []
    # Prefer longer matches first
    for _ in range(2):  # Spanish allows up to two clitics
        matched = None
        for clitic in sorted(SPANISH_CLITICS, key=len, reverse=True):
            if lower.endswith(clitic):
                matched = clitic
                break
        if matched is None:
            break
        clitics.append(matched)
        lower = lower[: -len(matched)]
        if len(lower) < 2:
            break
    if not clitics:
        return None
    return list(reversed(clitics))


SPANISH_CLITIC_FEATURES = {
    "lo": {"person": "3", "number": "SG", "gender": "MASC", "case": "ACC"},
    "la": {"person": "3", "number": "SG", "gender": "FEM", "case": "ACC"},
    "los": {"person": "3", "number": "PL", "gender": "MASC", "case": "ACC"},
    "las": {"person": "3", "number": "PL", "gender": "FEM", "case": "ACC"},
    "le": {"person": "3", "number": "SG", "case": "DAT"},
    "les": {"person": "3", "number": "PL", "case": "DAT"},
    "me": {"person": "1", "number": "SG"},
    "te": {"person": "2", "number": "SG"},
    "nos": {"person": "1", "number": "PL"},
    "os": {"person": "2", "number": "PL"},
    "se": {"reflexive": True},
}


def clitic_sequence_to_feature_tags(clitic_seq):
    """Convert a clitic sequence into feature tags for compositional morphology."""
    if not clitic_seq:
        return []
    tags = []
    for clitic in clitic_seq:
        info = SPANISH_CLITIC_FEATURES.get(clitic)
        if not info:
            continue
        if "person" in info:
            tags.append(f"CLITIC_PERSON_{info['person']}")
        if "number" in info:
            tags.append(f"CLITIC_NUMBER_{info['number']}")
        if "gender" in info:
            tags.append(f"CLITIC_GENDER_{info['gender']}")
        if "case" in info:
            tags.append(f"CLITIC_CASE_{info['case']}")
        if info.get("reflexive"):
            tags.append("CLITIC_REFLEXIVE")
    return list(OrderedSet(tags))


def clitic_sequence_to_surface_tags(clitic_seq, clitic_mode="surface2"):
    """Convert a clitic sequence into ordered surface tags."""
    if not clitic_seq:
        return []
    if clitic_mode not in {"surface2", "surface1"}:
        raise ValueError(f"Unsupported clitic_mode: {clitic_mode}")
    tags = []
    if len(clitic_seq) >= 1:
        tags.append(f"CLITIC1_SURFACE_{clitic_seq[0]}")
    if clitic_mode == "surface2" and len(clitic_seq) >= 2:
        tags.append(f"CLITIC2_SURFACE_{clitic_seq[1]}")
    return tags


def build_complete_transforms(
    base_transforms,
    space_transform,
    cap_transform,
    prefix_punct=None,
    suffix_punct=None,
    article=None,
    preposition=None,
    article_cap=None,
    prep_cap=None,
    article_space_prefix=None,
    prep_space_prefix=None,
    morphology_mode="atomic",
):
    """Ensure transforms contain exactly one value from each transformation group."""
    complete_transforms = [t for t in base_transforms]

    morph_prefixes = tuple(f"{group}_" for group in MORPHOLOGICAL_GROUPS)

    # Ensure every word gets exactly one value from each transformation group
    # Check for inflection: either atomic (inflect_*) or compositional (morphological groups)
    has_inflection = any(
        t.startswith("inflect_")
        or t == NO_INFLECTION
        or t == NA_INFLECTION
        or t.startswith(morph_prefixes)
        or t in MORPHOLOGICAL_GROUP_NO_VALUES.values()
        or t in MORPHOLOGICAL_GROUP_NA_VALUES.values()
        for t in complete_transforms
    )
    has_derivation = any(
        t.startswith("deriv_") or t == NO_DERIVATION or t == NA_DERIVATION
        for t in complete_transforms
    )
    has_prefix_punct = any(
        t.startswith("punct_prefix_") or t == NO_PREFIX_PUNCTUATION or t == NA_PREFIX_PUNCTUATION
        for t in complete_transforms
    )
    has_suffix_punct = any(
        t.startswith("punct_suffix_") or t == NO_SUFFIX_PUNCTUATION or t == NA_SUFFIX_PUNCTUATION
        for t in complete_transforms
    )
    has_article = any(
        t.startswith("article_") or t == NO_ARTICLE or t == NA_ARTICLE for t in complete_transforms
    )
    has_prep = any(
        t.startswith("prep_") or t == NO_PREPOSITION or t == NA_PREPOSITION
        for t in complete_transforms
    )
    has_article_cap = any(
        t in [NO_ARTICLE_CAPITALIZATION, ADD_ARTICLE_CAPITALIZATION, NA_ARTICLE_CAPITALIZATION]
        for t in complete_transforms
    )
    has_prep_cap = any(
        t in [NO_PREP_CAPITALIZATION, ADD_PREP_CAPITALIZATION, NA_PREP_CAPITALIZATION]
        for t in complete_transforms
    )
    has_article_space = any(
        t in [NO_ARTICLE_SPACE_PREFIX, ADD_ARTICLE_SPACE_PREFIX, NA_ARTICLE_SPACE_PREFIX]
        for t in complete_transforms
    )
    has_prep_space = any(
        t in [NO_PREP_SPACE_PREFIX, ADD_PREP_SPACE_PREFIX, NA_PREP_SPACE_PREFIX]
        for t in complete_transforms
    )

    if morphology_mode == "atomic" and not has_inflection:
        complete_transforms.append(NO_INFLECTION)
    if not has_derivation:
        complete_transforms.append(NO_DERIVATION)

    # Add space and capitalization transforms
    complete_transforms.append(space_transform)
    complete_transforms.append(cap_transform)

    # Add new transformation groups if provided
    if not has_prefix_punct:
        complete_transforms.append(
            prefix_punct if prefix_punct is not None else NO_PREFIX_PUNCTUATION
        )
    if not has_suffix_punct:
        complete_transforms.append(
            suffix_punct if suffix_punct is not None else NO_SUFFIX_PUNCTUATION
        )
    if not has_article:
        complete_transforms.append(article if article is not None else NO_ARTICLE)
    if not has_prep:
        complete_transforms.append(preposition if preposition is not None else NO_PREPOSITION)
    if not has_article_cap:
        complete_transforms.append(
            article_cap if article_cap is not None else NO_ARTICLE_CAPITALIZATION
        )
    if not has_prep_cap:
        complete_transforms.append(prep_cap if prep_cap is not None else NO_PREP_CAPITALIZATION)
    if article_space_prefix is not None and not has_article_space:
        complete_transforms.append(article_space_prefix)
    if prep_space_prefix is not None and not has_prep_space:
        complete_transforms.append(prep_space_prefix)

    return complete_transforms


def get_variation_transformations(
    word,
    base_form=None,
    transforms=None,
    space_prefix=" ",
    decompose_spaces=True,
    decompose_capitalization=True,
    morphology_mode="atomic",
    use_relative_space_cap_transforms=False,
    article_space_prefix=None,
    prep_space_prefix=None,
):
    """
    Get variations of a word with transformation tags.

    Args:
        word: The word to transform
        base_form: The original base form for comparison
        space_prefix: The space prefix character
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation
        use_relative_space_cap_transforms: Whether to model space/cap changes as add/remove

    Returns:
        Dict mapping word variations to their transformation tags
    """
    result = {}
    has_space_prefix = word.startswith(space_prefix)
    word_without_prefix = word[len(space_prefix) :] if has_space_prefix else word

    def _cap_state(text):
        if not text:
            return "none"
        alpha_chars = [c for c in text if c.isalpha()]
        if alpha_chars and all(c.isupper() for c in alpha_chars):
            return "all_caps"
        if text[0].isupper():
            return "first_cap"
        return "none"

    def _apply_cap_state(text, state):
        if state == "all_caps":
            return text.upper()
        if state == "first_cap":
            return text.capitalize()
        return text.lower()

    def _relative_space_transform(base_has_space, var_has_space):
        if var_has_space == base_has_space:
            return NO_SPACE_PREFIX_TRANSFORM
        return WITH_SPACE_PREFIX_TRANSFORM if var_has_space else REMOVE_SPACE_PREFIX_TRANSFORM

    def _relative_cap_transform(base_cap_state, var_cap_state):
        if var_cap_state == base_cap_state:
            return NO_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "all_caps":
            return ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "first_cap":
            return (
                ADD_BASE_CAPITALIZATION_TRANSFORM
                if base_cap_state == "none"
                else REMOVE_BASE_CAPITALIZATION_TRANSFORM
            )
        return REMOVE_BASE_CAPITALIZATION_TRANSFORM

    def _absolute_cap_transform(var_cap_state):
        if var_cap_state == "all_caps":
            return ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "first_cap":
            return ADD_BASE_CAPITALIZATION_TRANSFORM
        return NO_BASE_CAPITALIZATION_TRANSFORM

    base_has_space_prefix = has_space_prefix
    current_cap_state = _cap_state(word_without_prefix)
    base_cap_state = current_cap_state

    # Set transformation types
    if base_form:
        base_has_space_prefix = base_form.startswith(space_prefix)
        base_word_without_prefix = (
            base_form[len(space_prefix) :] if base_has_space_prefix else base_form
        )
        base_cap_state = _cap_state(base_word_without_prefix)

    if use_relative_space_cap_transforms:
        space_transform = _relative_space_transform(base_has_space_prefix, has_space_prefix)
        cap_transform = _relative_cap_transform(base_cap_state, current_cap_state)
    else:
        # Base-form-independent absolute capitalization transform.
        space_transform = (
            WITH_SPACE_PREFIX_TRANSFORM if has_space_prefix else NO_SPACE_PREFIX_TRANSFORM
        )
        cap_transform = _absolute_cap_transform(current_cap_state)
    # Add current word with transformations
    transforms = [] if transforms is None else list(transforms)
    # If it's the base form without transformations, add default transforms
    if not transforms:
        if morphology_mode == "atomic":
            transforms.extend([NO_INFLECTION, NO_DERIVATION])
        else:
            transforms.extend([NO_DERIVATION])

    base_transforms = [t for t in transforms]
    complete_transforms = build_complete_transforms(
        base_transforms,
        space_transform,
        cap_transform,
        prefix_punct=None,
        suffix_punct=None,
        article=None,
        preposition=None,
        article_cap=None,
        prep_cap=None,
        article_space_prefix=article_space_prefix,
        prep_space_prefix=prep_space_prefix,
        morphology_mode=morphology_mode,
    )
    result[word] = complete_transforms

    # Generate additional space/capitalization variants.
    space_options = {has_space_prefix}
    if decompose_spaces:
        space_options.add(not has_space_prefix)
    cap_options = {current_cap_state}
    if decompose_capitalization:
        cap_options.update({"none", "first_cap", "all_caps"})

    normalized_core = (
        word_without_prefix.lower() if decompose_capitalization else word_without_prefix
    )
    for var_has_space in sorted(space_options):
        for var_cap_state in sorted(cap_options):
            variant_core = normalized_core
            if decompose_capitalization:
                variant_core = _apply_cap_state(normalized_core, var_cap_state)
            variant_str = (space_prefix if var_has_space else "") + variant_core
            if variant_str == word:
                continue
            if use_relative_space_cap_transforms:
                var_space_transform = _relative_space_transform(
                    base_has_space_prefix, var_has_space
                )
                var_cap_transform = _relative_cap_transform(base_cap_state, var_cap_state)
            else:
                var_space_transform = (
                    WITH_SPACE_PREFIX_TRANSFORM if var_has_space else NO_SPACE_PREFIX_TRANSFORM
                )
                var_cap_transform = _absolute_cap_transform(var_cap_state)
            result[variant_str] = build_complete_transforms(
                transforms,
                var_space_transform,
                var_cap_transform,
                prefix_punct=None,
                suffix_punct=None,
                article=None,
                preposition=None,
                article_cap=None,
                prep_cap=None,
                article_space_prefix=article_space_prefix,
                prep_space_prefix=prep_space_prefix,
                morphology_mode=morphology_mode,
            )

    return result


def get_chained_derivations(
    derived_form,
    root_token,
    current_deriv_chain,
    unimorph,
    space_prefix=" ",
    decompose_spaces=True,
    decompose_capitalization=True,
    should_preserve_caps=False,
    original_token_repr=None,
    token_has_space_prefix=True,
    max_depth=2,
    use_english_resources=True,
    language="en",
    morphology_mode="atomic",
    clitic_mode="surface2",
    use_relative_space_cap_transforms=False,
    article_space_prefix=None,
    prep_space_prefix=None,
):
    """
    Recursively find derivations of derived forms to create chained derivations.

    Args:
        derived_form: The derived form to check for further derivations
        root_token: The ultimate root token
        current_deriv_chain: List of derivation types so far (e.g., ["deriv_ist"])
        unimorph: UniMorph analyzer instance
        max_depth: Maximum chain length to prevent infinite recursion

    Returns:
        List of (final_form, derivation_chain, form_variations) tuples
    """
    if len(current_deriv_chain) >= max_depth:
        return []

    results = []

    # Check for further derivations of this derived form
    further_derivations = unimorph.get_root_derivations(derived_form)

    for further_derived, type_pos_sets in further_derivations.items():
        # Apply capitalization pattern if needed
        if should_preserve_caps and original_token_repr:
            further_derived = apply_capitalization_pattern(original_token_repr, further_derived)

        for deriv_type, pos in type_pos_sets:
            # Create new derivation chain
            new_deriv_chain = current_deriv_chain + [f"deriv_{deriv_type}"]

            # Add this level of derivation
            further_derived_with_prefix = (
                space_prefix + further_derived if token_has_space_prefix else further_derived
            )

            # Get variations for this chained derivation
            chained_variations = get_variation_transformations(
                further_derived_with_prefix,
                root_token,
                new_deriv_chain,
                space_prefix,
                decompose_spaces,
                decompose_capitalization,
                morphology_mode=morphology_mode,
                use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                article_space_prefix=article_space_prefix,
                prep_space_prefix=prep_space_prefix,
            )

            results.append((further_derived, new_deriv_chain, chained_variations))

            # Recursively check for even deeper derivations
            if len(new_deriv_chain) < max_depth:
                deeper_results = get_chained_derivations(
                    further_derived,
                    root_token,
                    new_deriv_chain,
                    unimorph,
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                    should_preserve_caps,
                    original_token_repr,
                    token_has_space_prefix,
                    max_depth,
                    use_english_resources=use_english_resources,
                    language=language,
                    morphology_mode=morphology_mode,
                    clitic_mode=clitic_mode,
                    use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                    article_space_prefix=article_space_prefix,
                    prep_space_prefix=prep_space_prefix,
                )
                results.extend(deeper_results)

            # Also check for inflections of this chained derived form
            derived_inflections = unimorph.get_inflections(further_derived)
            for inflected_chained_form, derived_tag_sets in derived_inflections.items():
                if inflected_chained_form == further_derived:
                    continue

                # Apply capitalization pattern if needed
                if should_preserve_caps and original_token_repr:
                    inflected_chained_form = apply_capitalization_pattern(
                        original_token_repr, inflected_chained_form
                    )

                # Process tag sets for chained derived form inflections
                for derived_tag_set in derived_tag_sets:
                    derived_tag_str = ";".join(derived_tag_set)

                    if morphology_mode == "atomic":
                        combined_transforms = new_deriv_chain + [f"inflect_{derived_tag_str}"]
                    else:
                        groups = map_unimorph_tags_to_compositional_groups(
                            derived_tag_set, language, clitic_mode=clitic_mode
                        )
                        morph_transforms = [
                            f"{group}_{value}" for group, value in groups.items() if value != "none"
                        ]
                        combined_transforms = new_deriv_chain + morph_transforms

                    # Add inflection of chained derivation with proper space prefix
                    inflected_chained_with_prefix = (
                        space_prefix + inflected_chained_form
                        if token_has_space_prefix
                        else inflected_chained_form
                    )

                    # Get variations of the inflected chained derived form
                    inflected_chained_variations = get_variation_transformations(
                        inflected_chained_with_prefix,
                        root_token,
                        combined_transforms,
                        space_prefix,
                        decompose_spaces,
                        decompose_capitalization,
                        morphology_mode=morphology_mode,
                        use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                        article_space_prefix=article_space_prefix,
                        prep_space_prefix=prep_space_prefix,
                    )

                    results.append(
                        (inflected_chained_form, combined_transforms, inflected_chained_variations)
                    )

            # Also check PyInflect for inflections of chained derived form
            if use_english_resources:
                chained_pos_set = get_pos_from_wordnet(further_derived)
                pyinflect_chained_variations = get_pyinflect_variations(
                    further_derived, chained_pos_set
                )

                for inflected_chained_form, derived_tag_str in pyinflect_chained_variations.items():
                    # Apply capitalization pattern if needed
                    if should_preserve_caps and original_token_repr:
                        inflected_chained_form = apply_capitalization_pattern(
                            original_token_repr, inflected_chained_form
                        )

                    # Check if it's a valid English word
                    if not is_english_word(inflected_chained_form):
                        continue

                    if morphology_mode == "atomic":
                        combined_transforms = new_deriv_chain + [f"inflect_{derived_tag_str}"]
                    else:
                        tag_sets = _split_tag_str(derived_tag_str)
                        group_mappings = [
                            map_unimorph_tags_to_compositional_groups(
                                tag_set, language, clitic_mode=clitic_mode
                            )
                            for tag_set in tag_sets
                        ]
                        if group_mappings:
                            merged_groups = merge_compositional_groups(group_mappings)
                            morph_transforms = [
                                f"{group}_{value}"
                                for group, value in merged_groups.items()
                                if value != "none"
                            ]
                        else:
                            morph_transforms = []
                        combined_transforms = new_deriv_chain + morph_transforms

                    # Add inflection of chained derivation with proper space prefix
                    inflected_chained_with_prefix = (
                        space_prefix + inflected_chained_form
                        if token_has_space_prefix
                        else inflected_chained_form
                    )

                    # Get variations of the inflected chained derived form
                    inflected_chained_variations = get_variation_transformations(
                        inflected_chained_with_prefix,
                        root_token,
                        combined_transforms,
                        space_prefix,
                        decompose_spaces,
                        decompose_capitalization,
                        morphology_mode=morphology_mode,
                        use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                        article_space_prefix=article_space_prefix,
                        prep_space_prefix=prep_space_prefix,
                    )

                    results.append(
                        (inflected_chained_form, combined_transforms, inflected_chained_variations)
                    )

    return results


def apply_capitalization_pattern(original, target):
    """Apply capitalization pattern from original to target string."""
    if original and len(original) > 0 and original[0].isupper() and len(target) > 0:
        return target[0].upper() + target[1:]
    return target


def handle_derived_form_conflict(
    derived_form,
    decomposition_map,
    root_token,
    deriv_type,
    base_tokens,
    token_has_space_prefix,
    space_prefix,
    decompose_spaces,
    decompose_capitalization,
    morphology_mode="atomic",
    use_relative_space_cap_transforms=False,
    article_space_prefix=None,
    prep_space_prefix=None,
    conflicted_base_token=None,
):
    """Handle case where derived form already exists as base form - convert it to derivation of root."""
    print(
        f"Handling conflict: {derived_form} exists as base form but is derivation of {root_token}"
    )

    # Find and remove the conflicted base form
    if conflicted_base_token is None:
        for base_token in list(decomposition_map.keys()):
            if base_token.lower().strip(space_prefix) == derived_form.lower():
                conflicted_base_token = base_token
                break

    if conflicted_base_token:
        # Save the inflections of the conflicted base form
        saved_inflections = decomposition_map[conflicted_base_token].copy()

        # Remove from base tokens and decomposition map
        if conflicted_base_token in base_tokens:
            del base_tokens[conflicted_base_token]
        del decomposition_map[conflicted_base_token]

        # Add the derived form to the root token's decomposition
        if root_token not in decomposition_map:
            decomposition_map[root_token] = {}

        # Add derivation with proper space prefix
        derived_with_prefix = (
            space_prefix + derived_form if token_has_space_prefix else derived_form
        )

        # Get variations of the derived form
        derived_variations = get_variation_transformations(
            derived_with_prefix,
            root_token,
            [deriv_type],
            space_prefix,
            decompose_spaces,
            decompose_capitalization,
            morphology_mode=morphology_mode,
            use_relative_space_cap_transforms=use_relative_space_cap_transforms,
            article_space_prefix=article_space_prefix,
            prep_space_prefix=prep_space_prefix,
        )

        # Add the derivation variations
        for var, transforms in derived_variations.items():
            if var not in decomposition_map[root_token]:
                decomposition_map[root_token][var] = transforms

        # Convert the saved inflections to be inflections of the derivation
        for inflection_form, inflection_transforms in saved_inflections.items():
            # Skip the base form itself (it's already added as derivation)
            if inflection_form == conflicted_base_token:
                continue

            # Combine derivation type with inflection types
            combined_transforms = [deriv_type] + [
                t for t in inflection_transforms if t != NO_INFLECTION
            ]
            if NO_INFLECTION in inflection_transforms:
                # Replace NO_INFLECTION with actual inflection type
                combined_transforms = [t for t in combined_transforms if t != NO_INFLECTION]

            # Add to root token's decomposition
            if inflection_form not in decomposition_map[root_token]:
                decomposition_map[root_token][inflection_form] = combined_transforms


# ==============================================================================
# Punctuation, Article, and Preposition Detection Functions
# ==============================================================================


def detect_prefix_punctuation(text):
    """
    Detect prefix punctuation combinations at the start of text.

    Returns:
        tuple: (punctuation_type_constant, remaining_text) or (None, text) if no match
    """
    if not text:
        return None, text

    # Define prefix patterns in order of priority (longest first)
    # Format: (pattern, constant)
    prefix_patterns = [
        # Two-character combinations
        ("'(", PREFIX_PUNCT_SINGLE_PAREN),
        ('"(', PREFIX_PUNCT_DOUBLE_PAREN),
        ("'[", PREFIX_PUNCT_SINGLE_SQUARE),
        ('"[', PREFIX_PUNCT_DOUBLE_SQUARE),
        ("('", PREFIX_PUNCT_PAREN_SINGLE),
        ("-(", PREFIX_PUNCT_HYPHEN_PAREN),
        ("-[", PREFIX_PUNCT_HYPHEN_SQUARE),
        # Single characters
        ("'", PREFIX_PUNCT_SINGLE_QUOTE),
        ("\u2018", PREFIX_PUNCT_SINGLE_QUOTE),
        ("\u2019", PREFIX_PUNCT_SINGLE_QUOTE),
        ('"', PREFIX_PUNCT_DOUBLE_QUOTE),
        ("\u201c", PREFIX_PUNCT_DOUBLE_QUOTE),
        ("\u201d", PREFIX_PUNCT_DOUBLE_QUOTE),
        ("`", PREFIX_PUNCT_BACKTICK),
        ("(", PREFIX_PUNCT_PAREN),
        ("[", PREFIX_PUNCT_SQUARE),
        ("{", PREFIX_PUNCT_CURLY),
        ("-", PREFIX_PUNCT_HYPHEN),
    ]

    for pattern, const in prefix_patterns:
        if text.startswith(pattern):
            return const, text[len(pattern) :]

    return None, text


def detect_suffix_punctuation(text):
    """
    Detect suffix punctuation combinations at the end of text.

    Returns:
        tuple: (punctuation_type_constant, remaining_text) or (None, text) if no match
    """
    if not text:
        return None, text

    # Define suffix patterns in order of priority (longest first)
    # Format: (pattern, constant)
    suffix_patterns = [
        # Two-character combinations
        (")'", SUFFIX_PUNCT_PAREN_SINGLE),
        (')"', SUFFIX_PUNCT_PAREN_DOUBLE),
        ("]'", SUFFIX_PUNCT_SQUARE_SINGLE),
        (']"', SUFFIX_PUNCT_SQUARE_DOUBLE),
        (".'", SUFFIX_PUNCT_PERIOD_SINGLE),
        ('."', SUFFIX_PUNCT_PERIOD_DOUBLE),
        ("!'", SUFFIX_PUNCT_EXCLAIM_SINGLE),
        ('!"', SUFFIX_PUNCT_EXCLAIM_DOUBLE),
        ("?'", SUFFIX_PUNCT_QUESTION_SINGLE),
        ('?"', SUFFIX_PUNCT_QUESTION_DOUBLE),
        (",'", SUFFIX_PUNCT_COMMA_SINGLE),
        (',"', SUFFIX_PUNCT_COMMA_DOUBLE),
        (".)", SUFFIX_PUNCT_PERIOD_PAREN),
        ("!)", SUFFIX_PUNCT_EXCLAIM_PAREN),
        ("?)", SUFFIX_PUNCT_QUESTION_PAREN),
        (",)", SUFFIX_PUNCT_COMMA_PAREN),
        (")-", SUFFIX_PUNCT_PAREN_HYPHEN),
        ("]-", SUFFIX_PUNCT_SQUARE_HYPHEN),
        ("'s", SUFFIX_PUNCT_POSSESSIVE_S),  # Must come before single quote
        ("s'", SUFFIX_PUNCT_POSSESSIVE_PLURAL),
        ("\u2019s", SUFFIX_PUNCT_POSSESSIVE_S),
        ("s\u2019", SUFFIX_PUNCT_POSSESSIVE_PLURAL),
        # Single characters
        ("'", SUFFIX_PUNCT_SINGLE_QUOTE),  # Can be both regular quote or possessive
        ("\u2019", SUFFIX_PUNCT_SINGLE_QUOTE),
        ('"', SUFFIX_PUNCT_DOUBLE_QUOTE),
        ("\u201c", SUFFIX_PUNCT_DOUBLE_QUOTE),
        ("\u201d", SUFFIX_PUNCT_DOUBLE_QUOTE),
        ("`", SUFFIX_PUNCT_BACKTICK),
        (")", SUFFIX_PUNCT_PAREN),
        ("]", SUFFIX_PUNCT_SQUARE),
        ("}", SUFFIX_PUNCT_CURLY),
        (".", SUFFIX_PUNCT_PERIOD),
        ("!", SUFFIX_PUNCT_EXCLAIM),
        ("?", SUFFIX_PUNCT_QUESTION),
        (",", SUFFIX_PUNCT_COMMA),
        (";", SUFFIX_PUNCT_SEMICOLON),
        (":", SUFFIX_PUNCT_COLON),
        ("-", SUFFIX_PUNCT_HYPHEN),
    ]

    for pattern, const in suffix_patterns:
        if text.endswith(pattern):
            return const, text[: -len(pattern)]

    return None, text


def detect_article_prefix(text, articles=None):
    """
    Detect article prefix at the start of text.

    Args:
        text: Input text to check
        articles: List of articles to check for (default: ["the", "a", "an"])

    Returns:
        tuple: (article_constant, is_capitalized, remaining_text) or (None, False, text) if no match
    """
    if not text:
        return None, False, text

    if articles is None:
        articles = ["the", "a", "an"]

    text_lower = text.lower()

    for article in articles:
        # Check if text starts with article followed by space
        if text_lower.startswith(article + " "):
            is_cap = text[: len(article)][0].isupper()
            remaining = text[len(article) + 1 :]  # +1 to skip the space

            # Return the appropriate constant
            if article == "the":
                return ARTICLE_THE, is_cap, remaining
            elif article == "a":
                return ARTICLE_A, is_cap, remaining
            elif article == "an":
                return ARTICLE_AN, is_cap, remaining

    return None, False, text


def detect_preposition_prefix(text, prepositions=None):
    """
    Detect preposition prefix at the start of text.

    Args:
        text: Input text to check
        prepositions: List of prepositions to check for (default: common prepositions)

    Returns:
        tuple: (preposition_constant, is_capitalized, remaining_text) or (None, False, text) if no match
    """
    if not text:
        return None, False, text

    if prepositions is None:
        # Default to most common prepositions
        prepositions = ["by", "at", "of", "to", "in", "on", "with", "for", "from"]

    text_lower = text.lower()

    # Sort by length (longest first) to match longer prepositions first
    for prep in sorted(prepositions, key=len, reverse=True):
        # Check if text starts with preposition followed by space
        if text_lower.startswith(prep + " "):
            is_cap = text[: len(prep)][0].isupper()
            remaining = text[len(prep) + 1 :]  # +1 to skip the space

            # Return the appropriate constant dynamically
            prep_const = f"prep_{prep}"
            return prep_const, is_cap, remaining

    return None, False, text


def strip_all_affixes(
    text,
    space_prefix=" ",
    decompose_punctuation=False,
    decompose_articles=False,
    decompose_prepositions=False,
    prepositions=None,
    track_article_prep_space_prefix=False,
):
    """
    Strip all affixes (space, punctuation, articles, prepositions) from text.

    Returns:
        tuple: (base_text, transformations_dict) where transformations_dict contains:
            - 'prefix_punct': prefix punctuation constant or None
            - 'suffix_punct': suffix punctuation constant or None
            - 'article': article constant or None
            - 'article_cap': whether article is capitalized
            - 'prep': preposition constant or None
            - 'prep_cap': whether preposition is capitalized
            - 'space_prefix': whether has space prefix
            - 'article_space_prefix': whether article had a leading space
            - 'prep_space_prefix': whether preposition had a leading space
    """
    transformations = {
        "prefix_punct": None,
        "suffix_punct": None,
        "article": None,
        "article_cap": False,
        "article_space_prefix": False,
        "prep": None,
        "prep_cap": False,
        "prep_space_prefix": False,
        "space_prefix": False,
    }

    # Strip space prefix first
    if text.startswith(space_prefix):
        transformations["space_prefix"] = True
        text = text[len(space_prefix) :]

    # Strip prefix punctuation
    if decompose_punctuation:
        punct, text = detect_prefix_punctuation(text)
        transformations["prefix_punct"] = punct

    # Strip preposition (must come before article since "the" can be in preposition phrase)
    if decompose_prepositions:
        prep, prep_cap, text = detect_preposition_prefix(text, prepositions)
        transformations["prep"] = prep
        transformations["prep_cap"] = prep_cap
        if track_article_prep_space_prefix and prep:
            transformations["prep_space_prefix"] = (
                transformations["space_prefix"] and transformations["prefix_punct"] is None
            )

    # Strip article
    if decompose_articles:
        article, article_cap, text = detect_article_prefix(text)
        transformations["article"] = article
        transformations["article_cap"] = article_cap
        if track_article_prep_space_prefix and article:
            transformations["article_space_prefix"] = bool(transformations["prep"]) or (
                transformations["space_prefix"] and transformations["prefix_punct"] is None
            )

    # Strip suffix punctuation
    if decompose_punctuation:
        punct, text = detect_suffix_punctuation(text)
        transformations["suffix_punct"] = punct

    return text, transformations


DEFAULT_UNIMORPH_ROOT = os.environ.get(
    "UNIMORPH_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "unimorph"),
)
DEFAULT_UNIMORPH_INFLECTIONS_PATH = os.path.join(DEFAULT_UNIMORPH_ROOT, "eng", "eng")
DEFAULT_UNIMORPH_DERIVATIONS_PATH = os.path.join(
    DEFAULT_UNIMORPH_ROOT, "eng", "eng.derivations.tsv"
)

# Mapping from 2-letter ISO 639-1 codes to (directory, filename) for UniMorph
# Used when --unimorph_root is provided to auto-resolve language-specific paths
#
# Format: "iso2": (inflections_stem, derivations_stem)
# - Most languages: same stem for both (e.g., "spa" for Spanish)
# - Special cases: Arabic uses "ara_new" for updated data
#
# Supported languages:
# - en (English), es (Spanish), fr (French), it (Italian), nl (Dutch)
# - de (German), pt (Portuguese), ru (Russian), tr (Turkish)
# - he (Hebrew), ar (Arabic), ja (Japanese), fa (Persian/Farsi)
#
# For unsupported languages:
# - Use 3-letter ISO 639-3 code as --language
# - OR provide explicit --unimorph_inflections_path
UNIMORPH_LANGUAGE_MAP = {
    "en": ("eng", "eng"),
    "es": ("spa", "spa"),
    "he": ("heb", "heb"),
    "ar": ("ara", "ara_new"),  # Special: ara_new for updated data
    "it": ("ita", "ita"),
    "nl": ("nld", "nld"),
    "fr": ("fra", "fra"),
    "pt": ("por", "por"),
    "ru": ("rus", "rus"),
    "de": ("deu", "deu"),
    "ja": ("jpn", "jpn"),
    "fa": ("fas", "fas"),
    "tr": ("tur", "tur"),
}


def _resolve_unimorph_paths(
    language, unimorph_root, unimorph_inflections_path, unimorph_derivations_path
):
    unimorph_root = unimorph_root or os.environ.get("UNIMORPH_ROOT")
    if unimorph_root is None:
        unimorph_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "resources", "unimorph"
        )
    if unimorph_inflections_path:
        if unimorph_derivations_path is None and (language or "en").lower() == "en":
            return unimorph_inflections_path, DEFAULT_UNIMORPH_DERIVATIONS_PATH
        return unimorph_inflections_path, unimorph_derivations_path

    lang = (language or "en").lower()
    if unimorph_root:
        lang_code, lang_file = UNIMORPH_LANGUAGE_MAP.get(lang, (None, None))
        if lang_code is None:
            if len(lang) == 3:
                lang_code = lang
                lang_file = lang
            else:
                supported = ", ".join(sorted(UNIMORPH_LANGUAGE_MAP.keys()))
                raise ValueError(
                    f"Language '{language}' not in UNIMORPH_LANGUAGE_MAP.\n"
                    f"Supported 2-letter codes: {supported}\n"
                    f"Alternatives:\n"
                    f"  - Use 3-letter ISO 639-3 code as --language\n"
                    f"  - Provide explicit --unimorph_inflections_path"
                )

        inflections_path = os.path.join(unimorph_root, lang_code, lang_file)
        if unimorph_derivations_path is None and lang == "en":
            deriv_stem = UNIMORPH_LANGUAGE_MAP["en"][1]
            unimorph_derivations_path = os.path.join(
                unimorph_root,
                lang_code,
                f"{deriv_stem}.derivations.tsv",
            )
    else:
        lang_code, lang_file = UNIMORPH_LANGUAGE_MAP.get(lang, (None, None))
        if lang_code is None and len(lang) == 3:
            lang_code = lang_file = lang
        if lang_code is None:
            raise ValueError(
                f"Language '{language}' requires UniMorph data. Set UNIMORPH_ROOT or pass "
                "--unimorph_root /path/to/unimorph (or an explicit --unimorph_inflections_path)."
            )
        inflections_path = os.path.join(unimorph_root, lang_code, lang_file)
        if lang == "en":
            deriv_stem = UNIMORPH_LANGUAGE_MAP["en"][1]
            unimorph_derivations_path = os.path.join(
                unimorph_root, lang_code, f"{deriv_stem}.derivations.tsv"
            )

    # Check if file exists (try common extensions)
    possible_paths = [
        inflections_path,
        f"{inflections_path}.tsv",
    ]
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path and found_path != inflections_path:
        inflections_path = found_path
    elif not found_path:
        raise FileNotFoundError(
            f"UniMorph inflections file not found for language '{language}'. Tried: "
            f"{', '.join(possible_paths)}. Set UNIMORPH_ROOT to the directory containing "
            "language subdirectories, or pass --unimorph_inflections_path."
        )

    return inflections_path, unimorph_derivations_path


def categorize_unimorph_tag(tag: str) -> str:
    """Categorize a UniMorph tag into its morphological group.

    Args:
        tag: UniMorph tag string (e.g., 'PRS', 'IND', 'MASC')

    Returns:
        Group name (e.g., 'tense', 'mood', 'gender')
    """
    # Handle special cases
    if tag.startswith("LGSPEC"):
        return "lgspec"
    if tag.startswith("V.PTCP") or tag.startswith("V.CVB") or tag.startswith("V.MSDR"):
        return "finiteness"

    # Look up in map, default to 'pos' for unknown tags
    return UNIMORPH_TAG_CATEGORY_MAP.get(tag, "pos")


def map_unimorph_tags_to_compositional_groups(
    tag_tuple: Tuple[str, ...], language: str = "en", clitic_mode: str = "surface2"
) -> Dict[str, str]:
    """Map UniMorph tag tuple to compositional feature groups.

    Args:
        tag_tuple: Tuple of UniMorph tags (e.g., ('V', 'IND', 'PRS', '1', 'SG'))
        language: Language code (for future language-specific handling)

    Returns:
        Dictionary mapping group names to values

    Example:
        >>> map_unimorph_tags_to_compositional_groups(('V', 'IND', 'PRS', '1', 'SG'))
        {'pos': 'V', 'mood': 'IND', 'tense': 'PRS', 'person': '1', 'number': 'SG', ...}
    """
    groups = {group: "none" for group in MORPHOLOGICAL_GROUPS}

    if not tag_tuple:
        return groups

    tags = list(tag_tuple)
    groups["pos"] = tags[0]

    def _set_group_value(group_name: str, value: str):
        if groups[group_name] == "none":
            groups[group_name] = value
        else:
            if value not in groups[group_name].split("+"):
                groups[group_name] += f"+{value}"

    for tag in tags[1:]:
        if clitic_mode == "features":
            if tag.startswith("CLITIC_PERSON_"):
                _set_group_value("clitic_person", tag.replace("CLITIC_PERSON_", ""))
                continue
            if tag.startswith("CLITIC_NUMBER_"):
                _set_group_value("clitic_number", tag.replace("CLITIC_NUMBER_", ""))
                continue
            if tag.startswith("CLITIC_GENDER_"):
                _set_group_value("clitic_gender", tag.replace("CLITIC_GENDER_", ""))
                continue
            if tag.startswith("CLITIC_CASE_"):
                _set_group_value("clitic_case", tag.replace("CLITIC_CASE_", ""))
                continue
            if tag == "CLITIC_REFLEXIVE":
                _set_group_value("clitic_reflexive", "REFL")
                continue
        else:
            if tag.startswith("CLITIC1_SURFACE_"):
                _set_group_value("clitic1_surface", tag.replace("CLITIC1_SURFACE_", ""))
                continue
            if tag.startswith("CLITIC2_SURFACE_"):
                _set_group_value("clitic2_surface", tag.replace("CLITIC2_SURFACE_", ""))
                continue

        if tag in ("PRO", "REFL"):
            if clitic_mode == "features":
                _set_group_value("clitic_type", tag)
            continue

        category = categorize_unimorph_tag(tag)
        _set_group_value(category, tag)

    for group in groups:
        groups[group] = _normalize_group_value(groups[group])

    return groups


def merge_compositional_groups(group_mappings: List[Dict[str, str]]) -> Dict[str, str]:
    """Merge multiple compositional group mappings for homophonous forms.

    This handles cases where the same surface form corresponds to multiple
    grammatical analyses (syncretism). Groups that have consistent values
    across all analyses are preserved, while groups with different values
    are merged with '+' separator.

    Args:
        group_mappings: List of group dicts from different analyses

    Returns:
        Merged dict with multi-valued groups joined by '+'

    Example:
        >>> merge_compositional_groups([
        ...     {'pos': 'N', 'case': 'NOM', 'number': 'PL', 'gender': 'MASC'},
        ...     {'pos': 'N', 'case': 'ACC', 'number': 'PL', 'gender': 'MASC'}
        ... ])
        {'pos': 'N', 'case': 'ACC+NOM', 'number': 'PL', 'gender': 'MASC', ...}
    """
    if len(group_mappings) == 1:
        return group_mappings[0]

    merged = {}
    for group in MORPHOLOGICAL_GROUPS:
        # Collect all values for this group across analyses
        values = [m.get(group, "none") for m in group_mappings]
        unique_vals = sorted(set(v for v in values if v != "none"), key=_group_value_sort_key)

        if unique_vals:
            # Multiple unique values -> join with '+'
            merged[group] = "+".join(unique_vals)
        else:
            # No values or all 'none'
            merged[group] = "none"

    return merged


def _split_tag_str(tag_str: str) -> List[Tuple[str, ...]]:
    """Split a UniMorph tag string (possibly with '+') into tag tuples."""
    tag_sets = []
    for part in tag_str.split("+"):
        part = part.strip()
        if not part:
            continue
        tag_sets.append(tuple(part.split(";")))
    return tag_sets


def _get_morph_group_from_transform(
    transform: str, groups: Optional[List[str]] = None
) -> Optional[str]:
    groups = groups or MORPHOLOGICAL_GROUPS
    for group in groups:
        if (
            transform == MORPHOLOGICAL_GROUP_NO_VALUES[group]
            or transform == MORPHOLOGICAL_GROUP_NA_VALUES[group]
        ):
            return group
        if transform.startswith(f"{group}_"):
            return group
    return None


def _infer_active_morph_groups_from_map(
    decomposition_map: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    active = set()
    for _, inflections in decomposition_map.items():
        for _, transforms in inflections.items():
            for t in transforms:
                group = _get_morph_group_from_transform(t)
                if group and t not in {
                    MORPHOLOGICAL_GROUP_NO_VALUES[group],
                    MORPHOLOGICAL_GROUP_NA_VALUES[group],
                }:
                    active.add(group)
    return [group for group in MORPHOLOGICAL_GROUPS if group in active]


def is_morph_transform_name(transform: str) -> bool:
    if transform.startswith("inflect_"):
        return True
    if transform.startswith("deriv_"):
        return True
    group = _get_morph_group_from_transform(transform)
    if group is None:
        return False
    return transform not in {
        MORPHOLOGICAL_GROUP_NO_VALUES[group],
        MORPHOLOGICAL_GROUP_NA_VALUES[group],
    }


def get_vocabulary_decomposition(
    tokenizer,
    unimorph_inflections_path=None,
    unimorph_derivations_path=None,
    language="en",
    unimorph_root=None,
    space_prefix=" ",
    include_base_only=True,
    include_all_caps=False,
    override_unimorph_with_inflections=True,
    decompose_spaces=True,
    decompose_capitalization=True,
    use_relative_space_cap_transforms=False,
    canonicalize_token_strings=False,
    transform_collision_mode="off",
    skip_if_no_space_prefix=False,
    skip_stop_words=True,
    include_derivations=False,
    use_type_str_as_type=False,
    no_diacritics=False,
    filter_propernouns_and_aux=True,
    merge_plural_and_present_singular=False,
    include_non_resource_latin_tokens=False,
    include_non_latin_tokens=False,
    min_derivation_count=50,
    prefer_popular_base_conflicts=True,
    allow_chained_derivations=False,
    decompose_punctuation=False,
    decompose_articles=False,
    decompose_prepositions=False,
    decompose_article_prep_space_prefix=False,
    preposition_list=None,
    modifier_generation_mode="full",
    skip_multi_token_words=False,
    skip_three_token_words=False,
    skip_four_token_words=False,
    skip_multi_token_bases=False,
    multitoken_allowed_groups=None,
    max_tokens_per_base_word=None,
    max_vocab_items=None,
    morphology_mode="atomic",
    clitic_mode="surface2",
    log_second_pass_bases=True,
    log_derivation_conflicts=False,
    enable_build_optimizations=False,
):
    """
    Create a comprehensive token decomposition map using UniMorph data and other resources.

    Args:
        tokenizer: Hugging Face tokenizer
        unimorph_inflections_path: Path to UniMorph inflections data
        unimorph_derivations_path: Path to UniMorph derivations data
        language: Language code for UniMorph selection (default: "en")
        unimorph_root: Base path to UniMorph language directories
        space_prefix: Character used for space prefix
        include_base_only: Whether to include words with no inflections
        include_all_caps: Deprecated. All-caps tokens are never used as base forms;
                         they are represented via capitalization transforms.
        override_unimorph_with_inflections: Whether to treat words as inflections even if they appear as base forms
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation
        use_relative_space_cap_transforms: Use add/remove for space/cap instead of absolute forms
        canonicalize_token_strings: Canonicalize tokens missing space/cap variants for decomposition
        transform_collision_mode: Handle transform collisions (off, keep_first, drop_both)
        include_derivations: Whether to include derivations
        min_derivation_count: Minimum count for common derivation types
        decompose_punctuation: Whether to enable punctuation decomposition
        decompose_articles: Whether to enable article decomposition
        decompose_prepositions: Whether to enable preposition decomposition
        decompose_article_prep_space_prefix: Track space prefix for article/preposition tokens
        preposition_list: List of prepositions to use (if decomposing prepositions)
        modifier_generation_mode: "full" (all combinations) or "vocab_only" (only tokens already in vocab)
        skip_multi_token_words: Skip multi-token inflections when extending tokenizer vocab
        skip_three_token_words: Skip inflections with 3+ tokens (allows up to 2)
        skip_four_token_words: Skip inflections with 4+ tokens (allows up to 3)
        skip_multi_token_bases: Skip processing multi-token base words from UniMorph
        max_tokens_per_base_word: Maximum tokens per base word (None = unlimited)
        max_vocab_items: Maximum vocabulary items to process (None = unlimited, for quick smoke tests)
        morphology_mode: Morphology representation mode: 'atomic' (default, each tag combo = one dimension)
                        or 'compositional' (decompose into feature groups like tense/mood/person)
        clitic_mode: "surface2" (ordered clitic1+clitic2), "surface1" (clitic1 only),
                     or "features" (clitic feature groups)
        log_second_pass_bases: Whether to log per-token second-pass base processing lines
        log_derivation_conflicts: Whether to log per-derivation conflict/skip lines inside inner loops
        enable_build_optimizations: Enable optional memoization/caching during map construction

    Returns:
        Tuple containing:
        - base_tokens: Dict mapping base forms to token IDs
        - decomposition_map: Dict mapping base forms to {inflection: {types: [], types_str: str}}
        - ambiguous_inflections: Dict of ambiguous inflections
        - inflection_baseform_conflicts: List of conflicts between base forms and inflections
    """
    # Initialize analyzers
    unimorph_inflections_path, unimorph_derivations_path = _resolve_unimorph_paths(
        language, unimorph_root, unimorph_inflections_path, unimorph_derivations_path
    )
    use_english_resources = (language or "en").lower() == "en"
    if not use_english_resources:
        filter_propernouns_and_aux = False
        skip_stop_words = False

    print("Loading UniMorph data...")
    unimorph = UniMorphAnalyzer(
        unimorph_inflections_path, unimorph_derivations_path, no_diacritics=no_diacritics
    )
    unimorph_skip_lemmas: Set[str] = set()
    if use_english_resources:
        for lemma, forms in unimorph.lemma_to_forms.items():
            if _has_whitespace(lemma):
                unimorph_skip_lemmas.add(lemma)
                continue
            if len(lemma) <= 3 and not _get_english_dictionary().check(lemma):
                if any(vocab_base_words_filter(form) for form in forms):
                    unimorph_skip_lemmas.add(lemma)

    def _should_skip_unimorph_pair(lemma: str, form: str, tag_sets: Set[Tuple[str, ...]]) -> bool:
        if not use_english_resources:
            return False
        if lemma in unimorph_skip_lemmas:
            return True
        if _has_whitespace(lemma) or _has_whitespace(form):
            return True
        lemma_lower = lemma.lower()
        form_lower = form.lower()
        if (len(lemma_lower) - len(form_lower) >= 3) and lemma_lower.startswith(form_lower):
            for tag_set in tag_sets:
                if _tag_set_has(tag_set, ["N", "PL"]) or _tag_set_has(
                    tag_set, ["V", "PRS", "3", "SG"]
                ):
                    return True
        if (
            len(lemma_lower) <= 3
            and vocab_base_words_filter(form_lower)
            and not vocab_base_words_filter(lemma_lower)
        ):
            return True
        return False

    # Retain no-op hooks so older internal call sites remain harmless while
    # keeping tracing and debugger traps out of the published runtime.

    # Set common derivation types if needed
    if include_derivations and unimorph_derivations_path:
        unimorph.set_common_derivation_types(min_count=min_derivation_count)

    lemma_popularity: Dict[str, int] = Counter()
    for lemma, forms in unimorph.lemma_to_forms.items():
        lemma_popularity[lemma.lower()] += len(forms) + 1
    if include_derivations and unimorph_derivations_path:
        for root, derivations in unimorph.root_to_derivations.items():
            lemma_popularity[root.lower()] += len(derivations)
    preferred_lemma_for_form: Dict[str, str] = {}
    if prefer_popular_base_conflicts:
        for form, lemmas in unimorph.form_to_lemmas.items():
            if len(lemmas) <= 1:
                continue
            best_lemma = min(
                (lemma.lower() for lemma in lemmas.keys()),
                key=lambda lemma: (-lemma_popularity.get(lemma, 0), len(lemma), lemma),
            )
            preferred_lemma_for_form[form.lower()] = best_lemma

    def _lemma_popularity(word: str) -> int:
        return lemma_popularity.get(word.lower(), 0)

    def _preferred_lemma_for_form(word: str) -> Optional[str]:
        if not prefer_popular_base_conflicts:
            return None
        return preferred_lemma_for_form.get(word.lower())

    def _candidate_allowed_for_form(form_word: str, candidate_lemma: str) -> bool:
        if not prefer_popular_base_conflicts:
            return True
        preferred_lemma = _preferred_lemma_for_form(form_word)
        if preferred_lemma is None:
            return True
        candidate_key = candidate_lemma.lower()
        if preferred_lemma == candidate_key:
            return True
        preferred_popularity = _lemma_popularity(preferred_lemma)
        candidate_popularity = _lemma_popularity(candidate_key)
        if preferred_popularity <= 0:
            return True
        return candidate_popularity >= (preferred_popularity * 2.0)

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
    filtered_transform_collisions = []  # For tracking inflections filtered due to transform collisions
    filtered_propernouns = set()
    filtered_aux = set()
    converted_titlecase_bases = []

    # Get the vocabulary
    vocab = tokenizer.get_vocab()
    vocab_items = sorted(vocab.items(), key=lambda x: x[1])

    use_build_optimizations = bool(enable_build_optimizations)

    single_token_cache: Dict[str, Optional[int]] = {}
    encoded_text_cache: Dict[str, Tuple[int, ...]] = {}
    english_word_cache: Dict[Tuple[str, bool], bool] = {}
    pos_cache: Dict[str, Set[str]] = {}
    pyinflect_cache: Dict[Tuple[str, Tuple[str, ...]], Dict[str, str]] = {}
    normalized_lookup_cache: Dict[str, str] = {}

    def _encode_cached(text: str) -> Tuple[int, ...]:
        if not use_build_optimizations:
            return tuple(tokenizer.encode(text, add_special_tokens=False))
        cached = encoded_text_cache.get(text)
        if cached is not None:
            return cached
        encoded = tuple(tokenizer.encode(text, add_special_tokens=False))
        encoded_text_cache[text] = encoded
        return encoded

    def _encode_variant_cached(text: str) -> Tuple[int, ...]:
        normalized = text.replace(space_prefix, " ") if space_prefix else text
        return _encode_cached(normalized)

    def _single_token_id_cached(text: str) -> Optional[int]:
        if not use_build_optimizations:
            token_ids = _encode_cached(text)
            return token_ids[0] if len(token_ids) == 1 else None
        cached = single_token_cache.get(text)
        if cached is not None or text in single_token_cache:
            return cached
        token_ids = _encode_cached(text)
        token_id = token_ids[0] if len(token_ids) == 1 else None
        single_token_cache[text] = token_id
        return token_id

    def _is_english_word_cached(word: str, use_unimorph: bool = True) -> bool:
        if not use_build_optimizations:
            return is_english_word(word, unimorph if use_unimorph else None)
        key = (word, use_unimorph)
        if key in english_word_cache:
            return english_word_cache[key]
        value = is_english_word(word, unimorph if use_unimorph else None)
        english_word_cache[key] = value
        return value

    def _get_pos_from_wordnet_cached(word: str) -> Set[str]:
        if not use_build_optimizations:
            return get_pos_from_wordnet(word)
        cached = pos_cache.get(word)
        if cached is not None:
            return cached
        value = get_pos_from_wordnet(word)
        pos_cache[word] = value
        return value

    def _get_pyinflect_variations_cached(word: str, pos_set: Set[str]) -> Dict[str, str]:
        if not use_build_optimizations:
            return get_pyinflect_variations(word, pos_set)
        key = (word, tuple(sorted(pos_set)))
        cached = pyinflect_cache.get(key)
        if cached is not None:
            return cached
        value = get_pyinflect_variations(word, pos_set)
        pyinflect_cache[key] = value
        return value

    def _normalize_lookup_key(text: str) -> str:
        if not use_build_optimizations:
            return text.lower().strip(space_prefix)
        cached = normalized_lookup_cache.get(text)
        if cached is not None:
            return cached
        normalized = text.lower().strip(space_prefix)
        normalized_lookup_cache[text] = normalized
        return normalized

    transform_collision_mode = (transform_collision_mode or "off").lower()
    if transform_collision_mode not in {"off", "keep_first", "drop_both"}:
        raise ValueError(f"Unknown transform_collision_mode: {transform_collision_mode}")

    article_space_prefix_default = None
    prep_space_prefix_default = None
    if decompose_article_prep_space_prefix:
        if decompose_articles:
            article_space_prefix_default = NO_ARTICLE_SPACE_PREFIX
        if decompose_prepositions:
            prep_space_prefix_default = NO_PREP_SPACE_PREFIX

    base_key_to_tokens: DefaultDict[str, List[str]] = defaultdict(list)
    derivation_key_to_bases: DefaultDict[str, List[str]] = defaultdict(list)

    def _append_unique(items: List[str], value: str) -> None:
        if value not in items:
            items.append(value)

    def _remove_value(items: List[str], value: str) -> None:
        if value in items:
            items.remove(value)

    def _index_base_entry(base_word: str, variants: Dict[str, List[str]]) -> None:
        base_key = _normalize_lookup_key(base_word)
        _append_unique(base_key_to_tokens[base_key], base_word)
        for variant_str, transforms in variants.items():
            if any(transform.startswith("deriv_") for transform in transforms):
                variant_key = _normalize_lookup_key(variant_str)
                _append_unique(derivation_key_to_bases[variant_key], base_word)

    def _remove_base_entry_from_indexes(
        base_word: str, variants: Optional[Dict[str, List[str]]] = None
    ) -> None:
        base_key = _normalize_lookup_key(base_word)
        if base_key in base_key_to_tokens:
            _remove_value(base_key_to_tokens[base_key], base_word)
            if not base_key_to_tokens[base_key]:
                del base_key_to_tokens[base_key]
        if variants is None:
            variants = decomposition_map.get(base_word, {})
        for variant_str, transforms in variants.items():
            if any(transform.startswith("deriv_") for transform in transforms):
                variant_key = _normalize_lookup_key(variant_str)
                if variant_key in derivation_key_to_bases:
                    _remove_value(derivation_key_to_bases[variant_key], base_word)
                    if not derivation_key_to_bases[variant_key]:
                        del derivation_key_to_bases[variant_key]

    def _rebuild_conflict_indexes() -> None:
        base_key_to_tokens.clear()
        derivation_key_to_bases.clear()
        for base_word, variants in decomposition_map.items():
            _index_base_entry(base_word, variants)

    MAX_DERIVATION_CLASSIFICATION_SAMPLES = 25
    derivation_decision_stats = Counter()
    derivation_decision_samples: DefaultDict[str, List[str]] = defaultdict(list)
    root_derivation_forms_cache: Dict[str, Set[str]] = {}
    form_candidate_roots_cache: Dict[str, Set[str]] = {}

    def _record_derivation_decision(bucket: str, message: str) -> None:
        derivation_decision_stats[bucket] += 1
        samples = derivation_decision_samples[bucket]
        if len(samples) < MAX_DERIVATION_CLASSIFICATION_SAMPLES:
            samples.append(message)

    def _get_root_derivation_forms(root_word: str) -> Set[str]:
        root_key = _normalize_lookup_key(root_word)
        if not use_build_optimizations:
            return {
                _normalize_lookup_key(derived_form)
                for derived_form in unimorph.get_root_derivations(root_key).keys()
            }
        cached = root_derivation_forms_cache.get(root_key)
        if cached is not None:
            return cached
        derived_forms = {
            _normalize_lookup_key(derived_form)
            for derived_form in unimorph.get_root_derivations(root_key).keys()
        }
        root_derivation_forms_cache[root_key] = derived_forms
        return derived_forms

    def _get_candidate_roots_for_form(form_word: str) -> Set[str]:
        form_key = _normalize_lookup_key(form_word)
        if not use_build_optimizations:
            candidates: Set[str] = set()
            lemma_map = unimorph.form_to_lemmas.get(form_key, {})
            for lemma in lemma_map.keys():
                normalized_lemma = _normalize_lookup_key(lemma)
                if normalized_lemma:
                    candidates.add(normalized_lemma)
            if form_key in unimorph.lemma_to_forms:
                candidates.add(form_key)
            return candidates
        cached = form_candidate_roots_cache.get(form_key)
        if cached is not None:
            return cached
        candidates: Set[str] = set()
        lemma_map = unimorph.form_to_lemmas.get(form_key, {})
        for lemma in lemma_map.keys():
            normalized_lemma = _normalize_lookup_key(lemma)
            if normalized_lemma:
                candidates.add(normalized_lemma)
        if form_key in unimorph.lemma_to_forms:
            candidates.add(form_key)
        form_candidate_roots_cache[form_key] = candidates
        return candidates

    def _classify_derivation_overlap(
        current_form: str,
        derived_form: str,
        existing_base_keys: Set[str],
    ) -> Tuple[Set[str], Set[str]]:
        normalized_derived = _normalize_lookup_key(derived_form)
        candidate_roots = _get_candidate_roots_for_form(current_form)
        roots_with_derived = {
            root_key
            for root_key in candidate_roots
            if normalized_derived in _get_root_derivation_forms(root_key)
        }
        covered_roots = roots_with_derived.intersection(existing_base_keys)
        return roots_with_derived, covered_roots

    # Limit vocabulary items for smoke testing if requested
    if max_vocab_items is not None:
        vocab_items = vocab_items[:max_vocab_items]
        print(f"SMOKE TEST MODE: Limited vocabulary to {len(vocab_items)} items")

    # Process vocabulary - in two passes
    # in the first pass we only allow *base* words, with *space prefix* and *without capitalization*
    # in the second pass we allow everything else, that isn't already covered in the decomposition map
    print("Processing vocabulary...")
    vocab_surface_items = []
    for token, token_id in vocab_items:
        token_str = tokenizer.convert_tokens_to_string([token])
        token_has_space_prefix = token_str.startswith(space_prefix)
        token_repr = token_str.strip(space_prefix)
        token_without_prefix = (
            token_str[len(space_prefix) :] if token_has_space_prefix else token_str
        )
        is_whitespace_only_token = bool(token_str) and all(ch.isspace() for ch in token_str)
        is_punctuation_only_token = bool(token_without_prefix) and all(
            (not ch.isalnum()) and (not ch.isspace()) for ch in token_without_prefix
        )
        is_symbol_token = is_whitespace_only_token or is_punctuation_only_token
        vocab_surface_items.append(
            (
                token,
                token_id,
                token_str,
                token_repr,
                token_has_space_prefix,
                token_without_prefix,
                is_symbol_token,
            )
        )

    two_pass_vocab_items = vocab_surface_items + [None] + vocab_surface_items
    first_pass = True
    for item in tqdm(two_pass_vocab_items):
        if item is None:
            first_pass = False
            continue
        (
            token,
            token_id,
            token_str,
            token_repr,
            token_has_space_prefix,
            token_without_prefix,
            is_symbol_token,
        ) = item

        if canonicalize_token_strings:
            canonical_repr = token_repr.lower()
            canonical_str = (
                f"{space_prefix}{canonical_repr}" if decompose_spaces else canonical_repr
            )
            if canonical_str != token_str:
                canonical_token_id = _single_token_id_cached(canonical_str)
                if canonical_token_id is None:
                    token_str = canonical_str
                    token_repr = canonical_repr
                    token_has_space_prefix = token_str.startswith(space_prefix)

        if decompose_capitalization and token_repr and token_repr != token_repr.lower():
            is_titlecase = token_repr[0].isupper() and token_repr[1:].islower()
            if not is_titlecase and not token_repr.isupper():
                lowercase_repr = token_repr.lower()
            if _get_english_dictionary().check(lowercase_repr):
                lowercase_str = (
                    f"{space_prefix}{lowercase_repr}" if token_has_space_prefix else lowercase_repr
                )
                lowercase_token_id = _single_token_id_cached(lowercase_str)
                if lowercase_token_id is not None:
                    token_str = lowercase_str
                    token_repr = lowercase_repr
                    token_has_space_prefix = token_str.startswith(space_prefix)

        original_token_str = token_str
        original_token_repr = token_repr
        if decompose_capitalization and (not first_pass) and token_repr:
            is_titlecase = token_repr[0].isupper() and token_repr[1:].islower()
            if is_titlecase:
                lowercase_repr = token_repr.lower()
                lowercase_str = (
                    f"{space_prefix}{lowercase_repr}" if token_has_space_prefix else lowercase_repr
                )
                lowercase_token_id = _single_token_id_cached(lowercase_str)
                if lowercase_token_id is None:
                    token_str = lowercase_str
                    token_repr = lowercase_repr
                    token_has_space_prefix = token_str.startswith(space_prefix)
                    converted_titlecase_bases.append((original_token_str, token_str))
                elif lowercase_str in decomposition_map:
                    continue

        if token_str in processed_tokens and token_str in decomposition_map:
            continue

        # Skip characters and very short words
        if len(token_repr) < 2 and (not is_symbol_token):
            continue

        # Skip if we're decomposing spaces and this token doesn't have a space prefix
        if (
            decompose_spaces
            and (not token_has_space_prefix)
            and (skip_if_no_space_prefix or first_pass)
            and (not is_symbol_token)
        ):
            continue

        # Skip if not English or contains punctuation
        if use_english_resources:
            if (
                (not include_non_resource_latin_tokens or first_pass)
                and (not is_symbol_token)
                and not _is_english_word_cached(token_repr, use_unimorph=True)
            ):
                continue

        # All-caps surfaces are modeled via capitalization transforms and
        # should not be used as base-form entries.
        if token_repr.isupper() and (not is_symbol_token):
            continue

        # Preserve original capitalization when decompose_capitalization is False
        original_token_repr = token_repr

        # Check if we're dealing with a capitalized form when decompose_capitalization is False
        is_capitalized = (
            token_repr != token_repr.lower()
            and token_repr[0].isupper()
            and not token_repr.isupper()
        )
        should_preserve_caps = is_capitalized and not decompose_capitalization

        if is_capitalized and first_pass:
            continue

        # Try with original capitalization first when needed, otherwise use lowercase
        token_repr_for_lookup = token_repr if should_preserve_caps else token_repr.lower()

        # Skip multi-word expressions from UniMorph (e.g., "door to door")
        # These should not be processed as decomposable tokens
        if " " in token_repr_for_lookup.strip():
            continue

        # Check if token represents a base word in UniMorph
        is_base_form = unimorph.is_base_form(token_repr_for_lookup)
        if prefer_popular_base_conflicts and is_base_form:
            if not _candidate_allowed_for_form(token_repr_for_lookup, token_repr_for_lookup):
                is_base_form = False
        if use_english_resources and token_repr_for_lookup in unimorph_skip_lemmas:
            is_base_form = False

        # NOTE: We allow words that are both lemmas and inflections to remain bases.
        # Any conflicting inflection entries are removed later when we drop inflections
        # that are also base words.

        # For non-UniMorph Latin tokens, check if we should include them
        is_latin_token = False
        is_non_latin_token = False
        if (
            (include_non_resource_latin_tokens or include_non_latin_tokens)
            and (not token_str in processed_tokens)
            and (not first_pass)
        ):  # unimorph.is_unimorph_word(token_repr_for_lookup)
            if is_english_alphabet(token_repr):
                # IMPORTANT: Don't treat inflected forms as standalone bases
                # Check if this word is an inflected form in UniMorph's form_to_lemmas
                if token_repr_for_lookup not in unimorph.form_to_lemmas:
                    is_latin_token = True
                # If it IS in form_to_lemmas, skip it - it will be added as a variant of its lemma
            elif include_non_latin_tokens and (token_has_space_prefix or is_symbol_token):
                # For non-Latin tokens (numbers, symbols, etc.), check if we should include them
                is_non_latin_token = True
            else:
                continue

        other_positive_conditions = is_latin_token or is_non_latin_token

        if should_preserve_caps and not is_base_form:
            # Try with lowercase version
            token_repr_for_lookup = token_repr.lower()
            lowercase_is_base = unimorph.is_base_form(token_repr_for_lookup)

            if lowercase_is_base:
                is_base_form = True
                # We'll use lowercase for lookup but preserve capitalization pattern for results

        # Skip if not a base form or anything else we've decided to handle
        if not is_base_form and not other_positive_conditions:
            continue

        # Get possible parts of speech
        if use_english_resources:
            pos_set = _get_pos_from_wordnet_cached(token_repr_for_lookup)
            if not pos_set and not other_positive_conditions:
                continue
            if (
                filter_propernouns_and_aux
                and (pos_set & {"AUX", "DET"} or pos_set == {"PROPN"})
                and not other_positive_conditions
            ):
                if "PROPN" in pos_set:
                    filtered_propernouns.add(token_str)
                else:
                    filtered_aux.add(token_str)
                continue
        else:
            pos_set = set()

        # Filter special cases (e.g., stop words)
        if (
            use_english_resources
            and skip_stop_words
            and vocab_base_words_filter(token_repr_for_lookup)
        ):
            continue

        # Start building decomposition for this token
        inflections_map = {}

        has_inflections = False

        if is_base_form:
            if (
                log_second_pass_bases
                and (not first_pass)
                and (not token_has_space_prefix or not is_capitalized)
            ):
                print(f"Processing base word '{token_str}' in second pass")
            # Handle base form variations
            base_variations = get_variation_transformations(
                token_str,
                None,
                None,
                space_prefix,
                decompose_spaces,
                decompose_capitalization,
                morphology_mode=morphology_mode,
                use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                article_space_prefix=article_space_prefix_default,
                prep_space_prefix=prep_space_prefix_default,
            )

            for var, transforms in base_variations.items():
                if var not in inflections_map:
                    inflections_map[var] = transforms
                    has_inflections = True

            # Get inflections from UniMorph
            unimorph_inflections = unimorph.get_inflections(token_repr_for_lookup)

            # Use the main apply_capitalization_pattern function

            # Process UniMorph inflections
            for inflected_form, tag_sets in unimorph_inflections.items():
                if inflected_form == token_repr:
                    continue

                if _should_skip_unimorph_pair(token_repr_for_lookup, inflected_form, tag_sets):
                    continue

                if vocab_special_cases_filter(token_repr_for_lookup, inflected_form):
                    continue

                # Apply capitalization pattern if needed
                if should_preserve_caps:
                    inflected_form = apply_capitalization_pattern(
                        original_token_repr, inflected_form
                    )

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
                if (
                    use_english_resources
                    and len(tag_sets) == 1
                    and ";".join(list(tag_sets)[0]) == "V;PRS;3;SG"
                ):
                    inflected_pos_set = _get_pos_from_wordnet_cached(inflected_form)
                    if "NOUN" in inflected_pos_set:
                        tag_sets.add(("N", "PL"))

                # Disambiguate Spanish clitic duplicates by injecting ordered clitic surface tags
                if use_english_resources is False:
                    lang = (language or "").lower()
                    if lang in {"es", "spa", "spanish"} and any(
                        "PRO" in tag_set for tag_set in tag_sets
                    ):
                        clitic_seq = extract_spanish_clitic_sequence(inflected_form)
                        if clitic_seq:
                            if clitic_mode == "features":
                                clitic_tags = clitic_sequence_to_feature_tags(clitic_seq)
                            else:
                                if clitic_mode == "surface1" and len(clitic_seq) > 1:
                                    # Skip multi-clitic forms when only modeling the first clitic.
                                    continue
                                clitic_tags = clitic_sequence_to_surface_tags(
                                    clitic_seq, clitic_mode=clitic_mode
                                )
                            if clitic_tags:
                                updated_tag_sets = set()
                                for tag_set in tag_sets:
                                    if "PRO" in tag_set:
                                        updated_tag_sets.add(tuple(list(tag_set) + clitic_tags))
                                    else:
                                        updated_tag_sets.add(tag_set)
                                tag_sets = updated_tag_sets

                # Check for ambiguous inflections (after clitic injection)
                if len(tag_sets) > 1:
                    # Combine tags maintaining a consistent order
                    combined_tags = []
                    for tag_set in tag_sets:
                        tag_str = ";".join(tag_set)
                        combined_tags.append(tag_str)

                    # Sort for consistent order across words
                    tag_str = "+".join(sorted(combined_tags))

                    if tag_str == "ADJ;CMPR+V;V.PTCP;PRS":
                        # weird case, shouldn't be possible
                        print(
                            f"Inflected form '{inflected_form}' for '{token_str}' has tag_str {tag_str} - changed to 'V;V.PTCP;PRS'"
                        )
                        tag_str = "V;V.PTCP;PRS"
                    elif tag_str == "ADJ;SPRL+V;PST":
                        # weird case, shouldn't be possible
                        print(
                            f"Inflected form '{inflected_form}' for '{token_str}' has tag_str {tag_str} - changed to 'V;PST'"
                        )
                        tag_str = "V;PST"
                    elif tag_str == "V;NFIN;IMP+SBJV":
                        # weird case, shouldn't be possible
                        print(
                            f"Inflected form '{inflected_form}' for '{token_str}' has tag_str {tag_str} - skipping"
                        )
                        continue
                else:
                    tag_str = ";".join(list(tag_sets)[0])

                # Check for conflicts with other base forms
                if inflected_form.lower() in unimorph.lemma_to_forms:
                    if not _candidate_allowed_for_form(inflected_form, token_repr_for_lookup):
                        continue
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

                if morphology_mode == "atomic":
                    transform_list = [f"inflect_{tag_str}"]
                else:
                    group_mappings = [
                        map_unimorph_tags_to_compositional_groups(
                            tag_set, language, clitic_mode=clitic_mode
                        )
                        for tag_set in tag_sets
                    ]
                    if group_mappings:
                        merged_groups = (
                            merge_compositional_groups(group_mappings)
                            if len(group_mappings) > 1
                            else group_mappings[0]
                        )
                        transform_list = [
                            f"{group}_{value}"
                            for group, value in merged_groups.items()
                            if value != "none"
                        ]
                    else:
                        transform_list = []

                # Add inflection with proper space prefix
                inflected_with_prefix = (
                    space_prefix + inflected_form if token_has_space_prefix else inflected_form
                )

                # Get variations of the inflected form based on decompose_capitalization setting
                # transform_list was created above based on morphology_mode
                inflected_variations = get_variation_transformations(
                    inflected_with_prefix,
                    token_str,
                    transform_list,
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                    morphology_mode=morphology_mode,
                    use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                    article_space_prefix=article_space_prefix_default,
                    prep_space_prefix=prep_space_prefix_default,
                )

                # Add the inflection type to all variations
                for var, transforms in inflected_variations.items():
                    if var not in inflections_map:
                        inflections_map[var] = transforms
                        has_inflections = True

            # Get additional inflections from PyInflect if they're not in UniMorph
            if use_english_resources:
                pyinflect_variations = _get_pyinflect_variations_cached(
                    token_repr_for_lookup, pos_set
                )
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
                    if not _is_english_word_cached(inflected_form, use_unimorph=False):
                        continue

                    # Check for conflicts with base forms
                    if inflected_form.lower() in unimorph.lemma_to_forms:
                        if not _candidate_allowed_for_form(inflected_form, token_repr_for_lookup):
                            continue
                        inflection_baseform_conflicts.append(
                            (original_token_repr, inflected_form, tag_str)
                        )
                        # Skip if we're not overriding
                        if not override_unimorph_with_inflections:
                            continue

                    if morphology_mode == "atomic":
                        transform_list = [f"inflect_{tag_str}"]
                    else:
                        tag_sets = _split_tag_str(tag_str)
                        group_mappings = [
                            map_unimorph_tags_to_compositional_groups(
                                tag_set, language, clitic_mode=clitic_mode
                            )
                            for tag_set in tag_sets
                        ]
                        if group_mappings:
                            merged_groups = merge_compositional_groups(group_mappings)
                            transform_list = [
                                f"{group}_{value}"
                                for group, value in merged_groups.items()
                                if value != "none"
                            ]
                        else:
                            transform_list = []

                    inflected_variations = get_variation_transformations(
                        inflected_with_prefix,
                        token_str,
                        transform_list,
                        space_prefix,
                        decompose_spaces,
                        decompose_capitalization,
                        morphology_mode=morphology_mode,
                        use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                        article_space_prefix=article_space_prefix_default,
                        prep_space_prefix=prep_space_prefix_default,
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
                        # Check if this derived form already exists as a base form (handle conflicts first)
                        derived_form_conflicts_with_base = False
                        derived_form_key = _normalize_lookup_key(derived_form)
                        conflicted_tokens = base_key_to_tokens.get(derived_form_key, [])
                        conflicted_base_token = conflicted_tokens[0] if conflicted_tokens else None
                        if conflicted_base_token is not None:
                            if log_derivation_conflicts:
                                print(
                                    f"Found conflict: {derived_form} is both base form and derivation of {token_str}"
                                )
                            if conflicted_base_token in decomposition_map:
                                _remove_base_entry_from_indexes(
                                    conflicted_base_token,
                                    decomposition_map[conflicted_base_token],
                                )
                            root_existing_variants = decomposition_map.get(token_str)
                            if root_existing_variants is not None:
                                _remove_base_entry_from_indexes(token_str, root_existing_variants)
                            # Handle the conflict by converting base form to derivation
                            handle_derived_form_conflict(
                                derived_form,
                                decomposition_map,
                                token_str,
                                f"deriv_{deriv_type}",
                                base_tokens,
                                token_has_space_prefix,
                                space_prefix,
                                decompose_spaces,
                                decompose_capitalization,
                                morphology_mode=morphology_mode,
                                use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                                article_space_prefix=article_space_prefix_default,
                                prep_space_prefix=prep_space_prefix_default,
                                conflicted_base_token=conflicted_base_token,
                            )
                            if token_str in decomposition_map:
                                _index_base_entry(token_str, decomposition_map[token_str])
                            _record_derivation_decision(
                                "base_derivation_conflicts_handled",
                                f"{derived_form} -> {token_str} (conflicted_base={conflicted_base_token})",
                            )
                            derived_form_conflicts_with_base = True
                            has_inflections = True

                        if derived_form_conflicts_with_base:
                            continue

                        # Check for chained derivations (only after handling conflicts)
                        # A chained derivation means: BaseA -> DerivedForm (as derivation), and now DerivedForm -> CurrentToken
                        # We should only skip if DerivedForm exists as a derivation of a DIFFERENT base (not current token)
                        if not allow_chained_derivations:
                            current_base_key = _normalize_lookup_key(token_str)
                            existing_bases = derivation_key_to_bases.get(derived_form_key, [])
                            existing_base_keys = {
                                _normalize_lookup_key(base_form) for base_form in existing_bases
                            }
                            if existing_base_keys and current_base_key not in existing_base_keys:
                                roots_with_derived, covered_roots = _classify_derivation_overlap(
                                    token_str,
                                    derived_form,
                                    existing_base_keys,
                                )
                                if roots_with_derived:
                                    if covered_roots:
                                        covered_root = sorted(covered_roots)[0]
                                        if log_derivation_conflicts:
                                            print(
                                                "Skipping derivation+inflection (covered): "
                                                f"{covered_root} -> {derived_form} -> {token_str}"
                                            )
                                        _record_derivation_decision(
                                            "deriv_plus_inflection_skipped_covered",
                                            (
                                                f"{covered_root} -> {derived_form} -> {token_str} "
                                                f"(existing={sorted(existing_base_keys)[:3]})"
                                            ),
                                        )
                                        continue
                                    _record_derivation_decision(
                                        "deriv_plus_inflection_allowed_uncovered",
                                        (
                                            f"{token_str} + deriv({derived_form}) via roots={sorted(roots_with_derived)[:3]} "
                                            f"(existing={sorted(existing_base_keys)[:3]})"
                                        ),
                                    )
                                else:
                                    chain_base_form = next(
                                        (
                                            base_form
                                            for base_form in existing_bases
                                            if _normalize_lookup_key(base_form) != current_base_key
                                        ),
                                        None,
                                    )
                                    if log_derivation_conflicts:
                                        print(
                                            f"Skipping chained derivation: {chain_base_form} -> {derived_form} -> {token_str}"
                                        )
                                    _record_derivation_decision(
                                        "chained_derivation_skipped",
                                        f"{chain_base_form} -> {derived_form} -> {token_str}",
                                    )
                                    continue

                        # Check if this derived form already exists as an inflection
                        derived_with_prefix = (
                            space_prefix + derived_form if token_has_space_prefix else derived_form
                        )
                        already_exists_as_inflection = False
                        for existing_var in inflections_map.keys():
                            if existing_var.lower() == derived_with_prefix.lower():
                                # Check if it has an inflection transform
                                existing_transforms = inflections_map[existing_var]
                                if morphology_mode == "atomic":
                                    has_inflection_transform = any(
                                        t.startswith("inflect_") for t in existing_transforms
                                    )
                                else:
                                    has_inflection_transform = any(
                                        _get_morph_group_from_transform(t)
                                        for t in existing_transforms
                                    )
                                if has_inflection_transform:
                                    already_exists_as_inflection = True
                                    break

                        if already_exists_as_inflection:
                            continue

                        # Check for conflicts with base forms in unimorph
                        if derived_form.lower() in unimorph.lemma_to_forms:
                            if not _candidate_allowed_for_form(derived_form, token_repr_for_lookup):
                                continue
                            inflection_baseform_conflicts.append(
                                (original_token_repr, derived_form, deriv_type)
                            )
                            # Skip if we're not overriding
                            if not override_unimorph_with_inflections:
                                continue

                        # Get variations of the derived form based on decompose_capitalization setting
                        derived_variations = get_variation_transformations(
                            derived_with_prefix,
                            token_str,
                            [f"deriv_{deriv_type}"],
                            space_prefix,
                            decompose_spaces,
                            decompose_capitalization,
                            morphology_mode=morphology_mode,
                            use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                            article_space_prefix=article_space_prefix_default,
                            prep_space_prefix=prep_space_prefix_default,
                        )

                        # Add the derivation type to all variations
                        for var, transforms in derived_variations.items():
                            if var not in inflections_map:
                                inflections_map[var] = transforms
                                has_inflections = True

                        # Now check for inflections of this derived form
                        derived_inflections = unimorph.get_inflections(derived_form)
                        for inflected_derived_form, derived_tag_sets in derived_inflections.items():
                            if inflected_derived_form == derived_form:
                                continue

                            # Apply capitalization pattern if needed
                            if should_preserve_caps:
                                inflected_derived_form = apply_capitalization_pattern(
                                    original_token_repr, inflected_derived_form
                                )

                            # Process tag sets for derived form inflections
                            for derived_tag_set in derived_tag_sets:
                                derived_tag_str = ";".join(derived_tag_set)

                                if morphology_mode == "atomic":
                                    combined_transforms = [
                                        f"deriv_{deriv_type}",
                                        f"inflect_{derived_tag_str}",
                                    ]
                                else:
                                    groups = map_unimorph_tags_to_compositional_groups(
                                        derived_tag_set, language, clitic_mode=clitic_mode
                                    )
                                    morph_transforms = [
                                        f"{group}_{value}"
                                        for group, value in groups.items()
                                        if value != "none"
                                    ]
                                    combined_transforms = [f"deriv_{deriv_type}"] + morph_transforms

                                # Add inflection of derivation with proper space prefix
                                inflected_derived_with_prefix = (
                                    space_prefix + inflected_derived_form
                                    if token_has_space_prefix
                                    else inflected_derived_form
                                )

                                # Get variations of the inflected derived form
                                inflected_derived_variations = get_variation_transformations(
                                    inflected_derived_with_prefix,
                                    token_str,
                                    combined_transforms,
                                    space_prefix,
                                    decompose_spaces,
                                    decompose_capitalization,
                                    morphology_mode=morphology_mode,
                                    use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                                    article_space_prefix=article_space_prefix_default,
                                    prep_space_prefix=prep_space_prefix_default,
                                )

                                # Add to inflections map
                                for var, transforms in inflected_derived_variations.items():
                                    if var not in inflections_map:
                                        inflections_map[var] = transforms
                                        has_inflections = True

                        # Also check PyInflect for inflections of derived form
                        if use_english_resources:
                            derived_pos_set = _get_pos_from_wordnet_cached(derived_form)
                            pyinflect_derived_variations = _get_pyinflect_variations_cached(
                                derived_form, derived_pos_set
                            )
                            for (
                                inflected_derived_form,
                                derived_tag_str,
                            ) in pyinflect_derived_variations.items():
                                # Apply capitalization pattern if needed
                                if should_preserve_caps:
                                    inflected_derived_form = apply_capitalization_pattern(
                                        original_token_repr, inflected_derived_form
                                    )

                                # Skip if already in inflections map (from unimorph)
                                inflected_derived_with_prefix = (
                                    space_prefix + inflected_derived_form
                                    if token_has_space_prefix
                                    else inflected_derived_form
                                )
                                if any(
                                    var.lower() == inflected_derived_with_prefix.lower()
                                    for var in inflections_map
                                ):
                                    continue

                                # Check if it's a valid English word
                                if not _is_english_word_cached(
                                    inflected_derived_form, use_unimorph=False
                                ):
                                    continue

                                if morphology_mode == "atomic":
                                    combined_transforms = [
                                        f"deriv_{deriv_type}",
                                        f"inflect_{derived_tag_str}",
                                    ]
                                else:
                                    tag_sets = _split_tag_str(derived_tag_str)
                                    group_mappings = [
                                        map_unimorph_tags_to_compositional_groups(
                                            tag_set, language, clitic_mode=clitic_mode
                                        )
                                        for tag_set in tag_sets
                                    ]
                                    if group_mappings:
                                        merged_groups = merge_compositional_groups(group_mappings)
                                        morph_transforms = [
                                            f"{group}_{value}"
                                            for group, value in merged_groups.items()
                                            if value != "none"
                                        ]
                                    else:
                                        morph_transforms = []
                                    combined_transforms = [f"deriv_{deriv_type}"] + morph_transforms

                                # Get variations of the inflected derived form
                                inflected_derived_variations = get_variation_transformations(
                                    inflected_derived_with_prefix,
                                    token_str,
                                    combined_transforms,
                                    space_prefix,
                                    decompose_spaces,
                                    decompose_capitalization,
                                    morphology_mode=morphology_mode,
                                    use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                                    article_space_prefix=article_space_prefix_default,
                                    prep_space_prefix=prep_space_prefix_default,
                                )

                                # Add to inflections map
                                for var, transforms in inflected_derived_variations.items():
                                    if var not in inflections_map:
                                        inflections_map[var] = transforms
                                        has_inflections = True

                        # Check for chained derivations if enabled
                        if use_english_resources and allow_chained_derivations:
                            if log_derivation_conflicts:
                                print(f"Checking for chained derivations from {derived_form}...")
                            chained_results = get_chained_derivations(
                                derived_form,
                                token_str,
                                [f"deriv_{deriv_type}"],
                                unimorph,
                                space_prefix,
                                decompose_spaces,
                                decompose_capitalization,
                                should_preserve_caps,
                                original_token_repr,
                                token_has_space_prefix,
                                max_depth=2,
                                use_english_resources=use_english_resources,
                                language=language,
                                morphology_mode=morphology_mode,
                                clitic_mode=clitic_mode,
                                use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                                article_space_prefix=article_space_prefix_default,
                                prep_space_prefix=prep_space_prefix_default,
                            )

                            for chained_form, deriv_chain, chained_variations in chained_results:
                                if log_derivation_conflicts:
                                    print(
                                        f"Found chained derivation: {token_str} -> {' -> '.join([d.replace('deriv_', '') for d in deriv_chain if d.startswith('deriv_')])} -> {chained_form}"
                                    )

                                # Add all variations of this chained derivation
                                for var, transforms in chained_variations.items():
                                    if var not in inflections_map:
                                        inflections_map[var] = transforms
                                        has_inflections = True
                                    else:
                                        # Check if this is a better/more complete analysis
                                        existing_transforms = inflections_map[var]
                                        # If the new analysis has more derivation types, use it
                                        new_deriv_count = len(
                                            [t for t in transforms if t.startswith("deriv_")]
                                        )
                                        existing_deriv_count = len(
                                            [
                                                t
                                                for t in existing_transforms
                                                if t.startswith("deriv_")
                                            ]
                                        )
                                        if new_deriv_count > existing_deriv_count:
                                            if log_derivation_conflicts:
                                                print(
                                                    f"Replacing analysis for {var}: {existing_transforms} -> {transforms}"
                                                )
                                            inflections_map[var] = transforms
        elif other_positive_conditions:
            # For Latin tokens that aren't in UniMorph, just add capitalization and space-prefix variations
            if is_latin_token:
                # Handle base form variations
                base_transforms = [NA_DERIVATION]
                if morphology_mode == "atomic":
                    base_transforms.insert(0, NA_INFLECTION)
                base_variations = get_variation_transformations(
                    token_str,
                    None,
                    base_transforms,
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                    morphology_mode=morphology_mode,
                    use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                    article_space_prefix=article_space_prefix_default,
                    prep_space_prefix=prep_space_prefix_default,
                )

                for var, transforms in base_variations.items():
                    if var not in inflections_map:
                        inflections_map[var] = transforms
                        has_inflections = True

            # For non-Latin tokens (numbers, symbols, etc.), only add space prefix variations
            elif is_non_latin_token:
                # Only create space prefix variations, use NA for all morphological groups
                if token_has_space_prefix:
                    # Token has space prefix, create version without it
                    token_without_prefix = token_str[len(space_prefix) :]
                    if token_without_prefix:
                        base_transforms = [
                            NA_DERIVATION,
                            NO_SPACE_PREFIX_TRANSFORM,
                            NA_CAPITALIZATION_TRANSFORM,
                        ]
                        if morphology_mode == "atomic":
                            base_transforms.insert(0, NA_INFLECTION)
                        inflections_map[token_without_prefix] = base_transforms

                    # Also add the original with space prefix
                    base_transforms = [
                        NA_DERIVATION,
                        WITH_SPACE_PREFIX_TRANSFORM,
                        NA_CAPITALIZATION_TRANSFORM,
                    ]
                    if morphology_mode == "atomic":
                        base_transforms.insert(0, NA_INFLECTION)
                    inflections_map[token_str] = base_transforms

                    has_inflections = True
                else:
                    # Token doesn't have space prefix, create version with it
                    token_with_prefix = space_prefix + token_str
                    base_transforms = [
                        NA_DERIVATION,
                        WITH_SPACE_PREFIX_TRANSFORM,
                        NA_CAPITALIZATION_TRANSFORM,
                    ]
                    if morphology_mode == "atomic":
                        base_transforms.insert(0, NA_INFLECTION)
                    inflections_map[token_with_prefix] = base_transforms

                    # Also add the original without space prefix
                    base_transforms = [
                        NA_DERIVATION,
                        NO_SPACE_PREFIX_TRANSFORM,
                        NA_CAPITALIZATION_TRANSFORM,
                    ]
                    if morphology_mode == "atomic":
                        base_transforms.insert(0, NA_INFLECTION)
                    inflections_map[token_str] = base_transforms

                    has_inflections = True

        else:
            pass

        # Check if we have inflections
        if not has_inflections and not include_base_only:
            continue

        # Convert NO types to NA types for words that never have actual inflections/derivations
        # Check if this base word only ever uses NO or NA types for inflection/derivation groups
        if inflections_map:
            inflection_types = set()
            derivation_types = set()

            for transforms in inflections_map.values():
                for transform in transforms:
                    if transform.startswith("inflect_") or transform == NO_INFLECTION:
                        inflection_types.add(transform)
                    elif transform.startswith("deriv_") or transform == NO_DERIVATION:
                        derivation_types.add(transform)

            # Check if only NO/NA types are used for each group
            has_only_no_na_inflections = inflection_types.issubset({NO_INFLECTION, NA_INFLECTION})
            has_only_no_na_derivations = derivation_types.issubset({NO_DERIVATION, NA_DERIVATION})

            # Convert NO to NA if only NO/NA types exist for a group
            if has_only_no_na_inflections and NO_INFLECTION in inflection_types:
                for form, transforms in inflections_map.items():
                    inflections_map[form] = [
                        NA_INFLECTION if t == NO_INFLECTION else t for t in transforms
                    ]

            if has_only_no_na_derivations and NO_DERIVATION in derivation_types:
                for form, transforms in inflections_map.items():
                    inflections_map[form] = [
                        NA_DERIVATION if t == NO_DERIVATION else t for t in transforms
                    ]

        # Add to decomposition map
        decomposition_map[token_str] = {}
        transform_to_form = {}
        drop_sentinel = object()
        for form, transforms in inflections_map.items():
            if "N;SG" in transforms:
                continue
            # Format types as list and string
            if merge_plural_and_present_singular:
                transforms = list(
                    map(
                        lambda x: {
                            f"inflect_N;PL": f"inflect_N;PL+V;PRS;3;SG",
                            f"inflect_V;PRS;3;SG": f"inflect_N;PL+V;PRS;3;SG",
                        }.get(x, x),
                        transforms,
                    )
                )
            types_list = list(
                dict.fromkeys(transforms)
            )  # remove duplicates while keeping order of list
            types_str = "+".join(sorted(transforms))
            transform_key = tuple(sorted(types_list))

            if transform_collision_mode != "off":
                existing_form = transform_to_form.get(transform_key)
                if existing_form is None:
                    transform_to_form[transform_key] = form
                elif existing_form is drop_sentinel:
                    filtered_transform_collisions.append(
                        (token_str, None, form, types_list, transform_collision_mode)
                    )
                    continue
                elif existing_form != form:
                    filtered_transform_collisions.append(
                        (token_str, existing_form, form, types_list, transform_collision_mode)
                    )
                    if transform_collision_mode == "drop_both":
                        if existing_form in decomposition_map[token_str]:
                            del decomposition_map[token_str][existing_form]
                            processed_tokens.discard(existing_form)
                        transform_to_form[transform_key] = drop_sentinel
                    continue

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

            # Populate type_to_words for the current form only (avoid O(k^2) rescans).
            if use_type_str_as_type:
                type_to_words[types_list[0]].add(form)
            else:
                for type_tag in types_list:
                    type_to_words[type_tag].add(form)

        _index_base_entry(token_str, decomposition_map[token_str])

        # Add to base tokens
        base_tokens[token_str] = token_id

    # Prefer regular plural/3sg inflections over treating those forms as standalone bases.
    if use_english_resources:
        plural_like_tags = {
            "inflect_N;PL",
            "inflect_V;PRS;3;SG",
            "inflect_N;PL+V;PRS;3;SG",
        }
        inflection_bases_to_drop = set()
        for base_word, inflections in decomposition_map.items():
            for inflected_form, transforms in inflections.items():
                if inflected_form.lower() == base_word:
                    continue
                if inflected_form.lower() in decomposition_map:
                    if any(t in plural_like_tags for t in transforms):
                        inflection_bases_to_drop.add(inflected_form.lower())

        if inflection_bases_to_drop:
            for base_word in list(decomposition_map.keys()):
                if base_word.lower() in inflection_bases_to_drop:
                    removed_variants = decomposition_map.get(base_word, {})
                    _remove_base_entry_from_indexes(base_word, removed_variants)
                    decomposition_map.pop(base_word, None)
                    base_tokens.pop(base_word, None)

    if prefer_popular_base_conflicts:
        variant_to_bases: DefaultDict[str, List[str]] = defaultdict(list)
        for base_word, inflections in decomposition_map.items():
            for variant, transforms in inflections.items():
                if variant.lower() == base_word.lower():
                    continue
                if not any(is_morph_transform_name(t) for t in transforms):
                    continue
                variant_to_bases[_normalize_lookup_key(variant)].append(base_word)

        removed_variant_mappings = 0
        bases_to_drop: Set[str] = set()
        for variant_key, base_candidates in variant_to_bases.items():
            unique_candidates = list(dict.fromkeys(base_candidates))
            if len(unique_candidates) <= 1:
                continue
            best_base = min(
                unique_candidates,
                key=lambda base_word: (
                    -_lemma_popularity(_normalize_lookup_key(base_word)),
                    len(_normalize_lookup_key(base_word)),
                    _normalize_lookup_key(base_word),
                ),
            )
            best_base_key = _normalize_lookup_key(best_base)
            best_base_popularity = _lemma_popularity(best_base_key)
            for base_word in unique_candidates:
                if base_word == best_base:
                    continue
                base_variants = decomposition_map.get(base_word, {})
                for variant in list(base_variants.keys()):
                    if _normalize_lookup_key(variant) != variant_key:
                        continue
                    if not any(is_morph_transform_name(t) for t in base_variants[variant]):
                        continue
                    del base_variants[variant]
                    removed_variant_mappings += 1
                base_word_key = _normalize_lookup_key(base_word)
                if (
                    base_word_key == variant_key
                    and _lemma_popularity(base_word_key) < best_base_popularity
                ):
                    bases_to_drop.add(base_word)

        if bases_to_drop:
            for base_word in sorted(bases_to_drop):
                removed_variants = decomposition_map.get(base_word, {})
                _remove_base_entry_from_indexes(base_word, removed_variants)
                decomposition_map.pop(base_word, None)
                base_tokens.pop(base_word, None)

        if removed_variant_mappings or bases_to_drop:
            _rebuild_conflict_indexes()
            print(
                "Popular-base conflict resolution: "
                f"removed {removed_variant_mappings} variant mappings, "
                f"dropped {len(bases_to_drop)} weaker base forms"
            )

    # Clean inflections that collide with existing base words using the final base set.
    # This runs after base-pruning to avoid permanently dropping variants that only
    # conflicted with bases that were removed by conflict resolution.
    #
    # Important: use exact-form collisions only. Lowercasing here is too aggressive
    # because it can drop case/space variants (e.g. " Whales") just because a
    # lowercase base (" whales") exists, even when that variant is needed.
    cleaned_inflection_conflicts = False
    for base_word, inflections in decomposition_map.items():
        replace_inflections = False
        valid_inflections = dict()
        for inflected_form, transforms in inflections.items():
            if inflected_form != base_word and inflected_form in decomposition_map:
                replace_inflections = True
            else:
                valid_inflections[inflected_form] = transforms
        if replace_inflections:
            decomposition_map[base_word] = valid_inflections
            cleaned_inflection_conflicts = True

    if cleaned_inflection_conflicts:
        _rebuild_conflict_indexes()

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
    if transform_collision_mode != "off":
        print(
            f"Filtered {len(filtered_transform_collisions)} inflections due to transform collisions"
        )
    print(f"Filtered {len(filtered_propernouns)} propernoun tokens")
    print(f"Filtered {len(filtered_aux)} AUX tokens")
    print(f"Converted {len(converted_titlecase_bases)} titlecase bases to lowercase aliases")
    if getattr(unimorph, "filtered_possessive_inflections", None):
        print(
            f"Filtered {len(unimorph.filtered_possessive_inflections)} possessive pronoun inflections"
        )

    # ==============================================================================
    # Phase 3: Process multi-token UniMorph lemmas directly
    # ==============================================================================
    # This phase handles UniMorph lemmas that tokenize to multiple tokens,
    # ensuring they're added to the decomposition map with their inflections.
    # Example: "tremble" (multi-token) with inflection "trembling" (single token)

    if not skip_multi_token_bases:
        print("\nProcessing multi-token UniMorph lemmas...")

        # Determine max token limit
        if skip_three_token_words:
            effective_max_tokens = 2
        elif skip_four_token_words:
            effective_max_tokens = 3
        elif max_tokens_per_base_word is not None:
            effective_max_tokens = max_tokens_per_base_word
        else:
            effective_max_tokens = None  # Unlimited

        multitoken_lemmas_processed = 0
        multitoken_lemmas_skipped_limit = 0
        multitoken_lemmas_skipped_already_added = 0
        unimorph_lemma_lookup = set(unimorph.lemma_to_forms.keys())

        # Get UniMorph lemmas
        unimorph_lemmas = list(unimorph.lemma_to_forms.keys())

        # Limit for smoke testing if requested
        if max_vocab_items is not None:
            unimorph_lemmas = unimorph_lemmas[:max_vocab_items]
            print(f"  SMOKE TEST MODE: Limited UniMorph lemmas to {len(unimorph_lemmas)} items")

        # Iterate over all UniMorph lemmas
        for lemma in tqdm(unimorph_lemmas, desc="UniMorph lemmas"):
            # Skip multi-word expressions (e.g., "door to door")
            # These should not be treated as decomposable bases
            if " " in lemma.strip():
                continue

            # Apply space prefix if needed
            if decompose_spaces:
                lemma_with_prefix = space_prefix + lemma
            else:
                lemma_with_prefix = lemma

            # Check if already in decomposition_map (from vocabulary processing)
            if lemma_with_prefix in decomposition_map or lemma in decomposition_map:
                multitoken_lemmas_skipped_already_added += 1
                continue

            # Tokenize the lemma
            lemma_tokens = _encode_cached(lemma_with_prefix)
            num_tokens = len(lemma_tokens)

            # Skip single-token lemmas (already handled in vocabulary processing)
            if num_tokens <= 1:
                continue

            # Check token count limit
            if effective_max_tokens is not None and num_tokens > effective_max_tokens:
                multitoken_lemmas_skipped_limit += 1
                continue

            # Apply same filters as main loop
            # Skip if no space prefix when required
            if (
                decompose_spaces
                and skip_if_no_space_prefix
                and not lemma_with_prefix.startswith(space_prefix)
            ):
                continue

            # Check POS
            if use_english_resources:
                pos_set = _get_pos_from_wordnet_cached(lemma)
                if not pos_set:
                    continue

                # Filter propernouns and aux
                if filter_propernouns_and_aux and (
                    pos_set & {"AUX", "DET"} or pos_set == {"PROPN"}
                ):
                    continue

                # Skip stop words
                if skip_stop_words and vocab_base_words_filter(lemma):
                    continue

            # All-caps lemmas should not be treated as base forms.
            if lemma.isupper():
                continue

            # Build inflections map for this multi-token lemma
            inflections_map = {}
            first_seen_lower_has_inflection_transform: Dict[str, bool] = {}
            lemma_has_space_prefix = lemma_with_prefix.startswith(space_prefix)

            def _add_inflection_variant(var: str, transforms: List[str]) -> None:
                if var in inflections_map:
                    return
                inflections_map[var] = transforms
                lower_key = var.lower()
                if lower_key not in first_seen_lower_has_inflection_transform:
                    if morphology_mode == "atomic":
                        has_inflection_transform = any(t.startswith("inflect_") for t in transforms)
                    else:
                        has_inflection_transform = any(
                            _get_morph_group_from_transform(t) for t in transforms
                        )
                    first_seen_lower_has_inflection_transform[lower_key] = has_inflection_transform

            # Add base form variations
            base_variations = get_variation_transformations(
                lemma_with_prefix,
                None,
                None,
                space_prefix,
                decompose_spaces,
                decompose_capitalization,
                morphology_mode=morphology_mode,
                use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                article_space_prefix=article_space_prefix_default,
                prep_space_prefix=prep_space_prefix_default,
            )

            for var, transforms in base_variations.items():
                _add_inflection_variant(var, transforms)

            # Get inflections from UniMorph
            unimorph_inflections = unimorph.get_inflections(lemma)

            # Process UniMorph inflections (same logic as main loop)
            for inflected_form, tag_sets in unimorph_inflections.items():
                if inflected_form == lemma:
                    continue

                if _should_skip_unimorph_pair(lemma, inflected_form, tag_sets):
                    continue

                if vocab_special_cases_filter(lemma, inflected_form):
                    continue

                # Remove problematic tags
                if ("V", "NFIN", "IMP+SBJV") in tag_sets and len(tag_sets) > 1:
                    tag_sets.remove(("V", "NFIN", "IMP+SBJV"))

                if ("V", "V.PTCP", "PST") in tag_sets and ("V", "PST") in tag_sets:
                    tag_sets.remove(("V", "V.PTCP", "PST"))

                # Handle N;PL+V;PRS;3;SG ambiguity
                if (
                    use_english_resources
                    and len(tag_sets) == 1
                    and ";".join(list(tag_sets)[0]) == "V;PRS;3;SG"
                ):
                    inflected_pos_set = _get_pos_from_wordnet_cached(inflected_form)
                    if "NOUN" in inflected_pos_set:
                        tag_sets.add(("N", "PL"))

                # Disambiguate Spanish clitic duplicates by injecting ordered clitic surface tags
                if use_english_resources is False:
                    lang = (language or "").lower()
                    if lang in {"es", "spa", "spanish"} and any(
                        "PRO" in tag_set for tag_set in tag_sets
                    ):
                        clitic_seq = extract_spanish_clitic_sequence(inflected_form)
                        if clitic_seq:
                            if clitic_mode == "features":
                                clitic_tags = clitic_sequence_to_feature_tags(clitic_seq)
                            else:
                                if clitic_mode == "surface1" and len(clitic_seq) > 1:
                                    continue
                                clitic_tags = clitic_sequence_to_surface_tags(
                                    clitic_seq, clitic_mode=clitic_mode
                                )
                            if clitic_tags:
                                updated_tag_sets = set()
                                for tag_set in tag_sets:
                                    if "PRO" in tag_set:
                                        updated_tag_sets.add(tuple(list(tag_set) + clitic_tags))
                                    else:
                                        updated_tag_sets.add(tag_set)
                                tag_sets = updated_tag_sets

                # Build tag string
                if len(tag_sets) > 1:
                    combined_tags = [";".join(tag_set) for tag_set in tag_sets]
                    tag_str = "+".join(sorted(combined_tags))
                else:
                    tag_str = ";".join(list(tag_sets)[0])

                # Handle merge_plural_and_present_singular
                if merge_plural_and_present_singular and tag_str == "N;PL+V;PRS;3;SG":
                    tag_str = "N;PL+V;PRS;3;SG"

                # Add inflection with proper space prefix
                inflected_with_prefix = (
                    space_prefix + inflected_form if lemma_has_space_prefix else inflected_form
                )

                # Build transform list based on morphology_mode
                if morphology_mode == "atomic":
                    transform_list = [f"inflect_{tag_str}"]
                else:
                    # COMPOSITIONAL MODE: Map tag_sets to compositional groups
                    group_mappings = [
                        map_unimorph_tags_to_compositional_groups(
                            tag_set, language, clitic_mode=clitic_mode
                        )
                        for tag_set in tag_sets
                    ]

                    # Merge if multiple analyses (homophony)
                    merged_groups = (
                        merge_compositional_groups(group_mappings)
                        if len(group_mappings) > 1
                        else group_mappings[0]
                    )

                    # Convert to transform strings: "group_value"
                    transform_list = [
                        f"{group}_{value}"
                        for group, value in merged_groups.items()
                        if value != "none"
                    ]

                # Get variations of the inflected form
                inflected_variations = get_variation_transformations(
                    inflected_with_prefix,
                    lemma_with_prefix,
                    transform_list,
                    space_prefix,
                    decompose_spaces,
                    decompose_capitalization,
                    morphology_mode=morphology_mode,
                    use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                    article_space_prefix=article_space_prefix_default,
                    prep_space_prefix=prep_space_prefix_default,
                )

                for var, transforms in inflected_variations.items():
                    _add_inflection_variant(var, transforms)

            # Add derivations if enabled (same logic as main loop for include_derivations)
            if include_derivations:
                # Get derivations from UniMorph
                unimorph_derivations = unimorph.get_root_derivations(lemma)

                for derived_form, deriv_types in unimorph_derivations.items():
                    if derived_form == lemma:
                        continue

                    for deriv_type, _pos in deriv_types:
                        # Skip chained derivations if not allowed
                        if not allow_chained_derivations:
                            derived_form_key = _normalize_lookup_key(derived_form)
                            current_base_key = _normalize_lookup_key(lemma_with_prefix)
                            existing_bases = derivation_key_to_bases.get(derived_form_key, [])
                            existing_base_keys = {
                                _normalize_lookup_key(base_form) for base_form in existing_bases
                            }
                            if existing_base_keys and current_base_key not in existing_base_keys:
                                roots_with_derived, covered_roots = _classify_derivation_overlap(
                                    lemma_with_prefix,
                                    derived_form,
                                    existing_base_keys,
                                )
                                if roots_with_derived:
                                    if covered_roots:
                                        covered_root = sorted(covered_roots)[0]
                                        _record_derivation_decision(
                                            "deriv_plus_inflection_skipped_covered",
                                            (
                                                f"{covered_root} -> {derived_form} -> {lemma_with_prefix} "
                                                f"(multi-token base)"
                                            ),
                                        )
                                        continue
                                    _record_derivation_decision(
                                        "deriv_plus_inflection_allowed_uncovered",
                                        (
                                            f"{lemma_with_prefix} + deriv({derived_form}) via roots="
                                            f"{sorted(roots_with_derived)[:3]} (multi-token base)"
                                        ),
                                    )
                                else:
                                    _record_derivation_decision(
                                        "chained_derivation_skipped",
                                        f"{sorted(existing_base_keys)[:1]} -> {derived_form} -> {lemma_with_prefix} (multi-token base)",
                                    )
                                    continue

                        # Add derivation with proper space prefix
                        derived_with_prefix = (
                            space_prefix + derived_form
                            if lemma_with_prefix.startswith(space_prefix)
                            else derived_form
                        )

                        # Check conflicts with inflections
                        if first_seen_lower_has_inflection_transform.get(
                            derived_with_prefix.lower(), False
                        ):
                            continue

                        # Check conflicts with base forms
                        if derived_form.lower() in unimorph_lemma_lookup:
                            if not _candidate_allowed_for_form(derived_form, lemma):
                                continue
                            inflection_baseform_conflicts.append((lemma, derived_form, deriv_type))
                            if not override_unimorph_with_inflections:
                                continue

                        # Get variations of the derived form
                        derived_variations = get_variation_transformations(
                            derived_with_prefix,
                            lemma_with_prefix,
                            [f"deriv_{deriv_type}"],
                            space_prefix,
                            decompose_spaces,
                            decompose_capitalization,
                            morphology_mode=morphology_mode,
                            use_relative_space_cap_transforms=use_relative_space_cap_transforms,
                            article_space_prefix=article_space_prefix_default,
                            prep_space_prefix=prep_space_prefix_default,
                        )

                        for var, transforms in derived_variations.items():
                            _add_inflection_variant(var, transforms)

            # Only add to decomposition_map if we have inflections
            if inflections_map:
                decomposition_map[lemma_with_prefix] = inflections_map
                _index_base_entry(lemma_with_prefix, inflections_map)
                multitoken_lemmas_processed += 1

                # Add to base_tokens with tuple of token IDs for multi-token bases
                # This is needed by extend_tokenizer_by_decomposition_map
                base_tokens[lemma_with_prefix] = lemma_tokens

                # Mark as processed
                for form in inflections_map.keys():
                    processed_tokens.add(form)

                # Populate type_to_words
                for form, types in inflections_map.items():
                    if use_type_str_as_type:
                        type_to_words[types[0]].add(form)
                    else:
                        for type_tag in types:
                            type_to_words[type_tag].add(form)

        print(f"Processed {multitoken_lemmas_processed} multi-token UniMorph lemmas")
        print(f"  Skipped {multitoken_lemmas_skipped_already_added} (already in decomposition_map)")
        print(f"  Skipped {multitoken_lemmas_skipped_limit} (exceeded token limit)")
        if effective_max_tokens:
            print(f"  Token limit: {effective_max_tokens}")

    # ==============================================================================
    # Generate tokens with punctuation, articles, and prepositions (new transformations)
    # ==============================================================================
    if base_tokens:
        removed_multitoken_bases = []
        for base_word, inflections in list(decomposition_map.items()):
            base_token_id = base_tokens.get(base_word)
            if not isinstance(base_token_id, tuple):
                continue
            for variant_str, transforms in inflections.items():
                if variant_str == base_word:
                    continue
                has_morph = False
                for t in transforms:
                    if t.startswith("inflect_") or t.startswith("deriv_"):
                        has_morph = True
                        break
                    group = _get_morph_group_from_transform(t)
                    if group and t not in {
                        MORPHOLOGICAL_GROUP_NO_VALUES[group],
                        MORPHOLOGICAL_GROUP_NA_VALUES[group],
                    }:
                        has_morph = True
                        break
                if not has_morph:
                    continue
                variant_ids = _encode_variant_cached(variant_str)
                if len(variant_ids) == 1:
                    removed_multitoken_bases.append((base_word, variant_str))
                    break

        if removed_multitoken_bases:
            for base_word, _ in removed_multitoken_bases:
                decomposition_map.pop(base_word, None)
                base_tokens.pop(base_word, None)
            sample = ", ".join(f"{b}->{v}" for b, v in removed_multitoken_bases[:5])
            print(
                f"Filtered {len(removed_multitoken_bases)} multi-token bases with single-token inflections/derivations (e.g. {sample})"
            )

    if decompose_punctuation or decompose_articles or decompose_prepositions:
        print("\nGenerating tokens with punctuation, articles, and prepositions...")

        if multitoken_allowed_groups is None:
            multitoken_allowed_groups = []

        if modifier_generation_mode is None:
            modifier_generation_mode = "full"
        modifier_generation_mode = modifier_generation_mode.lower()
        if modifier_generation_mode not in {"full", "vocab_only"}:
            raise ValueError(f"Unknown modifier_generation_mode: {modifier_generation_mode}")

        space_transform_values = {
            NO_SPACE_PREFIX_TRANSFORM,
            WITH_SPACE_PREFIX_TRANSFORM,
            REMOVE_SPACE_PREFIX_TRANSFORM,
            NA_SPACE_PREFIX_TRANSFORM,
        }
        base_cap_transform_values = {
            ADD_BASE_CAPITALIZATION_TRANSFORM,
            ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
            NO_BASE_CAPITALIZATION_TRANSFORM,
            REMOVE_BASE_CAPITALIZATION_TRANSFORM,
            NA_BASE_CAPITALIZATION_TRANSFORM,
        }

        def _extract_core_transform_values(existing_transforms):
            inflection_val = NO_INFLECTION
            derivation_val = NO_DERIVATION
            space_val = NO_SPACE_PREFIX_TRANSFORM
            base_cap_val = NO_BASE_CAPITALIZATION_TRANSFORM

            for t in existing_transforms:
                if t.startswith("inflect_") or t in [NO_INFLECTION, NA_INFLECTION]:
                    inflection_val = t
                elif t.startswith("deriv_") or t in [NO_DERIVATION, NA_DERIVATION]:
                    derivation_val = t
                elif t in space_transform_values:
                    space_val = t
                elif t in base_cap_transform_values:
                    base_cap_val = t
            return inflection_val, derivation_val, space_val, base_cap_val

        punct_article_prep_generated = 0

        if modifier_generation_mode == "vocab_only":
            # Fast path: only add modifiers for tokens that already exist in the vocab.
            variant_to_entries = defaultdict(list)
            for base_token_key, variants_dict in decomposition_map.items():
                for variant_token_str, existing_transforms in variants_dict.items():
                    variant_to_entries[variant_token_str].append(
                        (base_token_key, existing_transforms)
                    )

            preposition_words = preposition_list or [
                "by",
                "at",
                "of",
                "to",
                "in",
                "on",
                "with",
                "for",
                "from",
            ]
            vocab = tokenizer.get_vocab()
            for token_str in tqdm(
                vocab.keys(), desc="Scanning vocab for modifier tokens", total=len(vocab)
            ):
                normalized_token_str = token_str
                if normalized_token_str.startswith("Ġ"):
                    normalized_token_str = " " + normalized_token_str[1:]
                base_text, transforms = strip_all_affixes(
                    normalized_token_str,
                    space_prefix=space_prefix,
                    decompose_punctuation=decompose_punctuation,
                    decompose_articles=decompose_articles,
                    decompose_prepositions=decompose_prepositions,
                    prepositions=preposition_words,
                    track_article_prep_space_prefix=decompose_article_prep_space_prefix,
                )

                if (
                    transforms["prefix_punct"] is None
                    and transforms["suffix_punct"] is None
                    and transforms["article"] is None
                    and transforms["prep"] is None
                ):
                    continue

                base_variant = (
                    f"{space_prefix}{base_text}" if transforms["space_prefix"] else base_text
                )
                if base_variant not in variant_to_entries:
                    continue

                if skip_multi_token_words or skip_three_token_words or skip_four_token_words:
                    token_encoding = _encode_variant_cached(normalized_token_str)
                    if skip_multi_token_words and len(token_encoding) > 1:
                        continue
                    if skip_three_token_words and len(token_encoding) > 2:
                        continue
                    if skip_four_token_words and len(token_encoding) > 3:
                        continue

                for base_token_key, existing_transforms in variant_to_entries[base_variant]:
                    inflection_val, derivation_val, space_val, base_cap_val = (
                        _extract_core_transform_values(existing_transforms)
                    )
                    article_space_val = None
                    if article_space_prefix_default is not None:
                        article_space_val = (
                            ADD_ARTICLE_SPACE_PREFIX
                            if transforms["article"] and transforms["article_space_prefix"]
                            else NO_ARTICLE_SPACE_PREFIX
                        )
                    prep_space_val = None
                    if prep_space_prefix_default is not None:
                        prep_space_val = (
                            ADD_PREP_SPACE_PREFIX
                            if transforms["prep"] and transforms["prep_space_prefix"]
                            else NO_PREP_SPACE_PREFIX
                        )
                    new_transforms = [
                        inflection_val,
                        derivation_val,
                        space_val,
                        base_cap_val,
                        transforms["prefix_punct"] or NO_PREFIX_PUNCTUATION,
                        transforms["suffix_punct"] or NO_SUFFIX_PUNCTUATION,
                        transforms["article"] or NO_ARTICLE,
                        transforms["prep"] or NO_PREPOSITION,
                    ]
                    if article_space_val is not None:
                        new_transforms.append(article_space_val)
                    if prep_space_val is not None:
                        new_transforms.append(prep_space_val)
                    new_transforms.extend(
                        [
                            ADD_ARTICLE_CAPITALIZATION
                            if transforms["article"] and transforms["article_cap"]
                            else NO_ARTICLE_CAPITALIZATION,
                            ADD_PREP_CAPITALIZATION
                            if transforms["prep"] and transforms["prep_cap"]
                            else NO_PREP_CAPITALIZATION,
                        ]
                    )

                    if token_str not in decomposition_map[base_token_key]:
                        decomposition_map[base_token_key][token_str] = new_transforms
                        punct_article_prep_generated += 1
        else:
            # Define transformation options
            articles_list = [ARTICLE_A, ARTICLE_THE, ARTICLE_AN] if decompose_articles else []
            preps_list = (
                [
                    f"prep_{p}"
                    for p in (
                        preposition_list
                        or ["by", "at", "of", "to", "in", "on", "with", "for", "from"]
                    )
                ]
                if decompose_prepositions
                else []
            )

            # Punctuation combinations
            prefix_punct_list = (
                [
                    PREFIX_PUNCT_SINGLE_QUOTE,
                    PREFIX_PUNCT_DOUBLE_QUOTE,
                    PREFIX_PUNCT_BACKTICK,
                    PREFIX_PUNCT_PAREN,
                    PREFIX_PUNCT_SQUARE,
                    PREFIX_PUNCT_CURLY,
                    PREFIX_PUNCT_HYPHEN,
                ]
                if decompose_punctuation
                else []
            )

            suffix_punct_list = (
                [
                    SUFFIX_PUNCT_PERIOD,
                    SUFFIX_PUNCT_EXCLAIM,
                    SUFFIX_PUNCT_QUESTION,
                    SUFFIX_PUNCT_COMMA,
                    SUFFIX_PUNCT_SEMICOLON,
                    SUFFIX_PUNCT_COLON,
                    SUFFIX_PUNCT_SINGLE_QUOTE,
                    SUFFIX_PUNCT_DOUBLE_QUOTE,
                    SUFFIX_PUNCT_PAREN,
                    SUFFIX_PUNCT_SQUARE,
                    SUFFIX_PUNCT_CURLY,
                    SUFFIX_PUNCT_POSSESSIVE_S,
                    SUFFIX_PUNCT_HYPHEN,
                ]
                if decompose_punctuation
                else []
            )

            # Build combinations for punctuation/articles/prepositions
            # Limit combinations to avoid explosion:
            punct_combos = []
            if decompose_punctuation:
                # Single punctuation marks
                for p in prefix_punct_list:
                    punct_combos.append(("prefix", p, None))
                for p in suffix_punct_list:
                    punct_combos.append(("suffix", None, p))
                # Matching pairs
                matching_pairs = [
                    (PREFIX_PUNCT_SINGLE_QUOTE, SUFFIX_PUNCT_SINGLE_QUOTE),
                    (PREFIX_PUNCT_DOUBLE_QUOTE, SUFFIX_PUNCT_DOUBLE_QUOTE),
                    (PREFIX_PUNCT_PAREN, SUFFIX_PUNCT_PAREN),
                    (PREFIX_PUNCT_SQUARE, SUFFIX_PUNCT_SQUARE),
                    (PREFIX_PUNCT_CURLY, SUFFIX_PUNCT_CURLY),
                ]
                for prefix_p, suffix_p in matching_pairs:
                    punct_combos.append(("both", prefix_p, suffix_p))
            punct_combos.append(("none", None, None))  # No punctuation

            phrase_combos = []
            if decompose_articles or decompose_prepositions:
                # Articles only
                if decompose_articles:
                    for art in articles_list:
                        for art_cap in [True, False]:
                            phrase_combos.append(("article", art, None, art_cap, False))
                # Prepositions only
                if decompose_prepositions:
                    for prep in preps_list:
                        for prep_cap in [True, False]:
                            phrase_combos.append(("prep", None, prep, False, prep_cap))
                # Prep + article
                if decompose_articles and decompose_prepositions:
                    for art in articles_list:
                        for prep in preps_list:
                            for art_cap in [True, False]:
                                for prep_cap in [True, False]:
                                    phrase_combos.append(("both", art, prep, art_cap, prep_cap))
            phrase_combos.append(("none", None, None, False, False))  # No phrase

            # Generate new tokens from ALL tokens in decomposition_map
            for base_token_key, variants_dict in tqdm(
                list(decomposition_map.items()), desc="Generating new transformations"
            ):
                for variant_token_str, existing_transforms in list(variants_dict.items()):
                    # Check if variant is multi-token
                    variant_encoding = _encode_variant_cached(variant_token_str)
                    is_multitoken = len(variant_encoding) > 1

                    if skip_multi_token_words and is_multitoken:
                        continue

                    # Extract the base form of the variant (strip space prefix)
                    variant_has_space = variant_token_str.startswith(space_prefix)
                    variant_word = (
                        variant_token_str[len(space_prefix) :]
                        if variant_has_space
                        else variant_token_str
                    )

                    inflection_val, derivation_val, space_val, base_cap_val = (
                        _extract_core_transform_values(existing_transforms)
                    )

                    # Generate new combinations
                    for punct_type, prefix_p, suffix_p in punct_combos:
                        for phrase_type, art, prep, art_cap, prep_cap in phrase_combos:
                            # Skip if no new transformations applied
                            if punct_type == "none" and phrase_type == "none":
                                continue

                            # Determine active groups
                            active_groups = []
                            if punct_type != "none":
                                active_groups.append("punctuation")
                            if phrase_type in ["article", "both"]:
                                active_groups.append("articles")
                            if phrase_type in ["prep", "both"]:
                                active_groups.append("prepositions")

                            # Build new token string
                            parts = []
                            if variant_has_space:
                                parts.append(space_prefix)
                            if prefix_p:
                                parts.append(prefix_p.replace("punct_prefix_", ""))
                            if prep:
                                prep_str = prep.replace("prep_", "")
                                parts.append(prep_str.capitalize() if prep_cap else prep_str)
                                parts.append(" ")
                            if art:
                                art_str = art.replace("article_", "")
                                parts.append(art_str.capitalize() if art_cap else art_str)
                                parts.append(" ")
                            parts.append(variant_word)
                            if suffix_p:
                                parts.append(suffix_p.replace("punct_suffix_", ""))

                            new_token_str = "".join(parts)

                            article_space_val = None
                            if article_space_prefix_default is not None:
                                if art:
                                    article_space = prep is not None or (
                                        variant_has_space and not prefix_p
                                    )
                                    article_space_val = (
                                        ADD_ARTICLE_SPACE_PREFIX
                                        if article_space
                                        else NO_ARTICLE_SPACE_PREFIX
                                    )
                                else:
                                    article_space_val = NO_ARTICLE_SPACE_PREFIX

                            prep_space_val = None
                            if prep_space_prefix_default is not None:
                                if prep:
                                    prep_space = variant_has_space and not prefix_p
                                    prep_space_val = (
                                        ADD_PREP_SPACE_PREFIX
                                        if prep_space
                                        else NO_PREP_SPACE_PREFIX
                                    )
                                else:
                                    prep_space_val = NO_PREP_SPACE_PREFIX

                            # Build transformation list: preserve inflection/derivation/space/base_cap, replace new groups
                            new_transforms = [
                                inflection_val,
                                derivation_val,
                                space_val,
                                base_cap_val,
                                prefix_p or NO_PREFIX_PUNCTUATION,
                                suffix_p or NO_SUFFIX_PUNCTUATION,
                                art or NO_ARTICLE,
                                prep or NO_PREPOSITION,
                            ]
                            if article_space_val is not None:
                                new_transforms.append(article_space_val)
                            if prep_space_val is not None:
                                new_transforms.append(prep_space_val)
                            new_transforms.extend(
                                [
                                    ADD_ARTICLE_CAPITALIZATION
                                    if art and art_cap
                                    else NO_ARTICLE_CAPITALIZATION,
                                    ADD_PREP_CAPITALIZATION
                                    if prep and prep_cap
                                    else NO_PREP_CAPITALIZATION,
                                ]
                            )

                            # Add to decomposition map
                            if new_token_str not in decomposition_map[base_token_key]:
                                decomposition_map[base_token_key][new_token_str] = new_transforms
                                punct_article_prep_generated += 1

        print(
            f"Generated {punct_article_prep_generated} new tokens with punctuation/articles/prepositions"
        )

    if morphology_mode == "compositional":
        active_morph_groups = _infer_active_morph_groups_from_map(decomposition_map)
        if active_morph_groups:
            for _, inflections in decomposition_map.items():
                for _, transforms in inflections.items():
                    present_groups = {
                        _get_morph_group_from_transform(t, active_morph_groups) for t in transforms
                    }
                    for group in active_morph_groups:
                        if group not in present_groups:
                            transforms.append(MORPHOLOGICAL_GROUP_NO_VALUES[group])

            for group in active_morph_groups:
                no_val = MORPHOLOGICAL_GROUP_NO_VALUES[group]
                na_val = MORPHOLOGICAL_GROUP_NA_VALUES[group]
                group_values = set()
                for _, inflections in decomposition_map.items():
                    for transforms in inflections.values():
                        for t in transforms:
                            if _get_morph_group_from_transform(t, [group]) == group:
                                group_values.add(t)
                if group_values.issubset({no_val, na_val}):
                    for _, inflections in decomposition_map.items():
                        for form, transforms in inflections.items():
                            inflections[form] = [na_val if t == no_val else t for t in transforms]

    if include_derivations:
        print("\nDerivation routing summary:")
        print(
            f"  chained_derivation_skipped={derivation_decision_stats.get('chained_derivation_skipped', 0)}"
        )
        print(
            "  deriv_plus_inflection_skipped_covered="
            f"{derivation_decision_stats.get('deriv_plus_inflection_skipped_covered', 0)}"
        )
        print(
            "  deriv_plus_inflection_allowed_uncovered="
            f"{derivation_decision_stats.get('deriv_plus_inflection_allowed_uncovered', 0)}"
        )
        print(
            "  base_derivation_conflicts_handled="
            f"{derivation_decision_stats.get('base_derivation_conflicts_handled', 0)}"
        )
        if getattr(unimorph, "invalid_derivation_rows_count", 0):
            print(
                "  noisy_derivation_rows_filtered_at_load="
                f"{getattr(unimorph, 'invalid_derivation_rows_count', 0)}"
            )

        sample_order = [
            "chained_derivation_skipped",
            "deriv_plus_inflection_skipped_covered",
            "deriv_plus_inflection_allowed_uncovered",
            "base_derivation_conflicts_handled",
        ]
        for bucket in sample_order:
            samples = derivation_decision_samples.get(bucket, [])
            if not samples:
                continue
            print(
                f"  samples[{bucket}] ({len(samples)} shown max {MAX_DERIVATION_CLASSIFICATION_SAMPLES}):"
            )
            for sample in samples[:8]:
                print(f"    {sample}")

    type_to_words = {k: list(v) for k, v in type_to_words.items()}

    filtered = {
        "duplicates": filtered_duplicate_inflections,
        "transform_collisions": filtered_transform_collisions,
        "propernouns": list(filtered_propernouns),
        "aux": list(filtered_aux),
        "converted_titlecase_bases": converted_titlecase_bases,
        "filtered_possessive_inflections": getattr(unimorph, "filtered_possessive_inflections", []),
        "derivation_decision_stats": dict(derivation_decision_stats),
        "derivation_decision_samples": {k: list(v) for k, v in derivation_decision_samples.items()},
        "invalid_derivation_rows_count": getattr(unimorph, "invalid_derivation_rows_count", 0),
        "invalid_derivation_rows_samples": list(
            getattr(unimorph, "invalid_derivation_rows_samples", [])
        ),
    }
    return (
        base_tokens,
        decomposition_map,
        type_to_words,
        ambiguous_inflections,
        inflection_baseform_conflicts,
        filtered,
        unimorph,
    )


def get_label_maps_from_decomposition_map(
    decomposition_map,
    morphology_mode="atomic",
    active_morphological_groups=None,
    force_articles=False,
    force_prepositions=False,
    force_punctuation=False,
    force_article_space_prefix=False,
    force_prep_space_prefix=False,
    preposition_list=None,
):
    """
    Extract unique transformation types from the decomposition map.

    Args:
        decomposition_map (dict): Decomposition map created by get_vocabulary_decomposition
        morphology_mode (str): 'atomic' or 'compositional'
        active_morphological_groups (list): List of active morphological groups (for compositional mode)
        force_articles (bool): Always include ARTICLE_* transforms, even if absent in the map
        force_prepositions (bool): Always include prep_* transforms, even if absent in the map
        force_punctuation (bool): Always include punct_* transforms, even if absent in the map
        force_article_space_prefix (bool): Always include article space-prefix transforms
        force_prep_space_prefix (bool): Always include preposition space-prefix transforms
        preposition_list (list): List of prepositions to include (defaults to common English)

    Returns:
        tuple: (transformations_to_int dict, loss_indices dict)
    """

    def _collect_types():
        types = set()
        for _, inflections in decomposition_map.items():
            for _, inflection_info in inflections.items():
                types.update(inflection_info)
        return types

    def _collect_morph_groups(types):
        morph_group_types = defaultdict(set)
        for t in types:
            group = _get_morph_group_from_transform(t)
            if group and t not in {
                MORPHOLOGICAL_GROUP_NO_VALUES[group],
                MORPHOLOGICAL_GROUP_NA_VALUES[group],
            }:
                morph_group_types[group].add(t)
        allowed_groups = active_morphological_groups or MORPHOLOGICAL_GROUPS
        active_groups = [g for g in allowed_groups if g in morph_group_types]
        return morph_group_types, active_groups

    transformation_types = _collect_types()

    morph_group_types = defaultdict(set)
    active_morph_groups = []
    if morphology_mode == "compositional":
        morph_group_types, active_morph_groups = _collect_morph_groups(transformation_types)

    def _order_article_types(article_types):
        ordered = [ARTICLE_THE, ARTICLE_A, ARTICLE_AN]
        present = [t for t in ordered if t in article_types]
        extra = [t for t in article_types if t not in ordered]
        return present + sorted(extra)

    def _order_prep_types(prep_types):
        ordered = [f"prep_{p}" for p in default_preps]
        present = [t for t in ordered if t in prep_types]
        extra = [t for t in prep_types if t not in ordered]
        return present + sorted(extra)

    def _order_punct_types(punct_types, ordered):
        present = [t for t in ordered if t in punct_types]
        extra = [t for t in punct_types if t not in ordered]
        return present + sorted(extra)

    default_preps = preposition_list or ["by", "at", "of", "to", "in", "on", "with", "for", "from"]

    inflection_types = (
        [t for t in transformation_types if t.startswith("inflect_")]
        if morphology_mode == "atomic"
        else []
    )
    derivation_types = [t for t in transformation_types if t.startswith("deriv_")]
    prefix_punct_types = [t for t in transformation_types if t.startswith("punct_prefix_")]
    suffix_punct_types = [t for t in transformation_types if t.startswith("punct_suffix_")]
    article_types = [
        t
        for t in transformation_types
        if t.startswith("article_") and t not in [NO_ARTICLE, NA_ARTICLE]
    ]
    prep_types = [
        t
        for t in transformation_types
        if t.startswith("prep_") and t not in [NO_PREPOSITION, NA_PREPOSITION]
    ]

    prefix_punct_order = [
        PREFIX_PUNCT_DOUBLE_QUOTE,
        PREFIX_PUNCT_SINGLE_QUOTE,
        PREFIX_PUNCT_BACKTICK,
        PREFIX_PUNCT_PAREN,
        PREFIX_PUNCT_SQUARE,
        PREFIX_PUNCT_CURLY,
        PREFIX_PUNCT_HYPHEN,
        "punct_prefix_\u2014",
    ]
    suffix_punct_order = [
        SUFFIX_PUNCT_DOUBLE_QUOTE,
        SUFFIX_PUNCT_SINGLE_QUOTE,
        SUFFIX_PUNCT_BACKTICK,
        SUFFIX_PUNCT_PAREN,
        SUFFIX_PUNCT_SQUARE,
        SUFFIX_PUNCT_CURLY,
        SUFFIX_PUNCT_PERIOD,
        SUFFIX_PUNCT_EXCLAIM,
        SUFFIX_PUNCT_QUESTION,
        SUFFIX_PUNCT_COMMA,
        SUFFIX_PUNCT_SEMICOLON,
        SUFFIX_PUNCT_COLON,
        SUFFIX_PUNCT_HYPHEN,
        "punct_suffix_\u2014",
        SUFFIX_PUNCT_POSSESSIVE_S,
    ]

    if force_punctuation:
        prefix_punct_types = prefix_punct_order
        suffix_punct_types = suffix_punct_order
    else:
        if prefix_punct_types:
            prefix_punct_types = _order_punct_types(prefix_punct_types, prefix_punct_order)
        if suffix_punct_types:
            suffix_punct_types = _order_punct_types(suffix_punct_types, suffix_punct_order)

    if force_articles:
        article_types = [ARTICLE_THE, ARTICLE_A, ARTICLE_AN]
    elif article_types:
        article_types = _order_article_types(article_types)

    if force_prepositions:
        prep_types = [f"prep_{p}" for p in default_preps]
    elif prep_types:
        prep_types = _order_prep_types(prep_types)

    article_space_active = (
        force_article_space_prefix or ADD_ARTICLE_SPACE_PREFIX in transformation_types
    )
    prep_space_active = force_prep_space_prefix or ADD_PREP_SPACE_PREFIX in transformation_types

    active_groups = {
        "inflections": morphology_mode == "atomic" and len(inflection_types) > 0,
        "derivations": len(derivation_types) > 0,
        "space_prefix": (
            WITH_SPACE_PREFIX_TRANSFORM in transformation_types
            or REMOVE_SPACE_PREFIX_TRANSFORM in transformation_types
        ),
        "base_capitalization": (
            ADD_BASE_CAPITALIZATION_TRANSFORM in transformation_types
            or ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM in transformation_types
            or REMOVE_BASE_CAPITALIZATION_TRANSFORM in transformation_types
        ),
        "prefix_punctuation": len(prefix_punct_types) > 0,
        "suffix_punctuation": len(suffix_punct_types) > 0,
        "articles": len(article_types) > 0,
        "prepositions": len(prep_types) > 0,
        "article_space_prefix": article_space_active,
        "prep_space_prefix": prep_space_active,
        "article_capitalization": ADD_ARTICLE_CAPITALIZATION in transformation_types,
        "preposition_capitalization": ADD_PREP_CAPITALIZATION in transformation_types,
    }

    allowed_transformations = set()
    if active_groups["inflections"]:
        allowed_transformations.update(inflection_types)
        allowed_transformations.update([NO_INFLECTION, NA_INFLECTION])
    if morphology_mode == "compositional":
        for group in active_morph_groups:
            allowed_transformations.update(morph_group_types[group])
            allowed_transformations.update(
                [MORPHOLOGICAL_GROUP_NO_VALUES[group], MORPHOLOGICAL_GROUP_NA_VALUES[group]]
            )
    if active_groups["derivations"]:
        allowed_transformations.update(derivation_types)
        allowed_transformations.update([NO_DERIVATION, NA_DERIVATION])
    if active_groups["space_prefix"]:
        allowed_transformations.update(
            [
                NO_SPACE_PREFIX_TRANSFORM,
                WITH_SPACE_PREFIX_TRANSFORM,
                REMOVE_SPACE_PREFIX_TRANSFORM,
                NA_SPACE_PREFIX_TRANSFORM,
            ]
        )
    if active_groups["base_capitalization"]:
        allowed_transformations.update(
            [
                ADD_BASE_CAPITALIZATION_TRANSFORM,
                ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
                REMOVE_BASE_CAPITALIZATION_TRANSFORM,
                NO_BASE_CAPITALIZATION_TRANSFORM,
                NA_BASE_CAPITALIZATION_TRANSFORM,
            ]
        )
    if active_groups["prefix_punctuation"]:
        allowed_transformations.update(prefix_punct_types)
        allowed_transformations.update([NO_PREFIX_PUNCTUATION, NA_PREFIX_PUNCTUATION])
    if active_groups["suffix_punctuation"]:
        allowed_transformations.update(suffix_punct_types)
        allowed_transformations.update([NO_SUFFIX_PUNCTUATION, NA_SUFFIX_PUNCTUATION])
    if active_groups["articles"]:
        allowed_transformations.update(article_types)
        allowed_transformations.update([NO_ARTICLE, NA_ARTICLE])
    if active_groups["prepositions"]:
        allowed_transformations.update(prep_types)
        allowed_transformations.update([NO_PREPOSITION, NA_PREPOSITION])
    if active_groups["article_space_prefix"]:
        allowed_transformations.update(
            [NO_ARTICLE_SPACE_PREFIX, ADD_ARTICLE_SPACE_PREFIX, NA_ARTICLE_SPACE_PREFIX]
        )
    if active_groups["prep_space_prefix"]:
        allowed_transformations.update(
            [NO_PREP_SPACE_PREFIX, ADD_PREP_SPACE_PREFIX, NA_PREP_SPACE_PREFIX]
        )
    if active_groups["article_capitalization"]:
        allowed_transformations.update(
            [ADD_ARTICLE_CAPITALIZATION, NO_ARTICLE_CAPITALIZATION, NA_ARTICLE_CAPITALIZATION]
        )
    if active_groups["preposition_capitalization"]:
        allowed_transformations.update(
            [ADD_PREP_CAPITALIZATION, NO_PREP_CAPITALIZATION, NA_PREP_CAPITALIZATION]
        )

    if allowed_transformations:
        for _, inflections in decomposition_map.items():
            for inflection_form, inflection_info in inflections.items():
                inflections[inflection_form] = [
                    t for t in inflection_info if t in allowed_transformations
                ]

    transformation_types = _collect_types()
    inflection_types = (
        sorted([t for t in transformation_types if t.startswith("inflect_")])
        if morphology_mode == "atomic"
        else []
    )
    derivation_types = sorted([t for t in transformation_types if t.startswith("deriv_")])
    prefix_punct_types = [t for t in transformation_types if t.startswith("punct_prefix_")]
    suffix_punct_types = [t for t in transformation_types if t.startswith("punct_suffix_")]
    article_types = [
        t
        for t in transformation_types
        if t.startswith("article_") and t not in [NO_ARTICLE, NA_ARTICLE]
    ]
    prep_types = [
        t
        for t in transformation_types
        if t.startswith("prep_") and t not in [NO_PREPOSITION, NA_PREPOSITION]
    ]

    if force_punctuation:
        prefix_punct_types = prefix_punct_order
        suffix_punct_types = suffix_punct_order
    else:
        if prefix_punct_types:
            prefix_punct_types = _order_punct_types(prefix_punct_types, prefix_punct_order)
        if suffix_punct_types:
            suffix_punct_types = _order_punct_types(suffix_punct_types, suffix_punct_order)

    if force_articles:
        article_types = [ARTICLE_THE, ARTICLE_A, ARTICLE_AN]
    elif article_types:
        article_types = _order_article_types(article_types)

    if force_prepositions:
        prep_types = [f"prep_{p}" for p in default_preps]
    elif prep_types:
        prep_types = _order_prep_types(prep_types)

    if morphology_mode == "compositional":
        morph_group_types, active_morph_groups = _collect_morph_groups(transformation_types)

    # Build transformations_names with consecutive groups
    # IMPORTANT: For UnifiedModifierArray, rel=0 must be "no transformation" (NO_*),
    # so we put NO_* first in each group, then actual transformations, then NA_* last
    transformations_names = []
    loss_indices = {}
    current_idx = 0

    if morphology_mode == "compositional":
        transformations_names.extend(
            [
                NO_SPACE_PREFIX_TRANSFORM,
                WITH_SPACE_PREFIX_TRANSFORM,
                REMOVE_SPACE_PREFIX_TRANSFORM,
                NA_SPACE_PREFIX_TRANSFORM,
            ]
        )
        transformations_names.extend(
            [
                NO_BASE_CAPITALIZATION_TRANSFORM,
                ADD_BASE_CAPITALIZATION_TRANSFORM,
                ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
                REMOVE_BASE_CAPITALIZATION_TRANSFORM,
                NA_BASE_CAPITALIZATION_TRANSFORM,
            ]
        )

        if derivation_types:
            transformations_names.append(NO_DERIVATION)
            transformations_names.extend(derivation_types)
            transformations_names.append(NA_DERIVATION)

        for group in active_morph_groups:
            group_start = len(transformations_names)
            transformations_names.append(MORPHOLOGICAL_GROUP_NO_VALUES[group])
            for value in sorted(morph_group_types[group]):
                transformations_names.append(value)
            transformations_names.append(MORPHOLOGICAL_GROUP_NA_VALUES[group])
            group_end = len(transformations_names)
            loss_indices[group] = (group_start, group_end)

    else:
        inflections_start = current_idx
        transformations_names.append(NO_INFLECTION)
        transformations_names.extend(inflection_types)
        transformations_names.append(NA_INFLECTION)
        inflections_end = len(transformations_names)
        loss_indices["inflection"] = (inflections_start, inflections_end)
        loss_indices["inflections"] = (inflections_start, inflections_end)
        current_idx = inflections_end

        transformations_names.append(NO_DERIVATION)
        transformations_names.extend(derivation_types)
        transformations_names.append(NA_DERIVATION)

        transformations_names.extend(
            [
                NO_SPACE_PREFIX_TRANSFORM,
                WITH_SPACE_PREFIX_TRANSFORM,
                REMOVE_SPACE_PREFIX_TRANSFORM,
                NA_SPACE_PREFIX_TRANSFORM,
            ]
        )
        transformations_names.extend(
            [
                NO_BASE_CAPITALIZATION_TRANSFORM,
                ADD_BASE_CAPITALIZATION_TRANSFORM,
                ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
                REMOVE_BASE_CAPITALIZATION_TRANSFORM,
                NA_BASE_CAPITALIZATION_TRANSFORM,
            ]
        )

    transformations_names.append(NO_PREFIX_PUNCTUATION)
    transformations_names.extend(prefix_punct_types)
    transformations_names.append(NA_PREFIX_PUNCTUATION)

    transformations_names.append(NO_SUFFIX_PUNCTUATION)
    transformations_names.extend(suffix_punct_types)
    transformations_names.append(NA_SUFFIX_PUNCTUATION)

    transformations_names.append(NO_ARTICLE)
    transformations_names.extend(article_types)
    transformations_names.append(NA_ARTICLE)
    if article_space_active:
        transformations_names.extend(
            [
                NO_ARTICLE_SPACE_PREFIX,
                ADD_ARTICLE_SPACE_PREFIX,
                NA_ARTICLE_SPACE_PREFIX,
            ]
        )

    transformations_names.append(NO_PREPOSITION)
    transformations_names.extend(prep_types)
    transformations_names.append(NA_PREPOSITION)
    if prep_space_active:
        transformations_names.extend(
            [
                NO_PREP_SPACE_PREFIX,
                ADD_PREP_SPACE_PREFIX,
                NA_PREP_SPACE_PREFIX,
            ]
        )

    transformations_names.extend(
        [NO_ARTICLE_CAPITALIZATION, ADD_ARTICLE_CAPITALIZATION, NA_ARTICLE_CAPITALIZATION]
    )
    transformations_names.extend(
        [NO_PREP_CAPITALIZATION, ADD_PREP_CAPITALIZATION, NA_PREP_CAPITALIZATION]
    )

    transformations_to_int = dict(zip(transformations_names, range(len(transformations_names))))

    if morphology_mode == "compositional":
        prefix_start = 0
        prefix_end = transformations_names.index(NA_SPACE_PREFIX_TRANSFORM) + 1

        base_cap_start = prefix_end
        base_cap_end = transformations_names.index(NA_BASE_CAPITALIZATION_TRANSFORM) + 1

        if derivation_types:
            derivations_start = base_cap_end
            derivations_end = transformations_names.index(NA_DERIVATION) + 1
        else:
            derivations_start = base_cap_end
            derivations_end = base_cap_end

        loss_indices.update(
            {
                "space_prefix": (prefix_start, prefix_end),
                "base_capitalization": (base_cap_start, base_cap_end),
                "derivation": (derivations_start, derivations_end),
                "derivations": (derivations_start, derivations_end),
                "prefix": (prefix_start, prefix_end),
            }
        )

        current_pos = base_cap_end
        if derivation_types:
            current_pos = derivations_end

        prefix_punct_start = current_pos
        prefix_punct_end = transformations_names.index(NA_PREFIX_PUNCTUATION) + 1
        current_pos = prefix_punct_end

        suffix_punct_start = current_pos
        suffix_punct_end = transformations_names.index(NA_SUFFIX_PUNCTUATION) + 1
        current_pos = suffix_punct_end

        article_start = current_pos
        article_end = transformations_names.index(NA_ARTICLE) + 1
        current_pos = article_end

        if article_space_active:
            article_space_start = current_pos
            article_space_end = transformations_names.index(NA_ARTICLE_SPACE_PREFIX) + 1
            current_pos = article_space_end

        prep_start = current_pos
        prep_end = transformations_names.index(NA_PREPOSITION) + 1
        current_pos = prep_end

        if prep_space_active:
            prep_space_start = current_pos
            prep_space_end = transformations_names.index(NA_PREP_SPACE_PREFIX) + 1
            current_pos = prep_space_end

        article_cap_start = current_pos
        article_cap_end = transformations_names.index(NA_ARTICLE_CAPITALIZATION) + 1
        current_pos = article_cap_end

        prep_cap_start = current_pos
        prep_cap_end = transformations_names.index(NA_PREP_CAPITALIZATION) + 1

        loss_indices.update(
            {
                "prefix_punctuation": (prefix_punct_start, prefix_punct_end),
                "suffix_punctuation": (suffix_punct_start, suffix_punct_end),
                "articles": (article_start, article_end),
                "prepositions": (prep_start, prep_end),
                "article_capitalization": (article_cap_start, article_cap_end),
                "prep_capitalization": (prep_cap_start, prep_cap_end),
                "preposition_capitalization": (prep_cap_start, prep_cap_end),
            }
        )
        if article_space_active:
            loss_indices["article_space_prefix"] = (article_space_start, article_space_end)
        if prep_space_active:
            loss_indices["prep_space_prefix"] = (prep_space_start, prep_space_end)

    else:
        inflections_start = 0
        inflections_end = transformations_names.index(NA_INFLECTION) + 1

        derivations_start = inflections_end
        derivations_end = transformations_names.index(NA_DERIVATION) + 1

        prefix_start = derivations_end
        prefix_end = transformations_names.index(NA_SPACE_PREFIX_TRANSFORM) + 1

        base_cap_start = prefix_end
        base_cap_end = transformations_names.index(NA_BASE_CAPITALIZATION_TRANSFORM) + 1

        prefix_punct_start = base_cap_end
        prefix_punct_end = transformations_names.index(NA_PREFIX_PUNCTUATION) + 1

        suffix_punct_start = prefix_punct_end
        suffix_punct_end = transformations_names.index(NA_SUFFIX_PUNCTUATION) + 1

        article_start = suffix_punct_end
        article_end = transformations_names.index(NA_ARTICLE) + 1
        current_pos = article_end

        if article_space_active:
            article_space_start = current_pos
            article_space_end = transformations_names.index(NA_ARTICLE_SPACE_PREFIX) + 1
            current_pos = article_space_end

        prep_start = current_pos
        prep_end = transformations_names.index(NA_PREPOSITION) + 1
        current_pos = prep_end

        if prep_space_active:
            prep_space_start = current_pos
            prep_space_end = transformations_names.index(NA_PREP_SPACE_PREFIX) + 1
            current_pos = prep_space_end

        article_cap_start = current_pos
        article_cap_end = transformations_names.index(NA_ARTICLE_CAPITALIZATION) + 1

        prep_cap_start = article_cap_end
        prep_cap_end = transformations_names.index(NA_PREP_CAPITALIZATION) + 1

        loss_indices.update(
            {
                "inflection": (inflections_start, inflections_end),
                "derivation": (derivations_start, derivations_end),
                "space_prefix": (prefix_start, prefix_end),
                "base_capitalization": (base_cap_start, base_cap_end),
                "prefix_punctuation": (prefix_punct_start, prefix_punct_end),
                "suffix_punctuation": (suffix_punct_start, suffix_punct_end),
                "articles": (article_start, article_end),
                "prepositions": (prep_start, prep_end),
                "article_capitalization": (article_cap_start, article_cap_end),
                "prep_capitalization": (prep_cap_start, prep_cap_end),
                "inflections": (inflections_start, inflections_end),
                "derivations": (derivations_start, derivations_end),
                "prefix": (prefix_start, prefix_end),
                "preposition_capitalization": (prep_cap_start, prep_cap_end),
            }
        )
        if article_space_active:
            loss_indices["article_space_prefix"] = (article_space_start, article_space_end)
        if prep_space_active:
            loss_indices["prep_space_prefix"] = (prep_space_start, prep_space_end)

    return transformations_to_int, loss_indices


def extend_tokenizer_by_decomposition_map(
    tokenizer,
    base_tokens,
    decomposition_map,
    space_prefix=None,
    skip_multi_token_words=False,
    skip_three_token_words=False,
    skip_four_token_words=False,
    transformations_to_int=None,
    multitoken_allowed_groups=None,
    tokenization_mode="single_id",
    new_transform_groups=None,
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

    # Default multitoken_allowed_groups to empty list if not provided
    if multitoken_allowed_groups is None:
        multitoken_allowed_groups = []

    newest_token_id = max(vocab_size, max(tokenizer.added_tokens_decoder.keys()) + 1)
    for base_word, inflections in decomposition_map.items():
        base_token_id = base_tokens[base_word]

        # Handle multi-token bases
        if isinstance(base_token_id, tuple):
            if tokenization_mode == "single_id":
                raise ValueError(
                    f"Multi-token base word '{base_word}' (tokens: {base_token_id}) found in decomposition_map, "
                    f"but tokenization_mode is 'single_id'. Multi-token bases require dual_stream mode.\n"
                    f"Solutions:\n"
                    f"  1. Use --tokenization_mode dual_stream (recommended for multi-token base support), OR\n"
                    f"  2. Use --skip_multi_token_bases to exclude multi-token bases (keeps multi-token inflections)"
                )
            # In dual_stream mode, multi-token bases are handled via SequenceMap, not extended vocab
            # Skip adding to final_decomposition_map (which is for single-token bases only)
            continue

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

            # Get transformation list to check for new transformations
            transformations_list = decomposition_map[base_word][inflection_word]

            # add scaffold tokens and merges if necessary
            if len(inflection_word_encoding) == 1:
                decomposed_token_id = inflection_word_encoding[0]
                existing_inflection_token_ids.add(decomposed_token_id)
            else:
                # Check if token has new transformations (punctuation, articles, prepositions)
                has_new_transformations = any(
                    t.startswith("punct_prefix_")
                    or t.startswith("punct_suffix_")
                    or (t.startswith("article_") and t not in [NO_ARTICLE, NA_ARTICLE])
                    or (t.startswith("prep_") and t not in [NO_PREPOSITION, NA_PREPOSITION])
                    for t in transformations_list
                )

                # Determine active groups for whitelist checking
                active_groups = []
                if any(
                    t.startswith("punct_prefix_") or t.startswith("punct_suffix_")
                    for t in transformations_list
                ):
                    active_groups.append("punctuation")
                if any(
                    t.startswith("article_") and t not in [NO_ARTICLE, NA_ARTICLE]
                    for t in transformations_list
                ):
                    active_groups.append("articles")
                if any(
                    t.startswith("prep_") and t not in [NO_PREPOSITION, NA_PREPOSITION]
                    for t in transformations_list
                ):
                    active_groups.append("prepositions")

                # Check if whitelisted
                is_whitelisted = any(g in multitoken_allowed_groups for g in active_groups)

                # Apply multi-token filtering with whitelist override
                if not is_whitelisted:
                    if skip_multi_token_words:
                        continue
                    if skip_three_token_words and len(inflection_word_encoding) > 2:
                        continue
                    if skip_four_token_words and len(inflection_word_encoding) > 3:
                        continue

                # DUAL-STREAM MODE: Skip scaffold token creation for multi-token sequences that
                # are better represented via SequenceMap (e.g., explicit space-prefix variants).
                has_with_space_prefix_transform = (
                    WITH_SPACE_PREFIX_TRANSFORM in transformations_list
                )
                skip_scaffold_creation = tokenization_mode == "dual_stream" and (
                    has_new_transformations or has_with_space_prefix_transform
                )

                if skip_scaffold_creation:
                    # Let string-based sequence decomposition handle this variant to avoid
                    # collapsing multiple variants onto base_token_id in final_decomposition_map.
                    continue
                else:
                    # Create scaffold tokens as usual (single_id mode or old transformations)
                    curr_new_token = (
                        f"{tokenizer._tokenizer.id_to_token(inflection_word_encoding[0])}"
                    )
                    curr_token_id = None

                    for i in range(len(inflection_word_encoding) - 1):
                        curr_merge = [
                            f"{curr_new_token}",
                            f"{tokenizer._tokenizer.id_to_token(inflection_word_encoding[i + 1])}",
                        ]

                        curr_new_token = f"{curr_new_token}{tokenizer._tokenizer.id_to_token(inflection_word_encoding[i + 1])}"
                        if curr_new_token in new_tokens:
                            # Token already exists - reuse its ID
                            curr_token_id = new_tokens[curr_new_token]
                            continue
                        scaffold_merges.append(curr_merge)
                        new_tokens[curr_new_token] = newest_token_id
                        curr_token_id = newest_token_id
                        if i < len(inflection_word_encoding) - 2:
                            # This is an intermediate scaffold token
                            scaffold_vocab[curr_new_token] = newest_token_id
                            # Add to decomposition map so it has a valid base mapping
                            # Map it to the base word's token ID
                            final_decomposition_map[newest_token_id] = (base_token_id, [])
                        newest_token_id += 1

                    decomposed_token_id = curr_token_id

            decomposed_token_ids.append(decomposed_token_id)
            decomposed_to_original_id[decomposed_token_id] = inflection_word_encoding[0]
            decomposed_to_original_seq[decomposed_token_id] = inflection_word_encoding
            final_decomposition_map[decomposed_token_id] = (
                base_token_id,
                [transformations_to_int[transformation] for transformation in transformations_list],
            )

    new_tokenizer = add_words_to_core_vocab(tokenizer, new_tokens, scaffold_merges)
    existing_inflection_token_ids = list(existing_inflection_token_ids)
    negative_types = set(
        [
            transformations_to_int[type_name]
            for type_name in NO_EMBEDDING_TYPES
            if type_name in transformations_to_int
        ]
    )
    na_types = set(
        [
            transformations_to_int[type_name]
            for type_name in NA_EMBEDDING_TYPES
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
        na_types,
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
