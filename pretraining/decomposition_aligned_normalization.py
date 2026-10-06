import re
import os
import json
import hashlib
import pickle
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, Set, Tuple

from tqdm import tqdm

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


@dataclass
class DecompositionAlignedNormalizerStats:
    lemma_entries: int
    identity_pairs: int
    unimorph_inflection_pairs: int
    unimorph_derivation_pairs: int
    unimorph_derived_inflection_pairs: int
    pyinflect_pairs: int
    surface_to_base_size: int
    ambiguous_surface_forms: int


UNIMORPH_INFLECTION_WEIGHT = 8
UNIMORPH_DERIVATION_WEIGHT = 7
UNIMORPH_DERIVED_INFLECTION_WEIGHT = 9
PYINFLECT_WEIGHT = 5
IDENTITY_WEIGHT = 1
PYINFLECT_PREFERENCE_BONUS = 5
PREFERRED_LEMMA_BONUS = 2
POPULARITY_OVERRIDE_RATIO = 2.0
POPULARITY_OVERRIDE_MAX_SCORE_GAP = 3
NORMALIZER_CACHE_VERSION = 5


class DecompositionAlignedMorphNormalizer:
    """Morphology normalizer derived from decomposition-map logic."""

    def __init__(self, surface_to_base: Dict[str, str], stats: DecompositionAlignedNormalizerStats):
        self.surface_to_base = surface_to_base
        self.stats = stats

    def normalize_text(self, text: str) -> str:
        def _replace(match: re.Match) -> str:
            token = match.group(0)
            mapped = self.surface_to_base.get(token.lower())
            return mapped if mapped is not None else token

        return WORD_PATTERN.sub(_replace, text)


def _has_whitespace(text: str) -> bool:
    return any(ch.isspace() for ch in text)


def _tag_set_has(tag_set: Tuple[str, ...], required_tags: list) -> bool:
    return all(tag in tag_set for tag in required_tags)


def _should_skip_unimorph_pair(
    lemma: str,
    form: str,
    tag_sets: Set[Tuple[str, ...]],
    *,
    use_english_resources: bool,
    unimorph_skip_lemmas: Set[str],
    vocab_base_words_filter: Callable[[str], bool],
) -> bool:
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
            if _tag_set_has(tag_set, ["N", "PL"]) or _tag_set_has(tag_set, ["V", "PRS", "3", "SG"]):
                return True
    if (
        len(lemma_lower) <= 3
        and vocab_base_words_filter(form_lower)
        and not vocab_base_words_filter(lemma_lower)
    ):
        return True
    return False
