import os
import json
import pickle
import hashlib
import shutil
from collections import defaultdict
from transformers import AutoTokenizer
from scaffold_tokenizer import ScaffoldTokenizer
from decomposition_utils import (
    get_vocabulary_decomposition,
    extend_tokenizer_by_decomposition_map,
    get_label_maps_from_decomposition_map,
    ModifierMap,
    UnifiedModifierArray,
)


class CompositionalTokenizerBundle:
    def __init__(
        self,
        tokenizer,
        model_init_data,
        tokenization_mode="single_id",
        modifier_map=None,
        unified_modifier_array=None,
        dual_stream_tokenizer_config=None,
        sequence_map=None,
    ):
        """Initialize the tokenizer bundle.

        Args:
            tokenizer: The tokenizer (ScaffoldTokenizer or base tokenizer)
            model_init_data: Dictionary containing model initialization data
            tokenization_mode: "single_id" (default) or "dual_stream"
            modifier_map: ModifierMap instance (only for dual_stream mode with 1D modifiers)
            unified_modifier_array: UnifiedModifierArray instance (for 2D modifiers)
            dual_stream_tokenizer_config: Dict config for DualStreamTokenizer (optional)
            sequence_map: Pre-built SequenceMap instance (cached for performance)
        """
        self.tokenizer = tokenizer
        self.model_init_data = model_init_data
        self.tokenization_mode = tokenization_mode
        self.modifier_map = modifier_map
        self.unified_modifier_array = unified_modifier_array
        self.dual_stream_tokenizer_config = dual_stream_tokenizer_config
        self.sequence_map = sequence_map

    def _build_manifest(self):
        base_vocab_size = len(self.model_init_data.get("non_inflection_indices", []))
        return {
            "format_version": 1,
            "base_tokenizer_size": self.model_init_data.get("base_tokenizer_size"),
            "base_vocab_size": base_vocab_size,
            "extended_vocab_size": len(self.tokenizer),
            "unified_modifier_dtype": self.model_init_data.get("unified_modifier_dtype"),
            "type_groups": self.model_init_data.get("type_groups", {}),
            "transformation_names_to_int": self.model_init_data.get(
                "transformation_names_to_int", {}
            ),
            "required_keys": [
                "base_tokenizer_size",
                "type_groups",
                "transformation_names_to_int",
                "token_id_to_base_id_mapping",
                "base_token_indices",
                "non_inflection_indices",
            ],
            "present_keys": sorted(self.model_init_data.keys()),
        }

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

        # Determine tokenizer type
        tokenizer_type = "scaffold" if isinstance(self.tokenizer, ScaffoldTokenizer) else "auto"

        # Save model_init_data with tokenization mode and modifier_map
        bundle_data = {
            "model_init_data": self.model_init_data,
            "tokenization_mode": self.tokenization_mode,
            "modifier_map": self.modifier_map.to_dict() if self.modifier_map else None,
            "unified_modifier_array": self.unified_modifier_array.to_dict()
            if self.unified_modifier_array
            else None,
            "dual_stream_tokenizer_config": self.dual_stream_tokenizer_config,
            "tokenizer_type": tokenizer_type,  # Track which tokenizer class to use
        }

        with open(os.path.join(save_path, "model_init_data.pkl"), "wb") as f:
            pickle.dump(bundle_data, f)
        with open(os.path.join(save_path, "model_init_manifest.json"), "w") as f:
            json.dump(self._build_manifest(), f, indent=2, sort_keys=True)

        if self.sequence_map is not None:
            with open(os.path.join(save_path, "sequence_map.pkl"), "wb") as f:
                pickle.dump(self.sequence_map, f)
            print(f"Cached SequenceMap with {len(self.sequence_map)} sequences")

    @classmethod
    def load(cls, load_path, base_tokenizer_name=None):
        # First load metadata to determine tokenizer type
        with open(os.path.join(load_path, "model_init_data.pkl"), "rb") as f:
            bundle_data = pickle.load(f)

        # Determine tokenizer type from metadata
        tokenizer_type = bundle_data.get(
            "tokenizer_type", "scaffold"
        )  # Default to scaffold for backward compat

        if tokenizer_type == "auto":
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_path, "tokenizer"))
            print(f"Loaded tokenizer using AutoTokenizer (dual-stream mode)")
        else:
            # Use ScaffoldTokenizer for traditional mode
            tokenizer = ScaffoldTokenizer.from_pretrained(os.path.join(load_path, "tokenizer"))

        # If base_tokenizer_name is provided, restore special tokens from base tokenizer
        # This handles cases where cached tokenizers were created before special tokens were explicitly set
        if base_tokenizer_name is not None and (
            tokenizer.eos_token_id is None or tokenizer.bos_token_id is None
        ):
            print(f"Restoring special tokens from base tokenizer: {base_tokenizer_name}")
            base_tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)

            # Restore special tokens
            if tokenizer.eos_token_id is None and base_tokenizer.eos_token is not None:
                tokenizer.eos_token = base_tokenizer.eos_token
                print(
                    f"  Restored eos_token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})"
                )
            if tokenizer.bos_token_id is None and base_tokenizer.bos_token is not None:
                tokenizer.bos_token = base_tokenizer.bos_token
                print(
                    f"  Restored bos_token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})"
                )
            if tokenizer.pad_token_id is None and base_tokenizer.pad_token is not None:
                tokenizer.pad_token = base_tokenizer.pad_token
            if tokenizer.unk_token_id is None and base_tokenizer.unk_token is not None:
                tokenizer.unk_token = base_tokenizer.unk_token

        # bundle_data was already loaded at the start of load() to determine tokenizer type

        # Handle both old format (just dict) and new format (dict with mode/modifier_map)
        if isinstance(bundle_data, dict) and "model_init_data" in bundle_data:
            # New format: bundle_data contains model_init_data, tokenization_mode, modifier_map
            model_init_data = bundle_data["model_init_data"]
            tokenization_mode = bundle_data.get("tokenization_mode", "single_id")
            modifier_map_dict = bundle_data.get("modifier_map")
            modifier_map = ModifierMap.from_dict(modifier_map_dict) if modifier_map_dict else None

            unified_modifier_array_dict = bundle_data.get("unified_modifier_array")
            unified_modifier_array = (
                UnifiedModifierArray.from_dict(unified_modifier_array_dict)
                if unified_modifier_array_dict
                else None
            )
            if (
                unified_modifier_array is not None
                and "unified_modifier_dtype" not in model_init_data
            ):
                model_init_data["unified_modifier_dtype"] = (
                    unified_modifier_array.recommended_modifier_dtype_name()
                )

            # Load dual stream tokenizer config if present
            dual_stream_tokenizer_config = bundle_data.get("dual_stream_tokenizer_config")
            if (
                dual_stream_tokenizer_config is not None
                and "modifier_dtype" not in dual_stream_tokenizer_config
            ):
                dual_stream_tokenizer_config["modifier_dtype"] = model_init_data.get(
                    "unified_modifier_dtype"
                )

            sequence_map = None
            sequence_map_path = os.path.join(load_path, "sequence_map.pkl")
            if os.path.exists(sequence_map_path):
                with open(sequence_map_path, "rb") as f:
                    sequence_map = pickle.load(f)
                print(f"Loaded cached SequenceMap with {len(sequence_map)} sequences")

            print(f"Loaded bundle with tokenization_mode={tokenization_mode}")
            if modifier_map:
                print(f"  ModifierMap: {modifier_map}")
            if unified_modifier_array:
                print(f"  UnifiedModifierArray: {unified_modifier_array}")
        else:
            # Old format: bundle_data is directly the model_init_data
            model_init_data = bundle_data
            tokenization_mode = "single_id"
            modifier_map = None
            unified_modifier_array = None
            dual_stream_tokenizer_config = None
            sequence_map = None
            print("Loaded legacy bundle (single_id mode)")

        return cls(
            tokenizer,
            model_init_data,
            tokenization_mode,
            modifier_map,
            unified_modifier_array,
            dual_stream_tokenizer_config,
            sequence_map,
        )


def _resolve_modifier_generation_mode(args):
    mode = getattr(args, "modifier_generation_mode", None)
    if mode:
        mode = mode.lower()
        if mode != "auto":
            return mode
    tokenization_mode = getattr(args, "tokenization_mode", "single_id")
    pass
    return "full"


def _normalize_language_tag_for_cache(language: str) -> str:
    if not language:
        return "en"
    primary = language.split(":", 1)[0].strip()
    if not primary:
        primary = language.strip()
    primary = primary.replace("-", "_")
    if primary.startswith("en"):
        return "en"
    return primary


def get_tokenizer_cache_path(args, base_cache_dir=None):
    # Extract base name from tokenizer path (handles both HF Hub names and local paths)
    base_name = os.path.basename(args.base_tokenizer.rstrip(os.sep))
    if not base_name:  # Handle edge case of root path
        base_name = args.base_tokenizer.replace(os.sep, "_").replace("/", "_")

    features = []
    language = getattr(args, "language", "en")
    language_tag = _normalize_language_tag_for_cache(str(language))
    if language_tag != "en":
        features.append(f"lang_{language_tag}")
    if (
        getattr(args, "unimorph_root", None)
        or getattr(args, "unimorph_inflections_path", None)
        or getattr(args, "unimorph_derivations_path", None)
    ):
        features.append("unimorph_custom")
    if not args.dont_decompose_spaces:
        features.append("decomp_spaces")
    if args.skip_if_no_space_prefix:
        features.append("filter_no_space_prefix")
    if not args.dont_decompose_capitalized:
        features.append("decomp_caps")
    transform_collision_mode = getattr(args, "transform_collision_mode", "off")
    if transform_collision_mode and transform_collision_mode != "off":
        features.append(f"transform_collisions_{transform_collision_mode}")
    if args.include_derivations:
        features.append("derivations")
        min_derivation_count = int(getattr(args, "min_derivation_count", 50))
        features.append(f"min_deriv_{min_derivation_count}")
    if not args.filter_propernouns_and_aux:
        features.append("w_aux")
    features.append("w_subwords")
    features.append("w_symbols")
    if not args.skip_stop_words:
        features.append("w_stopwords")
    if args.use_type_str_as_type:
        features.append("type_as_str")
    if args.merge_plural_and_present_singular:
        features.append("merged_plural_vp3s")
    features.append("prefer_popular_base_conflicts")
    if not args.skip_multi_token_words:
        features.append("multi_token")
    if args.skip_three_token_words and not args.skip_multi_token_words:
        features.append("upto_two_tokens")
    elif getattr(args, "skip_four_token_words", False) and not args.skip_multi_token_words:
        features.append("upto_three_tokens")
    if getattr(args, "skip_multi_token_bases", False):
        features.append("skip_multi_token_bases")

    # New transformation groups
    modifier_generation_mode = _resolve_modifier_generation_mode(args).lower()
    if modifier_generation_mode == "vocab_only":
        features.append("modifiers_vocab_only")

    # Multi-token whitelist for new transformations
    if getattr(args, "max_tokens_per_base_word", None) is not None:
        features.append(f"max_tokens_per_base_{int(args.max_tokens_per_base_word)}")

    # Morphology mode
    morphology_mode = getattr(args, "morphology_mode", "atomic")
    if morphology_mode != "atomic":
        features.append(f"morph_{morphology_mode}")
    clitic_mode = getattr(args, "clitic_mode", "surface2")
    if clitic_mode != "surface2":
        features.append(f"clitics_{clitic_mode}")

    # Note: tokenization_mode is generally not encoded in cache path, except for

    feature_str = "_".join(features) if features else "base_only"
    cache_subdir = f"{base_name}__{feature_str}"

    # Avoid filesystem component length limits (commonly 255 bytes) while keeping stable uniqueness.
    max_component_len = 240
    if len(cache_subdir) > max_component_len:
        digest = hashlib.sha256(cache_subdir.encode("utf-8")).hexdigest()[:16]
        prefix_budget = max_component_len - len(digest) - 2
        safe_prefix = cache_subdir[: max(32, prefix_budget)]
        cache_subdir = f"{safe_prefix}__{digest}"

    if base_cache_dir:
        return os.path.join(base_cache_dir, cache_subdir)
    else:
        return cache_subdir


def create_compositional_tokenizer_bundle(args):
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    language = getattr(args, "language", "en")
    morphology_mode = getattr(args, "morphology_mode", "atomic")
    clitic_mode = getattr(args, "clitic_mode", "surface2")
    tokenization_mode = getattr(args, "tokenization_mode", "single_id")
    skip_multi_token_bases = bool(getattr(args, "skip_multi_token_bases", False))

    # Fail fast: in single_id mode, multi-token bases are unsupported during extension.
    # Requiring the skip flag here avoids spending ~20 minutes building the map only to fail late.
    if tokenization_mode == "single_id" and not skip_multi_token_bases:
        raise ValueError(
            "single_id mode requires --skip_multi_token_bases to avoid unsupported multi-token base entries.\n"
            "Use one of:\n"
            "  1. Add --skip_multi_token_bases (keeps multi-token inflections), OR\n"
            "  2. Switch to --tokenization_mode dual_stream."
        )

    # Language-specific handling of English-only transformations
    decompose_articles = args.decompose_articles
    decompose_prepositions = args.decompose_prepositions
    modifier_generation_mode = _resolve_modifier_generation_mode(args)
    phrase_space_prefix = getattr(args, "decompose_article_prep_space_prefix", False)

    if language.lower() != "en":
        # Articles: English-only, no customization option -> always disable
        if decompose_articles:
            print(f"WARNING: Articles decomposition (a/the/an) is English-only.")
            print(f"         Disabling for language '{language}'.")
            decompose_articles = False

        # Prepositions: Honor if user provided custom list, otherwise disable
        if decompose_prepositions:
            print(f"WARNING: Preposition decomposition defaults to English prepositions.")
            print(f"         Disabling for language '{language}'.")
            print(
                f"         To enable, provide --preposition_list with {language}-specific prepositions."
            )
            decompose_prepositions = False

    if not (decompose_articles or decompose_prepositions):
        phrase_space_prefix = False

    transform_collision_mode = getattr(args, "transform_collision_mode", "off")
    min_derivation_count = int(getattr(args, "min_derivation_count", 50))

    separate_global_space_cap = tokenization_mode == "dual_stream"
    morph_decompose_spaces = (not args.dont_decompose_spaces) and (not separate_global_space_cap)
    morph_decompose_capitalized = (not args.dont_decompose_capitalized) and (
        not separate_global_space_cap
    )
    if separate_global_space_cap and (
        not args.dont_decompose_spaces or not args.dont_decompose_capitalized
    ):
        print(
            "Dual-stream mode: keeping decomposition map morphology-focused "
            "(global space/capitalization handled separately in SequenceMap)"
        )
    if args.include_derivations:
        print(f"Using derivation minimum-count threshold: {min_derivation_count}")

    base_tokens, decomposition_map, types_to_words, ambiguous, conflicts, duplicates, unimorph = (
        get_vocabulary_decomposition(
            base_tokenizer,
            space_prefix=" ",
            language=language,
            morphology_mode=morphology_mode,
            clitic_mode=clitic_mode,
            unimorph_root=getattr(args, "unimorph_root", None),
            unimorph_inflections_path=getattr(args, "unimorph_inflections_path", None),
            unimorph_derivations_path=getattr(args, "unimorph_derivations_path", None),
            include_base_only=True,
            decompose_spaces=morph_decompose_spaces,
            skip_if_no_space_prefix=args.skip_if_no_space_prefix,
            decompose_capitalization=morph_decompose_capitalized,
            use_relative_space_cap_transforms=getattr(
                args, "use_relative_space_cap_transforms", False
            ),
            canonicalize_token_strings=getattr(args, "canonicalize_token_strings", False),
            transform_collision_mode=transform_collision_mode,
            include_derivations=args.include_derivations,
            min_derivation_count=min_derivation_count,
            allow_chained_derivations=args.allow_chained_derivations,
            use_type_str_as_type=args.use_type_str_as_type,
            no_diacritics=getattr(args, "no_diacritics", False),
            filter_propernouns_and_aux=args.filter_propernouns_and_aux,
            include_non_resource_latin_tokens=args.include_non_resource_latin_tokens,
            include_non_latin_tokens=args.include_non_latin_tokens,
            skip_stop_words=args.skip_stop_words,
            merge_plural_and_present_singular=args.merge_plural_and_present_singular,
            prefer_popular_base_conflicts=getattr(args, "prefer_popular_base_conflicts", True),
            # New transformation groups
            decompose_punctuation=args.decompose_punctuation,
            decompose_articles=decompose_articles,  # Use local var (language-gated)
            decompose_prepositions=decompose_prepositions,  # Use local var (language-gated)
            decompose_article_prep_space_prefix=phrase_space_prefix,
            preposition_list=args.preposition_list,
            modifier_generation_mode=modifier_generation_mode,
            # Multi-token handling
            skip_multi_token_words=args.skip_multi_token_words,
            skip_three_token_words=args.skip_three_token_words,
            skip_four_token_words=getattr(args, "skip_four_token_words", False),
            skip_multi_token_bases=getattr(args, "skip_multi_token_bases", False),
            multitoken_allowed_groups=args.multitoken_allowed_groups or [],
            # Smoke testing
            max_vocab_items=getattr(args, "max_vocab_items", None),
            log_second_pass_bases=bool(getattr(args, "log_second_pass_bases", False)),
            enable_build_optimizations=bool(getattr(args, "enable_build_optimizations", False)),
        )
    )

    syntactic_decomposition_map = {}

    # Determine active morphological groups based on morphology_mode
    from decomposition_utils import MORPHOLOGICAL_GROUPS

    candidate_morph_groups = MORPHOLOGICAL_GROUPS if morphology_mode == "compositional" else None

    article_space_prefix_enabled = phrase_space_prefix and decompose_articles
    prep_space_prefix_enabled = phrase_space_prefix and decompose_prepositions

    type_names_to_int, types_loss_indices_map = get_label_maps_from_decomposition_map(
        decomposition_map,
        morphology_mode=morphology_mode,
        active_morphological_groups=candidate_morph_groups,
        force_articles=decompose_articles,
        force_prepositions=decompose_prepositions,
        force_punctuation=args.decompose_punctuation,
        force_article_space_prefix=article_space_prefix_enabled,
        force_prep_space_prefix=prep_space_prefix_enabled,
        preposition_list=args.preposition_list,
    )

    pass

    # Extract actual active morphological groups (only those with values)
    if morphology_mode == "compositional":
        active_morph_groups = [g for g in MORPHOLOGICAL_GROUPS if g in types_loss_indices_map]
    else:
        active_morph_groups = None

    # Get tokenization mode (default to single_id for backward compatibility)
    tokenization_mode = getattr(args, "tokenization_mode", "single_id")

    # This is the modern, recommended approach for compositional tokenization
    use_unified_modifiers = tokenization_mode == "dual_stream"

    # Instead, we use the base tokenizer as-is and do post-processing via DualStreamTokenizer
    (
        extended_tokenizer,
        final_decomposition_map,
        special_tokens_map,
        scaffold_vocab,
        base_token_to_valid_types,
        transformation_names_to_int,
        existing_inflection_ids,
        decomposed_to_original_id,
        decomposed_to_original_seq,
        negative_type_ids,
        na_type_ids,
    ) = extend_tokenizer_by_decomposition_map(
        base_tokenizer,
        base_tokens,
        decomposition_map,
        space_prefix=" ",
        transformations_to_int=type_names_to_int,
        skip_multi_token_words=args.skip_multi_token_words,
        skip_three_token_words=args.skip_three_token_words,
        skip_four_token_words=getattr(args, "skip_four_token_words", False),
        multitoken_allowed_groups=args.multitoken_allowed_groups or [],
        tokenization_mode=tokenization_mode,
    )

    tokenizer = ScaffoldTokenizer(
        extended_tokenizer=extended_tokenizer,
        base_tokenizer=base_tokenizer,
        scaffold_vocab=scaffold_vocab,
        # Explicitly set special tokens from base tokenizer
        bos_token=base_tokenizer.bos_token,
        eos_token=base_tokenizer.eos_token,
        unk_token=base_tokenizer.unk_token,
        sep_token=base_tokenizer.sep_token,
        pad_token=base_tokenizer.pad_token,
        cls_token=base_tokenizer.cls_token,
        mask_token=base_tokenizer.mask_token,
    )

    # Prepare model initialization data
    all_inflections = set(final_decomposition_map.keys()) - set(base_tokens.values())
    base_token_indices = list(base_tokens.values())
    untouched_indices = (
        set(base_tokenizer.get_vocab().values())
        - set(base_token_indices)
        - set(existing_inflection_ids)
    )
    untouched_indices = sorted(list(untouched_indices))
    non_inflection_indices = set(base_token_indices) | set(untouched_indices)

    base_or_inflection_token_ids = list(set(base_token_indices) | all_inflections)
    base_or_inflection_token_ids = [
        v for v in base_or_inflection_token_ids if not isinstance(v, tuple)
    ]
    token_id_to_base_id_mapping = {
        token_id: base_token_id for token_id, (base_token_id, _) in final_decomposition_map.items()
    }
    for untouched_i in untouched_indices:
        if untouched_i not in token_id_to_base_id_mapping:
            token_id_to_base_id_mapping[untouched_i] = untouched_i

    unique_base_ids = sorted(set(token_id_to_base_id_mapping.values()) | set(base_token_indices))
    if len(non_inflection_indices) != len(unique_base_ids):
        # Debug mismatch: show which base IDs are missing/extra (with token strings).
        inv_vocab = {v: k for k, v in base_tokenizer.get_vocab().items()}
        missing_base_ids = sorted(set(unique_base_ids) - set(non_inflection_indices))
        extra_non_inflection_ids = sorted(set(non_inflection_indices) - set(unique_base_ids))

        if missing_base_ids:
            print("NOTE: base IDs present in mapping but missing from non_inflection_indices:")
            for base_id in missing_base_ids:
                base_tok = inv_vocab.get(base_id, None)
                base_tok_str = repr(base_tok) if base_tok is not None else "<unk_id>"
                # Show a few example token IDs that map to this base_id
                example_ext_ids = [
                    ext_id
                    for ext_id, b_id in token_id_to_base_id_mapping.items()
                    if b_id == base_id
                ][:3]
                example_ext_toks = [repr(inv_vocab.get(eid, "<unk_id>")) for eid in example_ext_ids]
                print(
                    f"  base_id={base_id} token={base_tok_str} example_ext={list(zip(example_ext_ids, example_ext_toks))}"
                )

        if extra_non_inflection_ids:
            print("NOTE: base IDs present in non_inflection_indices but absent from mapping:")
            for base_id in extra_non_inflection_ids:
                base_tok = inv_vocab.get(base_id, None)
                base_tok_str = repr(base_tok) if base_tok is not None else "<unk_id>"
                print(f"  base_id={base_id} token={base_tok_str}")

    # Base vocab indices are defined by the set of unique base IDs reachable via token_id_to_base_id_mapping.
    # Keep model_init_data['non_inflection_indices'] consistent with that set (it's used to size the base embedding table).
    prev_non_inflection_size = len(non_inflection_indices)
    non_inflection_indices = set(unique_base_ids)
    if len(non_inflection_indices) != prev_non_inflection_size:
        print(
            f"NOTE: Adjusting non_inflection_indices size {prev_non_inflection_size} → {len(non_inflection_indices)} "
            f"to match unique base IDs (dual-stream correctness)."
        )

    transformation_int_to_name = {v: k for k, v in transformation_names_to_int.items()}
    print(transformation_int_to_name)

    type_groups = {}
    transform_groups = []
    for group_name, group_indices in types_loss_indices_map.items():
        transforms = [
            {"name": transformation_int_to_name[i], "id": i}
            for i in range(group_indices[0], group_indices[1])
        ]
        transform_groups.append({"name": group_name, "transforms": transforms})
        type_groups[group_name] = len(transforms)

    # Calculate vocab stats
    inflections_single_token = set()
    tokenizer_vocab_ids = set(base_tokenizer.get_vocab().values())
    for inflection_token_id, (base_token_id, type_ids) in final_decomposition_map.items():
        if inflection_token_id == base_token_id:
            continue
        if inflection_token_id in tokenizer_vocab_ids:
            inflections_single_token.add(inflection_token_id)

    model_init_data = {
        "base_tokenizer_size": len(base_tokenizer),
        "base_token_indices": base_token_indices,
        "base_tokens": base_tokens,
        "final_decomposition_map": final_decomposition_map,
        "transform_groups": transform_groups,
        "untouched_indices": untouched_indices,
        "non_inflection_indices": sorted(list(non_inflection_indices)),
        "token_id_to_base_id_mapping": token_id_to_base_id_mapping,
        "base_or_inflection_token_ids": base_or_inflection_token_ids,
        "type_groups": type_groups,
        "transformation_names_to_int": transformation_names_to_int,
        "types_loss_indices_map": types_loss_indices_map,
        "decomposition_map": decomposition_map,
        "syntactic_decomposition_map": syntactic_decomposition_map,
        "negative_type_ids": negative_type_ids,
        "na_type_ids": na_type_ids,
        "all_inflections": list(all_inflections),
        "inflections_single_token": list(inflections_single_token),
        "morphology_mode": morphology_mode,
        "active_morphological_groups": active_morph_groups,
        "decomposition_filtered": duplicates,
        "drop_morphology_transforms": getattr(args, "drop_morphology_transforms", False),
    }
    print(
        f"# base tokens: {len(base_token_indices)} - # untouched: {len(untouched_indices)} - # single token inflections: {len(inflections_single_token)} - # all inflections: {len(all_inflections)}"
    )

    # Determine tokenization mode and create modifier_map if needed
    tokenization_mode = getattr(args, "tokenization_mode", "single_id")
    modifier_map = None

    print(f"\n=== Bundle Creation Debug ===")
    print(f"tokenization_mode: {tokenization_mode}")
    print(f"args.decompose_articles: {getattr(args, 'decompose_articles', 'MISSING')}")
    print(f"args.decompose_prepositions: {getattr(args, 'decompose_prepositions', 'MISSING')}")
    print(f"args.decompose_punctuation: {getattr(args, 'decompose_punctuation', 'MISSING')}")

    pass
    print(f"============================\n")

    unified_modifier_array = None
    dual_stream_tokenizer_config = None

    # Debug: check condition for UnifiedModifierArray creation
    print(f"=== UnifiedModifierArray Creation ===")
    print(f"tokenization_mode: {tokenization_mode}")
    print(f"Creating 2D unified modifier arrays for dual-stream mode")
    print(f"======================================\n")

    sequence_map = None

    return CompositionalTokenizerBundle(
        tokenizer,
        model_init_data,
        tokenization_mode,
        modifier_map,
        unified_modifier_array,
        dual_stream_tokenizer_config,
        sequence_map,
    )


def get_or_create_tokenizer_bundle(args, cache_base_dir="tokenizer_cache"):
    cache_dir = get_tokenizer_cache_path(args, cache_base_dir)
    if args.overwrite_bundle_cache and os.path.exists(cache_dir):
        print(f"Removing cached tokenizer bundle at {cache_dir}")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        else:
            os.remove(cache_dir)

    if (
        not args.overwrite_bundle_cache
        and os.path.exists(cache_dir)
        and os.path.exists(os.path.join(cache_dir, "tokenizer"))
        and os.path.exists(os.path.join(cache_dir, "model_init_data.pkl"))
    ):
        print(f"Loading cached tokenizer bundle from {cache_dir}")
        cached_bundle = CompositionalTokenizerBundle.load(
            cache_dir, base_tokenizer_name=args.base_tokenizer
        )
        return cached_bundle

    print(f"Creating and caching tokenizer bundle to {cache_dir}")
    bundle = create_compositional_tokenizer_bundle(args)
    bundle.save(cache_dir)
    return bundle
