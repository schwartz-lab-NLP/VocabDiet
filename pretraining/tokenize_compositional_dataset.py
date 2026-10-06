"""
Compositional dataset tokenization script for language model pretraining.
Uses ScaffoldTokenizer with decomposition mappings.

Examples:
    python tokenize_compositional_dataset.py --max_tokens 2B --int32
    python tokenize_compositional_dataset.py --max_tokens 2B --skip_multi_token_words
    python tokenize_compositional_dataset.py --max_tokens 2B --int32 --overwrite_bundle_cache
    python tokenize_compositional_dataset.py --max_tokens 1B --overwrite_bundle_cache
    python tokenize_compositional_dataset.py --dataset allenai/c4 --dataset_config en --text_field text --max_tokens 500M
"""

import os
import json
import argparse
import sys
import multiprocessing as mp
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import itertools

from compositional_tokenizer_bundle import (
    get_or_create_tokenizer_bundle,
    CompositionalTokenizerBundle,
    get_tokenizer_cache_path,
)
from tokenize_dataset import parse_token_count, write_datafile, get_dataset, iter_dataset_until_stop

pass
pass


# Global variables for multiprocessing
TOKENIZER = None
TEXT_FIELD = None
DTYPE = None
TOKENIZATION_MODE = None
MODIFIER_MAP = None
DECOMPOSITION_MAP = None
DECOMPOSITION_TOKEN_IDS = None
TRANSFORMATION_NAMES_TO_INT = None
NEW_TRANSFORM_GROUP_NAMES = None
TYPE_CONFIG = None
DECOMPOSE_ARTICLES = None
DECOMPOSE_PREPOSITIONS = None
DECOMPOSE_PUNCTUATION = None

USE_UNIFIED_MODIFIERS = None
UNIFIED_MODIFIER_ARRAY = None
NUM_MODIFIER_GROUPS = None
MODIFIER_DTYPE = None

# DualStreamTokenizer for multi-token sequence detection
DUAL_STREAM_TOKENIZER = None
ANALYSIS_ENABLED = False

# Cache for token ID sets (built once per worker)
ARTICLE_TOKEN_IDS = None
PREP_TOKEN_IDS = None
PREFIX_PUNCT_TOKEN_IDS = None
SUFFIX_PUNCT_TOKEN_IDS = None

PRELOADED_BUNDLE = None


def _configure_compositional_globals(
    bundle,
    text_field,
    dtype,
    decompose_articles=False,
    decompose_prepositions=False,
    decompose_punctuation=False,
    analysis_enabled=False,
    use_na_modifiers_for_non_decomposed=True,
    word_boundary_safety=False,
    multi_token_modifier_position="last",
    preposition_list=None,
    enable_build_optimizations=False,
):
    global TOKENIZER, TEXT_FIELD, DTYPE, TOKENIZATION_MODE, MODIFIER_MAP, DECOMPOSITION_MAP
    global DECOMPOSITION_TOKEN_IDS
    global TRANSFORMATION_NAMES_TO_INT, NEW_TRANSFORM_GROUP_NAMES, TYPE_CONFIG
    global DECOMPOSE_ARTICLES, DECOMPOSE_PREPOSITIONS, DECOMPOSE_PUNCTUATION
    global ARTICLE_TOKEN_IDS, PREP_TOKEN_IDS, PREFIX_PUNCT_TOKEN_IDS, SUFFIX_PUNCT_TOKEN_IDS
    global USE_UNIFIED_MODIFIERS, UNIFIED_MODIFIER_ARRAY, NUM_MODIFIER_GROUPS, MODIFIER_DTYPE
    global DUAL_STREAM_TOKENIZER
    global ANALYSIS_ENABLED

    TOKENIZER = bundle.tokenizer
    TEXT_FIELD = text_field
    DTYPE = dtype
    TOKENIZATION_MODE = bundle.tokenization_mode
    MODIFIER_MAP = bundle.modifier_map
    DECOMPOSITION_MAP = bundle.model_init_data.get("final_decomposition_map", {})
    DECOMPOSITION_TOKEN_IDS = set(DECOMPOSITION_MAP.keys())
    TRANSFORMATION_NAMES_TO_INT = bundle.model_init_data.get("transformation_names_to_int", {})
    NEW_TRANSFORM_GROUP_NAMES = bundle.modifier_map.type_groups if bundle.modifier_map else []

    USE_UNIFIED_MODIFIERS = bundle.unified_modifier_array is not None
    UNIFIED_MODIFIER_ARRAY = bundle.unified_modifier_array
    NUM_MODIFIER_GROUPS = (
        bundle.unified_modifier_array.num_groups if bundle.unified_modifier_array else 0
    )
    MODIFIER_DTYPE = None

    # Create TypeConfig for group slice information
    from modeling_compositional import TypeConfig

    TYPE_CONFIG = TypeConfig(
        bundle.model_init_data.get("type_groups", {}),
        transformation_names_to_int=TRANSFORMATION_NAMES_TO_INT,
    )

    # Store decomposition flags
    DECOMPOSE_ARTICLES = decompose_articles
    DECOMPOSE_PREPOSITIONS = decompose_prepositions
    DECOMPOSE_PUNCTUATION = decompose_punctuation

    pass

    DUAL_STREAM_TOKENIZER = None
    pass

    # Worker initialization complete (print removed to prevent multiprocessing stdout deadlock)
    ANALYSIS_ENABLED = analysis_enabled


def init_compositional_worker(
    bundle_path,
    text_field,
    dtype,
    base_tokenizer_name=None,
    decompose_articles=False,
    decompose_prepositions=False,
    decompose_punctuation=False,
    analysis_enabled=False,
    use_na_modifiers_for_non_decomposed=True,
    word_boundary_safety=False,
    multi_token_modifier_position="last",
    preposition_list=None,
    use_preloaded_bundle=False,
    enable_build_optimizations=False,
):
    global PRELOADED_BUNDLE
    if use_preloaded_bundle and PRELOADED_BUNDLE is not None:
        bundle = PRELOADED_BUNDLE
    else:
        bundle = CompositionalTokenizerBundle.load(
            bundle_path, base_tokenizer_name=base_tokenizer_name
        )
    _configure_compositional_globals(
        bundle,
        text_field,
        dtype,
        decompose_articles=decompose_articles,
        decompose_prepositions=decompose_prepositions,
        decompose_punctuation=decompose_punctuation,
        analysis_enabled=analysis_enabled,
        use_na_modifiers_for_non_decomposed=use_na_modifiers_for_non_decomposed,
        word_boundary_safety=word_boundary_safety,
        multi_token_modifier_position=multi_token_modifier_position,
        preposition_list=preposition_list,
        enable_build_optimizations=enable_build_optimizations,
    )


def tokenize(doc):
    """Tokenize a document using the global tokenizer.

    Returns:
        In single_id mode: np.array of token IDs
        In dual_stream mode: tuple of (input_ids_np, modifier_ids_np)
    """
    # Add EOS token at the start of each document (for document separation)
    if hasattr(TOKENIZER, "eos_token_id") and TOKENIZER.eos_token_id is not None:
        eos_token_id = TOKENIZER.eos_token_id
    else:
        # TODO need cleaner solution here for tokenizers that don't have this special token
        eos_token_id = TOKENIZER.encode("<|endoftext|>", add_special_tokens=False)[0]

    # Single-ID mode: return token IDs as before
    tokens = [eos_token_id]
    doc_tokens = TOKENIZER.encode(doc[TEXT_FIELD], add_special_tokens=False)
    tokens.extend(doc_tokens)

    tokens_np = np.array(tokens, dtype=DTYPE)
    if DTYPE == np.uint16:
        assert (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    if ANALYSIS_ENABLED:
        metadata = {
            "raw_len": len(tokens),
            "compressed_len": len(tokens),
            "tokens_removed": 0,
            "sequence_match_count": 0,
            "multi_token_base_count": 0,
            "multi_token_base_token_count": 0,
            "prefix_modifier_count": 0,
            "suffix_modifier_count": 0,
        }
        return tokens_np, metadata
    return tokens_np


def write_npy_datafile(filename, toks, dtype):
    assert len(toks) < 2**31, "token count too large"
    if not isinstance(toks, np.ndarray):
        toks_np = np.array(toks, dtype=dtype)
    else:
        toks_np = toks if toks.dtype == dtype else toks.astype(dtype)
    print(f"writing {len(toks_np):,} tokens to {filename}")
    mmap = np.memmap(filename, mode="w+", dtype=dtype, shape=toks_np.shape)
    mmap[:] = toks_np
    mmap.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Compositional dataset tokenization for pretraining"
    )
    parser.add_argument(
        "--dataset", type=str, default="HuggingFaceFW/fineweb", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="sample-10BT", help="Dataset configuration"
    )
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--shard_size", type=int, default=10**8, help="Size of each shard in tokens"
    )
    parser.add_argument(
        "--max_tokens",
        type=str,
        default=None,
        help="Maximum tokens to process (e.g., '1B', '800M')",
    )
    parser.add_argument(
        "--min_val_tokens",
        type=str,
        default=None,
        help="Require at least this many validation tokens to be written (e.g., '100M').",
    )
    parser.add_argument(
        "--min_train_tokens",
        type=str,
        default=None,
        help="Require at least this many training tokens to be written (e.g., '100M').",
    )
    parser.add_argument(
        "--val_docs",
        type=int,
        default=10000,
        help="Number of initial documents to reserve for validation "
        "(same validation text across runs). Set 0 for legacy shard-based split.",
    )
    parser.add_argument(
        "--text_field", type=str, default="text", help="Field name containing text in dataset"
    )

    # Compositional tokenizer args
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default="tokenizer_cache",
        help="Directory to save/load tokenizer cache",
    )
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="openai-community/gpt2",
        help="Base tokenizer to extend (HuggingFace Hub name or local path)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code for UniMorph selection (default: en)",
    )
    parser.set_defaults(morphology_mode="atomic")
    parser.set_defaults(clitic_mode="surface2")
    parser.add_argument(
        "--unimorph_root",
        type=str,
        default=os.environ.get(
            "UNIMORPH_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "unimorph"),
        ),
        help="Base path to UniMorph language directories (default: UNIMORPH_ROOT or resources/unimorph)",
    )
    parser.add_argument(
        "--unimorph_inflections_path",
        type=str,
        default=None,
        help="Explicit UniMorph inflections path (overrides --language/--unimorph_root)",
    )
    parser.add_argument(
        "--unimorph_derivations_path",
        type=str,
        default=None,
        help="Explicit UniMorph derivations path",
    )
    parser.set_defaults(no_diacritics=False)
    parser.add_argument(
        "--skip_multi_token_words",
        action="store_true",
        default=False,
        help="Skip multi-token inflections when extending tokenizer vocab",
    )
    parser.add_argument(
        "--skip_three_token_words",
        action="store_true",
        default=False,
        help="Skip inflections with 3+ tokens",
    )
    parser.add_argument(
        "--skip_four_token_words",
        action="store_true",
        default=False,
        help="Skip inflections with 4+ tokens",
    )
    parser.add_argument(
        "--skip_multi_token_bases",
        action="store_true",
        default=False,
        help="Skip multi-token base words from decomposition map construction "
        "(required for single_id mode)",
    )
    parser.add_argument("--dont_decompose_spaces", action="store_true", default=False)
    parser.add_argument("--skip_if_no_space_prefix", action="store_true", default=True)
    parser.add_argument(
        "--allow_no_space_prefix",
        action="store_true",
        default=False,
        help="Allow base tokens without space prefix for decomposition (sets skip_if_no_space_prefix=False).",
    )
    parser.add_argument("--dont_decompose_capitalized", action="store_true", default=False)
    parser.set_defaults(use_relative_space_cap_transforms=False)
    parser.set_defaults(canonicalize_token_strings=False)
    parser.set_defaults(transform_collision_mode="off")
    parser.set_defaults(filter_transform_collisions=False)
    parser.add_argument("--include_derivations", action="store_true", default=True)
    parser.add_argument(
        "--min_derivation_count",
        type=int,
        default=50,
        help="Minimum sdsdUniMorph derivation type frequency required to include derivation transforms.",
    )
    parser.set_defaults(prefer_popular_base_conflicts=True)
    parser.set_defaults(drop_morphology_transforms=False)
    parser.set_defaults(allow_chained_derivations=False)
    parser.add_argument("--use_type_str_as_type", action="store_true", default=False)
    parser.add_argument("--skip_stop_words", action="store_true", default=False)
    parser.add_argument("--filter_propernouns_and_aux", action="store_true", default=False)
    parser.add_argument("--merge_plural_and_present_singular", action="store_true", default=False)
    parser.set_defaults(include_non_resource_latin_tokens=True)
    parser.set_defaults(include_non_latin_tokens=True)
    # New transformation groups (opt-in)
    parser.set_defaults(decompose_punctuation=False)
    parser.set_defaults(decompose_articles=False)
    parser.set_defaults(decompose_prepositions=False)
    parser.set_defaults(decompose_article_prep_space_prefix=False)
    parser.set_defaults(preposition_list=None)
    parser.set_defaults(modifier_generation_mode="auto")
    parser.add_argument(
        "--no_na_modifiers_for_non_decomposed",
        action="store_true",
        default=False,
        help="Use empty (no_*) modifiers instead of NA for tokens not in the decomposition map",
    )
    parser.set_defaults(multitoken_allowed_groups=[])
    parser.set_defaults(tokenization_mode="single_id")
    parser.set_defaults(word_boundary_safety=False)
    parser.set_defaults(multi_token_modifier_position="last")
    parser.set_defaults(prune_punct_tokens=False)
    parser.set_defaults(possessive_separate_group=False)
    parser.add_argument("--overwrite_bundle_cache", action="store_true", default=False)
    parser.set_defaults(enable_build_optimizations=False)
    parser.add_argument("--int32", action="store_true", default=False)
    parser.add_argument(
        "--output_format",
        choices=["bin", "npy", "both"],
        default="bin",
        help="Write output shards as .bin, .npy, or both (default: bin)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for tokenization (default: cpu_count - 2)",
    )
    parser.add_argument(
        "--pool_chunksize",
        type=int,
        default=16,
        help="Multiprocessing imap chunksize (default: 16)",
    )
    parser.add_argument(
        "--mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: spawn)",
    )
    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=None,
        help="Restart workers after this many tasks to limit memory growth",
    )
    # Analysis options (enabled by default)
    parser.add_argument(
        "--analysis_disable",
        action="store_true",
        default=False,
        help="Disable tokenizer analysis and dashboard generation",
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="Output directory for analysis dashboard (default: DATA_CACHE_DIR/analysis_dashboard)",
    )
    parser.add_argument(
        "--analysis_sample_docs",
        type=int,
        default=24,
        help="Number of dataset documents to sample for tokenizer validation",
    )
    parser.add_argument(
        "--analysis_max_samples",
        type=int,
        default=12,
        help="Max number of sample analyses to include in the dashboard",
    )
    parser.add_argument(
        "--analysis_combo_sample_limit",
        type=int,
        default=200000,
        help="Max number of tokens sampled for modifier combination stats",
    )
    parser.add_argument(
        "--analysis_token_limit",
        type=int,
        default=200000000,
        help="Max compressed tokens to include in tokenizer analysis (default: 200M)",
    )
    parser.add_argument(
        "--analysis_test_texts",
        type=str,
        nargs="*",
        default=None,
        help="Custom texts to include in tokenizer validation samples",
    )
    parser.add_argument(
        "--analysis_collision_limit",
        type=int,
        default=50,
        help="Max collision entries to report in tokenizer analysis",
    )

    args = parser.parse_args()
    if args.allow_no_space_prefix:
        args.skip_if_no_space_prefix = False
    mp.set_start_method(args.mp_start_method, force=True)

    DATA_CACHE_DIR = get_tokenizer_cache_path(
        args,
        os.path.join(
            os.path.dirname(__file__),
            args.output_dir,
            "vocab_diet",
            f"{args.dataset.split('/')[-1]}_{args.dataset_config}",
        ),
    )

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    analysis_dir = None

    dataset = get_dataset(args.dataset, args.dataset_config, streaming=True)

    max_tokens = parse_token_count(args.max_tokens) if args.max_tokens else float("inf")
    min_val_tokens = parse_token_count(args.min_val_tokens) if args.min_val_tokens else None
    min_train_tokens = parse_token_count(args.min_train_tokens) if args.min_train_tokens else None
    if max_tokens != float("inf"):
        min_required_total = (min_val_tokens or 0) + (min_train_tokens or 0)
        if min_required_total > max_tokens:
            raise ValueError(
                f"Requested minimum split tokens ({min_required_total:,}) exceed --max_tokens ({int(max_tokens):,}). "
                "Increase --max_tokens or lower --min_val_tokens/--min_train_tokens."
            )
    if args.val_docs == 0 and max_tokens != float("inf") and max_tokens <= args.shard_size:
        raise ValueError(
            f"--max_tokens ({int(max_tokens):,}) must be greater than --shard_size ({args.shard_size:,}) "
            "to produce both val and train shards. "
            "The first shard is reserved for val. "
            "Use a smaller --shard_size, or set --val_docs > 0 for a fixed text validation split."
        )

    if args.num_workers is not None:
        nprocs = args.num_workers
    else:
        nprocs = max(1, os.cpu_count() - 2)
    print(
        f"Using {nprocs} worker processes for tokenization (start method: {args.mp_start_method})"
    )
    if args.val_docs > 0:
        print(f"Using fixed validation split: first {args.val_docs:,} documents -> val")
    if min_val_tokens is not None:
        print(f"Minimum validation tokens required: {min_val_tokens:,}")
    if min_train_tokens is not None:
        print(f"Minimum training tokens required: {min_train_tokens:,}")
    total_tokens_processed = 0

    bundle = get_or_create_tokenizer_bundle(args)
    model_init_data = bundle.model_init_data
    total_types = len(model_init_data.get("transformation_names_to_int", {}))
    final_decomp = model_init_data.get("final_decomposition_map", {})

    def _detect_type_ids_format(sample_items, total_types):
        for _, type_ids in sample_items:
            if type_ids is None or isinstance(type_ids, str):
                continue
            if hasattr(type_ids, "tolist"):
                type_ids = type_ids.tolist()
            try:
                size = len(type_ids)
            except TypeError:
                continue
            if size == 0:
                continue
            if size == total_types:
                is_one_hot = True
                for val in type_ids:
                    if isinstance(val, bool):
                        continue
                    if isinstance(val, int):
                        if val not in (0, 1):
                            is_one_hot = False
                            break
                    elif isinstance(val, float):
                        if val < -1e-6 or val > 1.0 + 1e-6:
                            is_one_hot = False
                            break
                    else:
                        is_one_hot = False
                        break
                return "one_hot" if is_one_hot else "indices"
            return "indices"
        return "unknown"

    if total_types <= 0:
        type_ids_format = "unknown (no transformation names)"
        print(f"Type IDs format check: {type_ids_format} (total_types={total_types})")
    elif not final_decomp:
        type_ids_format = "unknown (empty final_decomposition_map)"
        print(f"Type IDs format check: {type_ids_format} (total_types={total_types})")
    else:
        sample_items = itertools.islice(final_decomp.values(), 5)
        type_ids_format = _detect_type_ids_format(sample_items, total_types)
        print(f"Type IDs format check: {type_ids_format} (total_types={total_types})")
        if type_ids_format == "one_hot":
            print(
                "WARNING: final_decomposition_map appears to use one-hot vectors. "
                "Rebuild the tokenizer bundle cache to use list-of-indices."
            )

    vocab_size = len(bundle.tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")
    print(f"Tokenization mode: {bundle.tokenization_mode}")

    dtype = np.int32 if args.int32 else np.uint16
    cache_dir = get_tokenizer_cache_path(args, args.tokenizer_cache_dir)

    is_dual_stream = bundle.tokenization_mode == "dual_stream" and (
        bundle.modifier_map is not None or bundle.unified_modifier_array is not None
    )
    use_unified_mods = bundle.unified_modifier_array is not None
    num_modifier_groups = (
        bundle.unified_modifier_array.num_groups if bundle.unified_modifier_array else 0
    )

    analysis_enabled = (not args.analysis_disable) and is_dual_stream
    use_na_modifiers_for_non_decomposed = not args.no_na_modifiers_for_non_decomposed
    analysis_stats = None
    analysis_samples = {"documents": [], "sentences": []}
    analysis_meta = {}
    single_token_multi_base = []
    collision_report = None
    decomposition_token_ids = set(model_init_data.get("final_decomposition_map", {}).keys())
    pass
    analysis_written = False

    pass

    pass

    pass

    use_preloaded_bundle = args.mp_start_method == "fork"
    if use_preloaded_bundle:
        global PRELOADED_BUNDLE
        PRELOADED_BUNDLE = bundle
        print("Using preloaded tokenizer bundle for forked workers to share SequenceMap memory")
    ctx = mp.get_context(args.mp_start_method)
    stop_event = ctx.Event()
    pool = ctx.Pool(
        nprocs,
        initializer=init_compositional_worker,
        initargs=(
            cache_dir,
            args.text_field,
            dtype,
            args.base_tokenizer,
            args.decompose_articles,
            args.decompose_prepositions,
            args.decompose_punctuation,
            analysis_enabled,
            use_na_modifiers_for_non_decomposed,
            args.word_boundary_safety,
            args.multi_token_modifier_position,
            args.preposition_list,
            use_preloaded_bundle,
            args.enable_build_optimizations,
        ),
        maxtasksperchild=args.maxtasksperchild,
    )
    early_stop = False
    try:
        # Per-split state so validation text can be fixed by document count.
        split_buffers = {
            "val": {
                "tokens": np.empty((args.shard_size,), dtype=dtype),
                "count": 0,
                "shard_index": 0,
                "tokens_written": 0,
                "modifiers": None,
            },
            "train": {
                "tokens": np.empty((args.shard_size,), dtype=dtype),
                "count": 0,
                "shard_index": 0,
                "tokens_written": 0,
                "modifiers": None,
            },
        }

        pass

        def _write_split_shard(split_name, n_tokens):
            state = split_buffers[split_name]
            shard_idx = state["shard_index"]

            base_filename_ids = os.path.join(DATA_CACHE_DIR, f"data_{split_name}_{shard_idx:06d}")
            base_filename_modifiers = None

            tokens_to_write = state["tokens"][:n_tokens]
            if args.output_format in ("bin", "both"):
                write_datafile(f"{base_filename_ids}.bin", tokens_to_write, dtype)
            if args.output_format in ("npy", "both"):
                write_npy_datafile(f"{base_filename_ids}.npy", tokens_to_write, dtype)

            pass

            state["shard_index"] += 1
            state["tokens_written"] += n_tokens
            state["count"] = 0

        def _flush_split(split_name, force=False):
            state = split_buffers[split_name]
            if state["count"] == 0:
                return
            if not force and state["count"] < args.shard_size:
                return
            n_tokens = state["count"] if force else args.shard_size
            _write_split_shard(split_name, n_tokens)

        def _append_to_split(split_name, tokens_chunk, modifiers_chunk):
            state = split_buffers[split_name]
            pos = 0
            while pos < len(tokens_chunk):
                remaining = args.shard_size - state["count"]
                take = min(remaining, len(tokens_chunk) - pos)
                end = pos + take
                state["tokens"][state["count"] : state["count"] + take] = tokens_chunk[pos:end]
                pass
                state["count"] += take
                pos = end
                if state["count"] == args.shard_size:
                    _flush_split(split_name, force=False)

        def _append_legacy_split(tokens_chunk, modifiers_chunk):
            """Legacy behavior: fill first val shard by tokens, then route everything to train."""
            pos = 0
            while pos < len(tokens_chunk):
                val_filled = split_buffers["val"]["tokens_written"] + split_buffers["val"]["count"]
                if val_filled < args.shard_size:
                    split_name = "val"
                    cap = args.shard_size - val_filled
                else:
                    split_name = "train"
                    cap = len(tokens_chunk) - pos
                take = min(cap, len(tokens_chunk) - pos)
                end = pos + take
                chunk_mods = modifiers_chunk[pos:end] if is_dual_stream else None
                _append_to_split(split_name, tokens_chunk[pos:end], chunk_mods)
                pos = end

        def _split_token_count(split_name):
            state = split_buffers[split_name]
            return state["tokens_written"] + state["count"]

        progress_total = None if max_tokens == float("inf") else int(max_tokens)
        progress_bar = tqdm(total=progress_total, unit="tokens", desc="Tokenizing")
        doc_index = 0

        for result in pool.imap(
            tokenize, iter_dataset_until_stop(dataset, stop_event), chunksize=args.pool_chunksize
        ):
            if early_stop:
                continue

            # Unpack result based on mode
            tokens = result
            modifiers = None

            pass

            if args.val_docs > 0:
                split_name = "val" if doc_index < args.val_docs else "train"
                _append_to_split(split_name, tokens, modifiers)
            else:
                _append_legacy_split(tokens, modifiers)

            doc_index += 1
            if args.val_docs > 0 and min_val_tokens is not None and doc_index == args.val_docs:
                final_val_tokens = _split_token_count("val")
                if final_val_tokens < min_val_tokens:
                    raise ValueError(
                        f"Validation split too small with fixed --val_docs={args.val_docs:,}: "
                        f"{final_val_tokens:,} tokens < required {min_val_tokens:,}. "
                        "Increase --val_docs (keeps same first-doc ordering) or lower --min_val_tokens."
                    )
            total_tokens_processed += len(tokens)
            progress_bar.update(len(tokens))
            if total_tokens_processed >= max_tokens:
                early_stop = True
                stop_event.set()

        _flush_split("val", force=True)
        _flush_split("train", force=True)
        progress_bar.close()

        if split_buffers["train"]["tokens_written"] == 0:
            raise ValueError(
                "No train tokens were written. Increase --max_tokens or reduce --val_docs "
                "(or set --val_docs 0 to use legacy shard-based split)."
            )
        val_tokens_written = split_buffers["val"]["tokens_written"]
        train_tokens_written = split_buffers["train"]["tokens_written"]
        if min_val_tokens is not None and val_tokens_written < min_val_tokens:
            raise ValueError(
                f"Validation tokens written ({val_tokens_written:,}) below --min_val_tokens ({min_val_tokens:,}). "
                "Increase --val_docs or --max_tokens."
            )
        if min_train_tokens is not None and train_tokens_written < min_train_tokens:
            raise ValueError(
                f"Train tokens written ({train_tokens_written:,}) below --min_train_tokens ({min_train_tokens:,}). "
                "Increase --max_tokens or reduce --val_docs."
            )

    finally:
        pool.close()
        pool.join()

    print(f"Total tokens processed: {total_tokens_processed:,}")

    pass

    pass


if __name__ == "__main__":
    main()
