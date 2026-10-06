import torch
import argparse
import os
import gc
import json
import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from tabulate import tabulate
from tqdm import tqdm
import types
from typing import List, Dict, Union, Optional, Tuple, Any
from collections import defaultdict
from copy import deepcopy
import pickle
import os
from torch import nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed
import logging

from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

pass
pass
pass
from .utils.english_morphology import (
    extend_tokenizer_by_decomposition_map,
    get_label_maps_from_decomposition_map,
)
from .utils.multilingual_decomposition_utils import (
    get_multilingual_vocabulary_decomposition,
    get_language_config,
)
from .utils.english_morphology import get_initial_type_embeddings_from_strings
from .utils.english_morphology import get_ignored_types

pass
from .utils.patchscopes_utils import (
    run_patchscopes_on_additive_hidden_states,
    analyze_patchscopes_results,
)
from .utils.patchscopes_utils import build_word_pairs
from .utils.patchscopes_utils import run_logit_lens_on_hidden_states

pass
pass
pass
pass
pass


# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args, training_args):
    set_seed(args.seed)

    SKIP_MULTI_TOKEN_WORDS = args.skip_multi_token_words
    SKIP_THREE_TOKEN_WORDS = args.skip_three_token_words

    language = getattr(args, "language", "english")
    run_scope = os.path.join(args.model_name.replace("/", "--"), language)
    args.output_dir = os.path.join(args.output_dir, run_scope)
    args.save_path = os.path.join(args.save_path, run_scope)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    if "gemma-2" in args.model_name.lower():
        accelerator = Accelerator()
    else:
        accelerator = Accelerator(mixed_precision=mixed_precision)

    logger.info("Preparing vocabulary decomposition map...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    space_prefix = args.space_prefix

    lang_config = get_language_config(args.language)
    lang_code = lang_config["language_code"]

    # Create decomposition map
    base_tokens, decomposition_map, types_to_words, ambiguous, conflicts, duplicates = (
        get_multilingual_vocabulary_decomposition(
            tokenizer=base_tokenizer,
            language_code=lang_code,
            unimorph_base_path=args.unimorph_root,
            space_prefix=" ",
            include_base_only=True,
            include_all_caps=False,
            override_unimorph_with_inflections=True,
            decompose_spaces=False,
            skip_if_no_space_prefix=args.skip_if_no_space_prefix,
            include_derivations=args.include_derivations,
            min_derivation_count=1000,
            use_type_str_as_type=False,
            decompose_capitalization=lang_config["decompose_capitalization"],
            no_diacritics=args.language in ["hebrew", "arabic"],
        )
    )

    type_names_to_int, types_loss_indices_map = get_label_maps_from_decomposition_map(
        decomposition_map
    )

    orig_decomposition_map = decomposition_map

    vocab_stats = dict()

    if args.run_logit_lens:
        logger.info("Preparing for Logit Lens analysis...")

        logger.info("Loading baseline model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
        )

        model.eval()
        model = accelerator.prepare(model)
        input_embeddings = model.get_input_embeddings().weight.detach()
        output_embeddings = model.get_output_embeddings().weight.detach()

        input_type_embeddings_init, output_type_embeddings_init = (
            get_initial_type_embeddings_from_strings(
                base_tokens,
                decomposition_map,
                base_tokenizer,
                input_embeddings,
                output_embeddings,
                type_names_to_int,
                enforce_allowed_multitypes=not args.allow_multitypes_in_init,
            )
        )

        ignored_types = get_ignored_types()

        run_desc = f"space_decomp={'y' if not args.dont_decompose_spaces else 'n'}_skip={'y' if args.skip_if_no_space_prefix else 'n'}_capitalize_decomp={'y' if not args.dont_decompose_capitalized else 'n'}_deriv={'y' if args.include_derivations else 'n'}_type_str={'y' if args.use_type_str_as_type else 'n'}"
        output_dir = os.path.join(args.output_dir, run_desc, "logit_lens")
        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(args.save_path, run_desc, "logit_lens")
        load_from_file = args.load_from_cache
        save_results = True

        if load_from_file and os.path.isfile(os.path.join(base_path, "results.pickle")):
            with open(os.path.join(base_path, "results.pickle"), "rb") as fp:
                results = pickle.load(fp)
            with open(os.path.join(base_path, "results_by_type.pickle"), "rb") as fp:
                results_by_type = pickle.load(fp)

            word_pairs = build_word_pairs(
                base_tokenizer,
                decomposition_map,
                max_words=None,
                space_prefix=None,
                single_type_only=args.single_type_only,
                multi_type_only=args.multi_type_only,
                skip_multi_token_words=args.skip_multi_token_words,
                skip_single_token_words=args.skip_single_token_words,
                ignored_types=ignored_types,
            )
        else:
            results, results_by_type, word_pairs = run_logit_lens_on_hidden_states(
                model,
                base_tokenizer,
                base_tokens,
                decomposition_map,
                input_type_embeddings_init,
                type_names_to_int,
                last_layer=10,
                skip_multi_token_words=True,
                single_type_only=args.single_type_only,
                multi_type_only=args.multi_type_only,
                batch_size=64,
                space_prefix=None,
                ignored_types=ignored_types,
                max_words=args.max_words,
                top_k=args.logit_lens_top_k,
            )
            if save_results:
                os.makedirs(base_path, exist_ok=True)
                with open(os.path.join(base_path, "results.pickle"), "wb") as fp:
                    pickle.dump(results, fp)
                with open(os.path.join(base_path, "results_by_type.pickle"), "wb") as fp:
                    results_by_type_regular_dict = json.loads(json.dumps(results_by_type))
                    pickle.dump(results_by_type_regular_dict, fp)

        analysis, analysis_by_type, analysis_summary_by_type, resolution_by_type = (
            analyze_patchscopes_results(
                results,
                results_by_type,
                args.last_analysis_layer,
                word_pairs,
                tokenizer=base_tokenizer,
                ignored_types=ignored_types,
                space_prefix=space_prefix,
                top_k=args.logit_lens_top_k,
            )
        )
        x = pd.DataFrame.from_dict(analysis).drop_duplicates("word")
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            print(pd.DataFrame.from_dict(analysis_summary_by_type).T)

        pd.DataFrame.from_dict(analysis_summary_by_type).T.to_csv(
            os.path.join(output_dir, "logit_lens_analysis_summary_by_type.csv")
        )
        pd.DataFrame.from_dict(resolution_by_type).to_csv(
            os.path.join(output_dir, "logit_lens_resolution_layer.csv")
        )
        outliers = x[x["identified_additive"] == 0]["word"].tolist()
        model = accelerator.free_memory(model)

    if args.run_patchscopes:
        logger.info("Preparing for Patchscopes analysis...")

        # get some stats before removal
        _, final_decomposition_map, _, _, _, _, _, _, _, _ = extend_tokenizer_by_decomposition_map(
            base_tokenizer,
            base_tokens,
            decomposition_map,
            transformations_to_int=type_names_to_int,
            skip_multi_token_words=SKIP_MULTI_TOKEN_WORDS,
            skip_three_token_words=SKIP_THREE_TOKEN_WORDS,
        )
        inflections_single_token = set()
        all_inflections = set()
        tokenizer_vocab_ids = set(base_tokenizer.get_vocab().values())
        for inflection_token_id, (base_token_id, type_ids) in final_decomposition_map.items():
            if inflection_token_id == base_token_id:
                continue
            if inflection_token_id in tokenizer_vocab_ids:
                inflections_single_token.add(inflection_token_id)
            all_inflections.add(inflection_token_id)
        logger.info(
            f"Some numbers - before removal --- # base words: {len(decomposition_map)}, # inflections: {len(all_inflections)}, # inflections in vocab: {len(inflections_single_token)}"
        )
        vocab_stats["before_removal"] = {
            "n_base_words": len(decomposition_map),
            "n_inflections": len(all_inflections),
            "n_in_vocab_inflections": len(inflections_single_token),
        }

        logger.info("Comparing detokenized representations...")
        logger.info("Loading baseline model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
        )

        model.eval()
        model = accelerator.prepare(model)
        input_embeddings = model.get_input_embeddings().weight.detach()
        output_embeddings = model.get_output_embeddings().weight.detach()

        input_type_embeddings_init, _ = get_initial_type_embeddings_from_strings(
            base_tokens,
            decomposition_map,
            base_tokenizer,
            input_embeddings,
            output_embeddings,
            type_names_to_int,
            enforce_allowed_multitypes=not args.allow_multitypes_in_init,
        )

        ignored_types = get_ignored_types()

        run_desc = f"space_decomp={'y' if not args.dont_decompose_spaces else 'n'}_skip={'y' if args.skip_if_no_space_prefix else 'n'}_capitalize_decomp={'y' if not args.dont_decompose_capitalized else 'n'}_deriv={'y' if args.include_derivations else 'n'}_type_str={'y' if args.use_type_str_as_type else 'n'}"
        base_path = os.path.join(
            args.save_path, run_desc, "patchscopes", args.patchscopes_prompt.replace(" ", "_")
        )
        output_dir = os.path.join(
            args.output_dir, run_desc, "patchscopes", args.patchscopes_prompt.replace(" ", "_")
        )
        os.makedirs(output_dir, exist_ok=True)
        load_from_file = args.load_from_cache
        save_results = True
        if load_from_file and os.path.isfile(os.path.join(base_path, "results.pickle")):
            with open(os.path.join(base_path, "results.pickle"), "rb") as fp:
                results = pickle.load(fp)
            with open(os.path.join(base_path, "results_by_type.pickle"), "rb") as fp:
                results_by_type = pickle.load(fp)
            word_pairs = build_word_pairs(
                base_tokenizer,
                decomposition_map,
                max_words=None,
                space_prefix=None,
                single_type_only=args.single_type_only,
                multi_type_only=args.multi_type_only,
                skip_multi_token_words=args.skip_multi_token_words,
                skip_single_token_words=args.skip_single_token_words,
                ignored_types=ignored_types,
            )
        else:
            results, results_by_type, word_pairs = run_patchscopes_on_additive_hidden_states(
                model,
                base_tokenizer,
                base_tokens,
                decomposition_map,
                input_type_embeddings_init,
                type_names_to_int,
                patchscopes_prompt=args.patchscopes_prompt,
                last_layer=10,
                single_type_only=args.single_type_only,
                multi_type_only=args.multi_type_only,
                skip_multi_token_words=args.skip_multi_token_words,
                skip_single_token_words=args.skip_single_token_words,
                batch_size=64,
                space_prefix=None,
                ignored_types=ignored_types,
                max_words=args.max_words,
            )
            if save_results:
                os.makedirs(base_path, exist_ok=True)
                with open(os.path.join(base_path, "results.pickle"), "wb") as fp:
                    pickle.dump(results, fp)
                with open(os.path.join(base_path, "results_by_type.pickle"), "wb") as fp:
                    results_by_type_regular_dict = json.loads(json.dumps(results_by_type))
                    pickle.dump(results_by_type_regular_dict, fp)

        analysis, analysis_by_type, analysis_summary_by_type, resolution_by_type = (
            analyze_patchscopes_results(
                results,
                results_by_type,
                args.last_analysis_layer,
                word_pairs,
                tokenizer=base_tokenizer,
                ignored_types=ignored_types,
                space_prefix=space_prefix,
            )
        )
        x = pd.DataFrame.from_dict(analysis).drop_duplicates("word")
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.width", None
        ):
            print(pd.DataFrame.from_dict(analysis_summary_by_type).T)

        pd.DataFrame.from_dict(analysis_summary_by_type).T.to_csv(
            os.path.join(output_dir, "patchscopes_analysis_summary_by_type.csv")
        )

        pd.DataFrame.from_dict(resolution_by_type).to_csv(
            os.path.join(output_dir, "patchscopes_resolution_layer.csv")
        )

        outliers = x[x["identified_additive"] == 0]["word"].tolist()

        model = accelerator.free_memory(model)
    else:
        run_desc = f"space_decomp={'y' if not args.dont_decompose_spaces else 'n'}_skip={'y' if args.skip_if_no_space_prefix else 'n'}_capitalize_decomp={'y' if not args.dont_decompose_capitalized else 'n'}_deriv={'y' if args.include_derivations else 'n'}_type_str={'y' if args.use_type_str_as_type else 'n'}"
        output_dir = os.path.join(args.output_dir, run_desc)
        os.makedirs(output_dir, exist_ok=True)

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
    ) = extend_tokenizer_by_decomposition_map(
        base_tokenizer,
        base_tokens,
        decomposition_map,
        transformations_to_int=type_names_to_int,
        skip_multi_token_words=SKIP_MULTI_TOKEN_WORDS,
        skip_three_token_words=SKIP_THREE_TOKEN_WORDS,
    )

    # get some vocab stats
    inflections_single_token = set()
    all_inflections = set()
    tokenizer_vocab = set(base_tokenizer.get_vocab().keys())
    tokenizer_vocab_ids = set(base_tokenizer.get_vocab().values())
    for inflection_token_id, (base_token_id, type_ids) in final_decomposition_map.items():
        if inflection_token_id == base_token_id:
            continue
        if inflection_token_id in tokenizer_vocab_ids:
            inflections_single_token.add(inflection_token_id)
        all_inflections.add(inflection_token_id)
    logger.info(
        f"Some numbers --- # base words: {len(decomposition_map)}, # inflections: {len(all_inflections)}, # inflections in vocab: {len(inflections_single_token)}"
    )
    vocab_stats["after_removal"] = {
        "n_base_words": len(decomposition_map),
        "n_inflections": len(all_inflections),
        "n_in_vocab_inflections": len(inflections_single_token),
    }
    with open(os.path.join(output_dir, f"vocab_stats.json"), "w") as f:
        json.dump(vocab_stats, f, indent=4)

    config = vars(args)
    with open(os.path.join(output_dir, f"config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    logger.info(f"Results saved to: {output_dir}")

    return


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate word-level vocabulary expansion success."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="utter-project/EuroLLM-9B")
    parser.add_argument(
        "--unimorph_root",
        type=str,
        default=None,
        help="UniMorph directory (defaults to $UNIMORPH_ROOT or resources/unimorph).",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/patchscopes")
    parser.add_argument("--space_prefix", type=str, default=" ")
    parser.add_argument("--language", type=str, default="spanish")

    parser.add_argument("--skip_if_no_space_prefix", action="store_true", default=False)
    parser.add_argument("--dont_decompose_spaces", action="store_true", default=True)
    parser.add_argument("--dont_decompose_capitalized", action="store_true", default=False)

    parser.add_argument("--last_analysis_layer", type=int, default=10)
    parser.add_argument("--run_logit_lens", action="store_true", default=True)
    parser.add_argument("--logit_lens_top_k", type=int, default=None)
    parser.add_argument("--run_patchscopes", action="store_true", default=True)
    parser.set_defaults(run_patchscopes_second_init=False)
    parser.add_argument("--load_from_cache", action="store_true", default=False)
    parser.add_argument("--max_words", type=int, default=None)
    parser.add_argument("--save_path", type=str, default="./cache/patchscopes")
    parser.add_argument("--patchscopes_prompt", type=str, default="{word}, {word}, {word}, {word},")

    parser.add_argument("--skip_single_token_words", action="store_true", default=False)
    parser.add_argument("--multi_type_only", action="store_true", default=False)
    parser.add_argument("--single_type_only", action="store_true", default=False)

    parser.add_argument("--skip_multi_token_words", action="store_true", default=False)
    parser.add_argument("--skip_three_token_words", action="store_true", default=False)

    parser.add_argument("--include_derivations", action="store_true", default=False)
    parser.set_defaults(use_type_str_as_type=False)
    parser.set_defaults(allow_multitypes_in_init=False)

    args, _ = parser.parse_known_args()

    parser = HfArgumentParser(TrainingArguments)
    training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    return args, training_args


if __name__ == "__main__":
    args, training_args = parse_args()
    main(args, training_args)
