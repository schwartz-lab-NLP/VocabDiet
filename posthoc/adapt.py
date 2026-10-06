import torch
import argparse
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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

from .tokenization import ScaffoldTokenizer
from .checkpoint import save_adaptation, model_class_for_config
from .initialization import mean_offsets
from .utils.english_morphology import get_english_unimorph_paths
from .utils.english_morphology import extend_tokenizer_by_decomposition_map
from .utils.english_morphology import (
    get_vocabulary_decomposition,
    get_label_maps_from_decomposition_map,
)
from .utils.english_morphology import get_type_group_idx_from_decomposition_map
from .utils.english_morphology import get_initial_type_embeddings_from_strings
from .utils.english_morphology import get_ignored_types
from .utils.english_morphology import load_ner_cache
from .utils.decomposition import remove_outliers_from_decomposition_map
from .utils.patchscopes_utils import (
    run_patchscopes_on_additive_hidden_states,
    analyze_patchscopes_results,
)
from .utils.patchscopes_utils import build_word_pairs, filter_proper_nouns_from_patchscopes_results
from .utils.input_additive_data_utils import (
    load_lm_dataset,
    tokenize_and_prepare_dataset,
    get_sft_tokenize_func,
)
from .utils.eval_utils import eval_next_word_prediction
from .utils.eval_utils import count_tokens_in_dataset
from .utils.eval_utils import serialize_metrics
from .utils.downstream_utils import (
    evaluate_model as evaluate_english_model,
    multilingual_evaluate_model,
)
from .utils.multilingual_decomposition_utils import (
    get_multilingual_vocabulary_decomposition,
    get_language_config,
)


# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_optimizer_with_custom_lr(
    model,
    base_lr,
    input_lr_factor=1.0,
    output_lr_factor=1.0,
    weight_decay=0.0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
):
    """
    Create an optimizer with different learning rates for input and output embeddings.

    Args:
        model: The model to optimize
        base_lr: Base learning rate (for LoRA and other trainable params)
        input_lr_factor: Multiply base_lr by this for input embeddings (embed_types, base embeddings)
        output_lr_factor: Multiply base_lr by this for output embeddings (lm_head, types_lm_head)
        weight_decay: Weight decay coefficient
        adam_beta1, adam_beta2, adam_epsilon: AdamW parameters

    Returns:
        torch.optim.AdamW optimizer with parameter groups
    """
    # Identify input embedding parameters
    input_embed_params = []
    input_embed_names = []

    # embed_types (input type embeddings)
    if hasattr(model, "embed_types"):
        for name, param in model.embed_types.named_parameters():
            if param.requires_grad:
                input_embed_params.append(param)
                input_embed_names.append(f"embed_types.{name}")

    # Base input embeddings (model.get_input_embeddings())
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is not None:
        for name, param in input_embeddings.named_parameters():
            if param.requires_grad:
                input_embed_params.append(param)
                input_embed_names.append(f"input_embeddings.{name}")

    # Identify output embedding parameters
    output_embed_params = []
    output_embed_names = []

    # lm_head (base output embeddings)
    if hasattr(model, "lm_head"):
        # Handle EfficientMaskedLMHead wrapper
        if hasattr(model.lm_head, "shortened_lm_head"):
            for name, param in model.lm_head.shortened_lm_head.named_parameters():
                if param.requires_grad:
                    output_embed_params.append(param)
                    output_embed_names.append(f"lm_head.shortened_lm_head.{name}")
        else:
            for name, param in model.lm_head.named_parameters():
                if param.requires_grad:
                    output_embed_params.append(param)
                    output_embed_names.append(f"lm_head.{name}")

    # types_lm_head (type output embeddings)
    if hasattr(model, "types_lm_head"):
        for name, param in model.types_lm_head.named_parameters():
            if param.requires_grad:
                output_embed_params.append(param)
                output_embed_names.append(f"types_lm_head.{name}")

    # Collect all other trainable parameters (LoRA, type prediction heads, etc.)
    input_embed_ids = {id(p) for p in input_embed_params}
    output_embed_ids = {id(p) for p in output_embed_params}

    other_params = []
    other_names = []
    for name, param in model.named_parameters():
        if (
            param.requires_grad
            and id(param) not in input_embed_ids
            and id(param) not in output_embed_ids
        ):
            other_params.append(param)
            other_names.append(name)

    # Create parameter groups
    param_groups = []

    if input_embed_params and input_lr_factor != 1.0:
        param_groups.append(
            {
                "params": input_embed_params,
                "lr": base_lr * input_lr_factor,
                "weight_decay": weight_decay,
            }
        )
        logger.info(
            f"Input embedding params ({len(input_embed_params)}): LR = {base_lr * input_lr_factor:.2e}"
        )
        logger.info(
            f"  Parameters: {input_embed_names[:5]}{'...' if len(input_embed_names) > 5 else ''}"
        )

    if output_embed_params and output_lr_factor != 1.0:
        param_groups.append(
            {
                "params": output_embed_params,
                "lr": base_lr * output_lr_factor,
                "weight_decay": weight_decay,
            }
        )
        logger.info(
            f"Output embedding params ({len(output_embed_params)}): LR = {base_lr * output_lr_factor:.2e}"
        )
        logger.info(
            f"  Parameters: {output_embed_names[:5]}{'...' if len(output_embed_names) > 5 else ''}"
        )

    # If factors are 1.0, include those params in the base group
    base_group_params = other_params.copy()
    if input_lr_factor == 1.0 and input_embed_params:
        base_group_params.extend(input_embed_params)
    if output_lr_factor == 1.0 and output_embed_params:
        base_group_params.extend(output_embed_params)

    if base_group_params:
        param_groups.append(
            {
                "params": base_group_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            }
        )
        logger.info(f"Base params ({len(base_group_params)}): LR = {base_lr:.2e}")
        logger.info(f"  Parameters: {other_names[:5]}{'...' if len(other_names) > 5 else ''}")

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
    )

    return optimizer


def main(args, training_args):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Post-hoc adaptation requires a CUDA GPU. CPU component tests run without model downloads."
        )
    set_seed(args.seed)

    SKIP_MULTI_TOKEN_WORDS = args.skip_multi_token_words
    SKIP_THREE_TOKEN_WORDS = args.skip_three_token_words

    def evaluate_model(model, tokenizer, **kwargs):
        if args.language == "english":
            return evaluate_english_model(model, tokenizer, **kwargs)
        cache_args = {
            "use_cache",
            "cache_name",
            "use_per_run_cache",
            "per_run_cache_dir",
            "per_run_stage_name",
            "overwrite_cache",
            "overwrite_per_run_cache",
        }
        return multilingual_evaluate_model(
            model,
            tokenizer,
            args.language,
            **{k: v for k, v in kwargs.items() if k not in cache_args},
        )

    run_scope = os.path.join(args.model_name.replace("/", "--"), args.language)
    args.output_dir = os.path.join(args.output_dir, run_scope)
    args.patchscopes_save_path = os.path.join(args.patchscopes_save_path, run_scope)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    logger.info("Preparing vocabulary decomposition map and extended tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    space_prefix = args.space_prefix

    test_generation_prompt = "They revolted. They revolted 50 years ago. They revolted because"

    if args.language == "english":
        base_tokens, decomposition_map, types_to_words, ambiguous, conflicts, duplicates = (
            get_vocabulary_decomposition(
                base_tokenizer,
                unimorph_inflections_path=get_english_unimorph_paths(args.unimorph_root)[0],
                unimorph_derivations_path=get_english_unimorph_paths(args.unimorph_root)[1],
                space_prefix=" ",
                include_base_only=False,
                decompose_spaces=not args.dont_decompose_spaces,
                skip_if_no_space_prefix=args.skip_if_no_space_prefix,
                decompose_capitalization=not args.dont_decompose_capitalized,
                include_derivations=args.include_derivations,
                use_type_str_as_type=args.use_type_str_as_type,
                merge_plural_and_present_singular=False,
                filter_propernouns_and_aux=True,
                dont_decompose_proper_nouns=args.dont_decompose_proper_nouns,
                ner_cache_dir=args.ner_cache_dir,
                regenerate_ner_cache=args.regenerate_ner_cache,
                dont_use_capitalized_base_words=args.dont_use_capitalized_base_words,
            )
        )

    else:
        lang_config = get_language_config(args.language)
        base_tokens, decomposition_map, types_to_words, ambiguous, conflicts, duplicates = (
            get_multilingual_vocabulary_decomposition(
                tokenizer=base_tokenizer,
                language_code=lang_config["language_code"],
                unimorph_base_path=args.unimorph_root,
                space_prefix=" ",
                include_base_only=False,
                include_all_caps=False,
                override_unimorph_with_inflections=True,
                decompose_spaces=False,
                skip_if_no_space_prefix=args.skip_if_no_space_prefix,
                include_derivations=args.include_derivations,
                min_derivation_count=1000,
                use_type_str_as_type=False,
                decompose_capitalization=lang_config["decompose_capitalization"]
                and not args.dont_decompose_capitalized,
                no_diacritics=args.language == "arabic",
            )
        )

    type_names_to_int, types_loss_indices_map = get_label_maps_from_decomposition_map(
        decomposition_map
    )
    types_map_by_group = get_type_group_idx_from_decomposition_map(decomposition_map)

    orig_decomposition_map = decomposition_map

    vocab_stats = dict()

    if args.remove_patchscopes_mistakes:
        logger.info("Preparing for outlier removal...")

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
        base_path = os.path.join(
            args.patchscopes_save_path,
            run_desc,
            "patchscopes",
            args.patchscopes_prompt.replace(" ", "_"),
        )

        # Track which cache setting we're using for potential decomposition_map adjustment
        loaded_from_opposite_capitalize_setting = False

        output_dir = os.path.join(output_dir, run_desc, args.patchscopes_prompt.replace(" ", "_"))
        os.makedirs(output_dir, exist_ok=True)

        load_from_file = True
        save_results = True
        if load_from_file and os.path.isfile(os.path.join(base_path, "results.pickle")):
            with open(os.path.join(base_path, "results.pickle"), "rb") as fp:
                results = pickle.load(fp)
            with open(os.path.join(base_path, "results_by_type.pickle"), "rb") as fp:
                results_by_type = pickle.load(fp)

            # Filter proper nouns if flag is set (post-filtering, independent of cache)
            if args.dont_decompose_proper_nouns:
                logger.info(
                    "Filtering proper nouns from loaded patchscopes results using NER cache..."
                )
                # Load NER cache
                ner_cache_path = os.path.join(args.ner_cache_dir, "ner_cache.pkl")
                ner_cache = load_ner_cache(ner_cache_path)
                logger.info(f"Loaded NER cache with {len(ner_cache)} entries")

                num_words_before = len(results)
                num_types_before = len(results_by_type)
                results, results_by_type = filter_proper_nouns_from_patchscopes_results(
                    results, results_by_type, ner_cache
                )
                num_words_after = len(results)
                num_types_after = len(results_by_type)
                logger.info(
                    f"Filtered {num_words_before - num_words_after} proper noun words from results"
                )
                logger.info(f"Types reduced from {num_types_before} to {num_types_after}")

            # Build word_pairs from current decomposition_map
            # Note: If loaded from opposite capitalize setting, patchscopes may have extra/missing entries
            # Only words in word_pairs (from current decomposition_map) will be analyzed
            word_pairs = build_word_pairs(
                base_tokenizer,
                decomposition_map,
                max_words=None,
                space_prefix=None,
                single_type_only=False,
                skip_multi_token_words=False,
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
                skip_multi_token_words=False,
                single_type_only=False,
                batch_size=64,
                space_prefix=None,
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
                10,
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

        decomposition_map = remove_outliers_from_decomposition_map(decomposition_map, outliers)

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
        decompose_spaces=not args.dont_decompose_spaces,
        decompose_capitalization=not args.dont_decompose_capitalized,
        predict_negative_types=args.predict_negative_types,
    )

    # Log negative types configuration
    logger.info(f"Using default behavior: {len(negative_type_ids)} types with zero embeddings")

    # get some stats
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
    all_inflections = list(all_inflections)
    all_inflections_token_count = [
        len(decomposed_to_original_seq[tok_id]) for tok_id in all_inflections
    ]
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

    base_tokens_set = set(base_tokens.values())
    base_and_inflections_vocab_ids = list(inflections_single_token | base_tokens_set)
    base_token_ids = list(base_tokens.values())
    inflection_token_ids = list(inflections_single_token)
    inflection_replaced_sequences = list(decomposed_to_original_seq.values())
    transformation_int_to_name = {v: k for k, v in transformation_names_to_int.items()}
    logger.info(f"Transformation names to ID: {transformation_names_to_int}")
    existing_inflection_ids = list(inflections_single_token)
    logger.info("Creating ScaffoldTokenizer with scaffold tokens...")
    tokenizer = ScaffoldTokenizer(
        extended_tokenizer=extended_tokenizer,
        base_tokenizer=base_tokenizer,
        scaffold_vocab=scaffold_vocab,
    )

    single_token_extended_tokenizer, _, _, single_token_scaffold_vocab, _, _, _, _, _, _ = (
        extend_tokenizer_by_decomposition_map(
            base_tokenizer,
            base_tokens,
            decomposition_map,
            transformations_to_int=type_names_to_int,
            skip_multi_token_words=True,
            skip_three_token_words=True,
        )
    )

    single_token_tokenizer = ScaffoldTokenizer(
        extended_tokenizer=single_token_extended_tokenizer,
        base_tokenizer=base_tokenizer,
        scaffold_vocab=single_token_scaffold_vocab,
    )

    metrics = dict()
    downstream_metrics = dict()
    if args.eval_baselines or args.run_generate_tests:
        logger.info("Loading baseline model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
        )

        model.eval()
        model = accelerator.prepare(model)

        if args.eval_baselines:
            downstream_log_dir = (
                os.path.join(output_dir, "baseline") if args.save_downstream_outputs else None
            )
            downstream_metrics["baseline"] = curr_metrics = evaluate_model(
                model,
                base_tokenizer,
                limited=True,
                output_path=downstream_log_dir,
                examples_limit=args.downstream_limit,
                use_cache=False,
                cache_name="baseline",
                use_per_run_cache=False,
                per_run_cache_dir=output_dir,
                per_run_stage_name="baseline",
                overwrite_per_run_cache=args.overwrite_per_run_cache,
                overwrite_cache=args.overwrite_eval_cache,
            )
            logger.info(f"Baseline downstream results:\n{pd.Series(curr_metrics)}")

            final_downstream_metrics = defaultdict(dict)
            for eval_type, eval_results in downstream_metrics.items():
                for task_name, task_results in eval_results.items():
                    final_downstream_metrics[task_name][eval_type] = task_results
            with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
                json.dump(final_downstream_metrics, f, indent=4)

            for metric_name, metric_value in curr_metrics.items():
                logger.info(f"{metric_name}: {metric_value}")

            if args.eval_ppl:
                logger.info("Evaluate baseline model and tokenizer...")
                eval_dataset = load_lm_dataset(
                    args.eval_dataset,
                    language=args.dataset_language,
                    num_examples=args.eval_max_samples,
                )
                eval_dataset = eval_dataset[args.eval_dataset_split]
                eval_dataset = tokenize_and_prepare_dataset(
                    eval_dataset,
                    base_tokenizer,
                    accelerator,
                    max_length=args.eval_max_length,
                    text_col_name=args.eval_dataset_text_col,
                    max_samples=args.eval_max_samples,
                    pack_documents_with_eos=args.pack_documents_with_eos,
                )

                metrics["baseline"] = curr_metrics = eval_next_word_prediction(
                    model,
                    base_tokenizer,
                    eval_dataset,
                    accelerator,
                    base_token_ids=base_token_ids,
                    inflection_token_ids=inflection_token_ids,
                    inflection_replaced_sequences=inflection_replaced_sequences,
                    max_length=args.eval_max_length,
                    max_samples=args.eval_max_samples,
                    reduction="mean",
                    use_lm_preds_for_type_labels=args.use_lm_preds_for_type_labels,
                )
                logger.info(f"Baseline results:\n{pd.Series(curr_metrics)}")

        model = accelerator.free_memory(model)

    logger.info("Loading model...")
    model_args = {
        "types_loss_alpha": args.types_loss_alpha,
        "base_loss_alpha": args.base_loss_alpha,
        "torch_dtype": torch.bfloat16 if mixed_precision == "bf16" else torch.float16,
        "types_vocab_size": len(transformation_names_to_int),
        "model_vocab_size": len(base_tokenizer),
        "extended_vocab_size": len(extended_tokenizer),
        "probe_types": args.probe_types,
        "num_probe_layers": args.types_num_probe_layers,
        "types_prediction_layer": args.types_prediction_layer,
        "base_prediction_layer": args.base_prediction_layer,
        "types_loss_indices": types_loss_indices_map,
        "tokenizer": tokenizer,
        "use_lm_preds_for_type_labels": args.use_lm_preds_for_type_labels,
        "modeling_option": args.modeling_option,
        "lora_ft": args.lora_ft,
        "lora_adapter_toggling": args.lora_adapter_toggling,
        "k_last_layers": args.k_last_layers,
        # Weight tying parameters
        "tie_word_embeddings": args.tie_word_embeddings,
        "use_efficient_tied_embeddings": args.use_efficient_tied_embeddings,
    }
    additive_class = model_class_for_config(AutoConfig.from_pretrained(args.model_name))
    model = additive_class.from_pretrained(args.model_name, **model_args)

    model = accelerator.prepare(model)

    logger.info("Initializing type embeddings...")
    input_type_embeddings_init, output_type_embeddings_init = mean_offsets(
        final_decomposition_map,
        model.get_input_embeddings().weight.detach(),
        model.get_output_embeddings().weight.detach(),
        len(transformation_names_to_int),
        negative_type_ids,
    )

    model.set_input_type_embeddings(input_type_embeddings_init)

    model.set_output_type_embeddings(output_type_embeddings_init)

    classes_to_merge = None
    classes_to_merge = [
        type_names_to_int["N;PL"],
        type_names_to_int["N;PL+V;PRS;3;SG"],
        type_names_to_int["V;PRS;3;SG"],
    ]

    if len(negative_type_ids) > 0:
        model.freeze_rows_in_type_embeddings(negative_type_ids)
        model.set_negative_type_ids(negative_type_ids)

    model.set_token_types_filter(
        final_decomposition_map.keys(), [v[1] for v in final_decomposition_map.values()]
    )
    model.set_token_to_type_ids(final_decomposition_map, negative_type_ids)
    # Set decomposition map for conditioned transformation head (if enabled)
    model.set_seq_replacement_mapping(decomposed_to_original_seq)
    model.set_token_id_to_orig_first_id_mapping(decomposed_to_original_id)
    model.set_token_id_is_base_or_inflection_token(
        base_token_ids, all_inflections, all_inflections_token_count
    )
    model.set_ignored_logits_idx(existing_inflection_ids)

    untouched_indices = (
        set(base_tokenizer.get_vocab().values())
        - set(base_token_ids)
        - set(existing_inflection_ids)
    )
    untouched_indices = sorted(list(untouched_indices))
    model.set_logits_mapper(
        base_token_ids,
        final_decomposition_map,
        transformation_int_to_name,
        types_loss_indices_map,
        untouched_indices,
    )

    model.eval()

    # get evaluation data
    eval_dataset = load_lm_dataset(
        args.eval_dataset, language=args.dataset_language, num_examples=args.eval_max_samples
    )
    eval_dataset = eval_dataset[args.eval_dataset_split]

    efficiency_metrics = dict()
    baseline_vocab_total_tokens = count_tokens_in_dataset(
        eval_dataset, base_tokenizer, args.eval_dataset_text_col
    )
    new_vocab_total_tokens = count_tokens_in_dataset(
        eval_dataset, tokenizer, args.eval_dataset_text_col
    )
    logger.info(f"Baseline tokenizer - total tokens: {baseline_vocab_total_tokens}")
    logger.info(f"Inflection tokenizer - total tokens: {new_vocab_total_tokens}")
    efficiency_metrics["total_tokens"] = {
        "expanded": new_vocab_total_tokens,
        "baseline": baseline_vocab_total_tokens,
    }
    efficiency_metrics["tokens_saved"] = (
        1
        - efficiency_metrics["total_tokens"]["expanded"]
        / efficiency_metrics["total_tokens"]["baseline"]
    )

    eval_dataset = tokenize_and_prepare_dataset(
        eval_dataset,
        tokenizer,
        accelerator,
        max_length=args.eval_max_length,
        text_col_name=args.eval_dataset_text_col,
        pack_documents_with_eos=args.pack_documents_with_eos,
    )

    if args.eval_init:
        logger.info("Evaluate additive model and tokenizer...")

        if args.eval_downstream:
            if not args.dont_eval_init_end2end:
                downstream_log_dir = (
                    os.path.join(output_dir, "initialized")
                    if args.save_downstream_outputs
                    else None
                )
                downstream_metrics["initialized"] = curr_metrics = evaluate_model(
                    model,
                    tokenizer,
                    limited=True,
                    get_inflection_metrics=True,
                    output_path=downstream_log_dir,
                    examples_limit=args.downstream_limit,
                    use_per_run_cache=False,
                    per_run_cache_dir=output_dir,
                    per_run_stage_name="initialized",
                    overwrite_per_run_cache=args.overwrite_per_run_cache,
                    overwrite_cache=args.overwrite_eval_cache,
                )
                logger.info(f"Initialized downstream results:\n{pd.Series(curr_metrics)}")

                final_downstream_metrics = defaultdict(dict)
                for eval_type, eval_results in downstream_metrics.items():
                    for task_name, task_results in eval_results.items():
                        final_downstream_metrics[task_name][eval_type] = task_results
                with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
                    json.dump(final_downstream_metrics, f, indent=4)

                for metric_name, metric_value in curr_metrics.items():
                    logger.info(f"{metric_name}: {metric_value}")

            if args.eval_input_only:
                model.set_inference_with_input_types_only(True)
                downstream_log_dir = (
                    os.path.join(output_dir, "init_input_only_single_token")
                    if args.save_downstream_outputs
                    else None
                )
                downstream_metrics["init_input_only_single_token"] = curr_metrics = evaluate_model(
                    model,
                    single_token_tokenizer,
                    get_inflection_metrics=True,
                    limited=True,
                    examples_limit=args.downstream_limit,
                    output_path=downstream_log_dir,
                    use_per_run_cache=False,
                    per_run_cache_dir=output_dir,
                    per_run_stage_name="init_input_only_single_token",
                    overwrite_per_run_cache=args.overwrite_per_run_cache,
                    overwrite_cache=args.overwrite_eval_cache,
                )
                logger.info(
                    f"Initialized input only (single token) downstream results:\n{pd.Series(curr_metrics)}"
                )

                final_downstream_metrics = defaultdict(dict)
                for eval_type, eval_results in downstream_metrics.items():
                    for task_name, task_results in eval_results.items():
                        final_downstream_metrics[task_name][eval_type] = task_results
                with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
                    json.dump(final_downstream_metrics, f, indent=4)

                for metric_name, metric_value in curr_metrics.items():
                    logger.info(f"{metric_name}: {metric_value}")
                model.set_inference_with_input_types_only(False)

        if args.eval_ppl:
            metrics["initialized"] = curr_metrics = eval_next_word_prediction(
                model,
                tokenizer,
                eval_dataset,
                accelerator,
                type2name=transformation_int_to_name,
                base_token_ids=base_token_ids,
                inflection_token_ids=inflection_token_ids,
                inflection_replaced_sequences=inflection_replaced_sequences,
                max_length=args.eval_max_length,
                max_samples=args.eval_max_samples,
                eval_type_probes=args.probe_types,
                reduction="mean",
                use_lm_preds_for_type_labels=True,
                use_original_logits_as_target=False,
                map_labels_to_base_token=False,
            )
            logger.info(f"Initialized additive model results:")
            for metric_name, metric_value in curr_metrics.items():
                logger.info(f"{metric_name}: {metric_value}")

    if training_args.do_train:
        logger.info("Starting training pipeline...")
        logger.info(f"unfreeze_input_base_embeddings flag: {args.unfreeze_input_base_embeddings}")
        logger.info(f"unfreeze_output_base_embeddings flag: {args.unfreeze_output_base_embeddings}")
        logger.info(f"preinit_output_embeddings flag: {args.preinit_output_embeddings}")
        logger.info(f"post_train_output_embeddings flag: {args.post_train_output_embeddings}")
        logger.info(f"post_train_input_embeddings flag: {args.post_train_input_embeddings}")
        logger.info(f"different_data_per_stage flag: {args.different_data_per_stage}")
        logger.info(f"lora_ft flag: {args.lora_ft}")

        # get training data
        full_train_dataset = load_lm_dataset(args.train_dataset, language=args.dataset_language)

        # Determine how many stages will use data
        num_stages = 1  # Main second stage always runs
        num_stages += 1

        unprocessed_train_dataset = full_train_dataset["train"].select(
            range(2 * args.train_max_samples)
        )

        def get_stage_dataset(tokenizer_to_use, max_samples=None, stage_name=""):
            logger.info(f"Creating dataset for {stage_name} using shared data samples")
            tokenize_function = get_sft_tokenize_func(tokenizer_to_use) if args.sft else None
            return tokenize_and_prepare_dataset(
                unprocessed_train_dataset,
                tokenizer_to_use,
                accelerator,
                max_length=args.eval_max_length,
                text_col_name=args.eval_dataset_text_col,
                max_samples=max_samples if max_samples is not None else args.train_max_samples,
                add_start_token_after_grouping=True,
                pack_documents_with_eos=args.pack_documents_with_eos or args.sft,
                tokenize_function=tokenize_function,
            )

        # Create dataset for main training stage
        train_dataset = get_stage_dataset(tokenizer, stage_name="main_training")
        # freeze everything
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze types head model
        unfreeze_modules = [
            model.types_lm_head,
            # model.types_lm_proj_mat,
            # model.types_output_diff_norm,
        ]
        for module in unfreeze_modules:
            for param in module.parameters():
                param.requires_grad = True

        logger.info("Unfreezing input type embeddings!")
        # unfreeze types embedding
        for param in model.embed_types.parameters():
            param.requires_grad = True

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        # Training
        checkpoint = None

        # pre-init stage: train input embeddings
        model.set_lora_trainable(False)

        # Set true distillation mode if requested (teacher always uses original embeddings)

        orig_lr = training_args.learning_rate
        if args.second_learning_rate is not None:
            training_args.learning_rate = args.second_learning_rate
        model.set_train_input_types_only(True)
        model.set_distill_to_model_copy(True)
        single_token_train_dataset = get_stage_dataset(
            single_token_tokenizer,
            max_samples=args.train_max_samples_input_distill,
            stage_name="input_distillation",
        )
        # Create custom optimizer for first stage if using different LR factors
        first_stage_optimizer = None
        if args.input_lr_factor != 1.0 or args.output_lr_factor != 1.0:
            logger.info(
                f"Creating custom optimizer for 1st stage with input_lr_factor={args.input_lr_factor}"
            )
            first_stage_optimizer = create_optimizer_with_custom_lr(
                model=model,
                base_lr=training_args.learning_rate,
                input_lr_factor=args.input_lr_factor,
                output_lr_factor=args.output_lr_factor,
                weight_decay=training_args.weight_decay,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
            )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=single_token_train_dataset,
            eval_dataset=None,
            tokenizer=single_token_tokenizer,
            data_collator=default_data_collator,
            optimizers=(first_stage_optimizer, None)
            if first_stage_optimizer is not None
            else (None, None),
        )
        model.clm_loss_alpha = 1.0
        train_result = trainer.train()
        model.set_train_input_types_only(False)
        model.set_distill_to_model_copy(False)
        training_args.learning_rate = orig_lr

        # Always freeze input type embeddings after input training stage
        # (will be unfrozen later in 2nd stage if keep_input_embeddings_unfrozen is set)
        logger.info("Freezing input type embeddings after input training stage")
        for param in model.embed_types.parameters():
            param.requires_grad = False

        # Freeze input base embeddings if they were unfrozen

        if args.eval_ft:
            logger.info("Evaluate additive model and tokenizer")
            model.eval()

            if args.eval_downstream:
                if args.eval_input_only:
                    model.set_inference_with_input_types_only(True)
                    downstream_log_dir = (
                        os.path.join(output_dir, "ft_input_init_input_only_single_token")
                        if args.save_downstream_outputs
                        else None
                    )
                    downstream_metrics["ft_input_init_input_only_single_token"] = curr_metrics = (
                        evaluate_model(
                            model,
                            single_token_tokenizer,
                            get_inflection_metrics=True,
                            limited=True,
                            examples_limit=args.downstream_limit,
                            output_path=downstream_log_dir,
                            use_per_run_cache=False,
                            per_run_cache_dir=output_dir,
                            per_run_stage_name="ft_input_init_input_only_single_token",
                            overwrite_per_run_cache=args.overwrite_per_run_cache,
                            overwrite_cache=args.overwrite_eval_cache,
                        )
                    )
                    logger.info(
                        f"Finetuned input_init input only (single token) downstream results:\n{pd.Series(curr_metrics)}"
                    )

                    final_downstream_metrics = defaultdict(dict)
                    for eval_type, eval_results in downstream_metrics.items():
                        for task_name, task_results in eval_results.items():
                            final_downstream_metrics[task_name][eval_type] = task_results
                    with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
                        json.dump(final_downstream_metrics, f, indent=4)

                    for metric_name, metric_value in curr_metrics.items():
                        logger.info(f"{metric_name}: {metric_value}")
                    model.set_inference_with_input_types_only(False)

            if args.eval_ppl:
                metrics["ft_input_init_end_to_end"] = curr_metrics = eval_next_word_prediction(
                    model,
                    tokenizer,
                    eval_dataset,
                    accelerator,
                    type2name=transformation_int_to_name,
                    base_token_ids=base_token_ids,
                    inflection_token_ids=inflection_token_ids,
                    inflection_replaced_sequences=inflection_replaced_sequences,
                    max_length=args.eval_max_length,
                    max_samples=args.eval_max_samples,
                    eval_type_probes=args.probe_types,
                    reduction="mean",
                    use_lm_preds_for_type_labels=True,
                    use_original_logits_as_target=False,
                    map_labels_to_base_token=False,
                )

                logger.info(f"Finetuned input-init end-to-end model results:")
                for metric_name, metric_value in curr_metrics.items():
                    logger.info(f"{metric_name}: {metric_value}")

            if args.eval_downstream:
                if not args.dont_eval_init_end2end:
                    downstream_log_dir = (
                        os.path.join(output_dir, "ft_input_init_end_to_end")
                        if args.save_downstream_outputs
                        else None
                    )
                    downstream_metrics["ft_input_init_end_to_end"] = curr_metrics = evaluate_model(
                        model,
                        tokenizer,
                        get_inflection_metrics=True,
                        limited=True,
                        examples_limit=args.downstream_limit,
                        output_path=downstream_log_dir,
                        use_per_run_cache=False,
                        per_run_cache_dir=output_dir,
                        per_run_stage_name="ft_input_init_end_to_end",
                        overwrite_per_run_cache=args.overwrite_per_run_cache,
                        overwrite_cache=args.overwrite_eval_cache,
                    )
                    logger.info(
                        f"Finetuned input_init end-to-end downstream results:\n{pd.Series(curr_metrics)}"
                    )

                    final_downstream_metrics = defaultdict(dict)
                    for eval_type, eval_results in downstream_metrics.items():
                        for task_name, task_results in eval_results.items():
                            final_downstream_metrics[task_name][eval_type] = task_results
                    with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
                        json.dump(final_downstream_metrics, f, indent=4)

                    for metric_name, metric_value in curr_metrics.items():
                        logger.info(f"{metric_name}: {metric_value}")

            model.train()

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        # pre-init stage: train output type embeddings (before LoRA)

        if args.lora_ft:
            model.apply_lora_to_last_layer(
                target_modules=args.lora_target_modules,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )

        for param in model.types_lm_head.parameters():
            param.requires_grad = True

        # 2nd stage:
        # Modify learning rate
        logger.info("=== Starting 2nd stage training ===")
        logger.info(
            f"Checking unfreeze_output_base_embeddings flag: {args.unfreeze_output_base_embeddings}"
        )
        model.set_lora_trainable(True)

        # Unfreeze input type embeddings if requested for output stage

        logger.info("NOT unfreezing output base embeddings (flag is False)")
        if args.second_learning_rate is not None:
            training_args.learning_rate = args.second_learning_rate

        # Log trainable parameters before creating trainer
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Second stage trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of all params)"
        )

        # Create custom optimizer if using different LR factors
        optimizer = None
        if args.input_lr_factor != 1.0 or args.output_lr_factor != 1.0:
            logger.info(
                f"Creating custom optimizer with input_lr_factor={args.input_lr_factor}, output_lr_factor={args.output_lr_factor}"
            )
            optimizer = create_optimizer_with_custom_lr(
                model=model,
                base_lr=training_args.learning_rate,
                input_lr_factor=args.input_lr_factor,
                output_lr_factor=args.output_lr_factor,
                weight_decay=training_args.weight_decay,
                adam_beta1=training_args.adam_beta1,
                adam_beta2=training_args.adam_beta2,
                adam_epsilon=training_args.adam_epsilon,
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=default_data_collator,
            # Pass custom optimizer; Trainer will create scheduler automatically (maintaining relative LR ratios)
            optimizers=(optimizer, None) if optimizer is not None else (None, None),
        )
        model.set_distill_to_self(True)
        model.clm_loss_alpha = 1.0

        train_result = trainer.train()
        trainer_metrics = train_result.metrics
        train_max_samples = (
            args.train_max_samples if args.train_max_samples is not None else len(train_dataset)
        )
        trainer_metrics["train_samples"] = min(train_max_samples, len(train_dataset))
        trainer.log_metrics("train", trainer_metrics)

        # post-train output embeddings (base LM head + type LM head)

        # post-train input embeddings

    model.eval()
    model = accelerator.prepare(model)

    if args.eval_ft:
        logger.info("Evaluate additive model and tokenizer")

        if args.eval_ppl:
            metrics["finetuned"] = curr_metrics = eval_next_word_prediction(
                model,
                tokenizer,
                eval_dataset,
                accelerator,
                type2name=transformation_int_to_name,
                base_token_ids=base_token_ids,
                inflection_token_ids=inflection_token_ids,
                inflection_replaced_sequences=inflection_replaced_sequences,
                max_length=args.eval_max_length,
                max_samples=args.eval_max_samples,
                eval_type_probes=args.probe_types,
                reduction="mean",
                use_lm_preds_for_type_labels=True,
                use_original_logits_as_target=False,
                map_labels_to_base_token=False,
            )
            logger.info(f"Finetuned model results:")
            for metric_name, metric_value in curr_metrics.items():
                logger.info(f"{metric_name}: {metric_value}")

        if args.eval_downstream:
            downstream_log_dir = (
                os.path.join(output_dir, "finetuned") if args.save_downstream_outputs else None
            )
            downstream_metrics["finetuned"] = curr_metrics = evaluate_model(
                model,
                tokenizer,
                limited=True,
                get_inflection_metrics=True,
                output_path=downstream_log_dir,
                examples_limit=args.downstream_limit,
                use_per_run_cache=False,
                per_run_cache_dir=output_dir,
                per_run_stage_name="finetuned",
                overwrite_per_run_cache=args.overwrite_per_run_cache,
                overwrite_cache=args.overwrite_eval_cache,
            )
            logger.info(f"Finetuned downstream results:\n{pd.Series(curr_metrics)}")

            final_downstream_metrics = defaultdict(dict)
            for eval_type, eval_results in downstream_metrics.items():
                for task_name, task_results in eval_results.items():
                    final_downstream_metrics[task_name][eval_type] = task_results
            with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
                json.dump(final_downstream_metrics, f, indent=4)

            logger.info(f"Finetuned model downstream results:")
            for metric_name, metric_value in curr_metrics.items():
                logger.info(f"{metric_name}: {metric_value}")

    final_downstream_metrics = defaultdict(dict)
    for eval_type, eval_results in downstream_metrics.items():
        for task_name, task_results in eval_results.items():
            final_downstream_metrics[task_name][eval_type] = task_results
    with open(os.path.join(output_dir, f"efficiency_metrics.json"), "w") as f:
        json.dump(efficiency_metrics, f, indent=4)
    with open(os.path.join(output_dir, f"downstream_metrics.json"), "w") as f:
        json.dump(final_downstream_metrics, f, indent=4)

    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"))
    metrics = serialize_metrics(metrics)

    with open(os.path.join(output_dir, f"metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    config = vars(args)
    with open(os.path.join(output_dir, f"config.json"), "w") as config_file:
        json.dump(config, config_file, indent=4)

    if accelerator.is_main_process:
        save_adaptation(
            accelerator.unwrap_model(model),
            tokenizer,
            base_tokenizer,
            os.path.join(output_dir, "checkpoint"),
        )
        with open(os.path.join(output_dir, "removed_vocab_inflection_tokens.json"), "w") as f:
            json.dump(sorted(existing_inflection_ids), f, indent=2)
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=2)
    logger.info(f"Results saved to: {output_dir}")

    return


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Adapt an LLM with additive vocabulary representations."
    )
    parser.add_argument(
        "--language",
        choices=["english", "spanish", "german", "russian", "arabic"],
        default="english",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--unimorph_root",
        type=str,
        default=None,
        help="UniMorph directory (defaults to $UNIMORPH_ROOT or resources/unimorph).",
    )
    parser.add_argument("--output_dir", type=str, default="outputs/adaptation")
    parser.add_argument("--space_prefix", type=str, default=" ")

    parser.add_argument("--skip_if_no_space_prefix", action="store_true", default=False)
    parser.add_argument("--dont_decompose_spaces", action="store_true", default=False)
    parser.add_argument("--dont_decompose_capitalized", action="store_true", default=False)
    parser.add_argument(
        "--dont_decompose_proper_nouns",
        action="store_true",
        default=False,
        help="Exclude proper nouns/named entities from capitalization decomposition using NER",
    )
    parser.add_argument(
        "--dont_use_capitalized_base_words",
        action="store_true",
        default=False,
        help="Exclude capitalized words as base forms (e.g., 'Johns' won't be an inflection of 'John')",
    )
    parser.set_defaults(predict_negative_types=False)
    parser.set_defaults(use_efficient_tied_embeddings=False)
    parser.add_argument(
        "--ner_cache_dir",
        type=str,
        default="./cache",
        help="Directory to save/load NER cache for proper noun detection",
    )
    parser.add_argument(
        "--regenerate_ner_cache",
        action="store_true",
        default=False,
        help="Force regeneration of NER cache (use after changing NER logic)",
    )

    parser.set_defaults(run_generate_tests=False)
    parser.add_argument("--save_downstream_outputs", action="store_true", default=True)

    parser.add_argument(
        "--remove_patchscopes_mistakes", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--patchscopes_save_path", type=str, default="./cache/patchscopes")
    parser.add_argument("--patchscopes_prompt", type=str, default="{word}, {word}, {word}, {word},")

    parser.set_defaults(dump_outliers=False)

    parser.set_defaults(random_init=False)
    parser.set_defaults(unfreeze_input_embeddings=True)
    parser.set_defaults(keep_input_embeddings_unfrozen=False)
    parser.set_defaults(unfreeze_input_base_embeddings=False)
    parser.set_defaults(preinit_output_embeddings=False)
    parser.set_defaults(unfreeze_output_base_embeddings=False)
    parser.set_defaults(distill_for_input_embeddings=True)
    parser.set_defaults(always_distill_to_original=False)
    parser.set_defaults(distill_hidden_states=False)
    parser.set_defaults(distill_hidden_layer=7)
    parser.set_defaults(distill_hidden_alpha=1.0)
    parser.set_defaults(clm_for_output_embeddings=False)
    parser.set_defaults(post_train_input_embeddings=False)
    parser.set_defaults(post_train_output_embeddings=False)
    parser.set_defaults(post_train_both=False)
    parser.add_argument("--lora_ft", action=argparse.BooleanOptionalAction, default=True)
    parser.set_defaults(lora_adapter_toggling=True)
    parser.add_argument("--lora_r", type=int, default=256, help="LoRA rank (default: 256)")
    parser.add_argument(
        "--lora_alpha", type=int, default=256, help="LoRA alpha parameter (default: 256)"
    )
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout parameter")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["q", "k", "v", "o", "mlp"],
        help="LoRA target modules: q, k, v, o for attention projections, mlp for all MLP layers (default: q k v o mlp)",
    )
    parser.set_defaults(profile_training=False)
    parser.set_defaults(different_data_per_stage=False)
    parser.set_defaults(tie_word_embeddings=False)

    parser.set_defaults(num_transformation_clusters=1)

    parser.set_defaults(modeling_option=None)
    parser.set_defaults(use_lm_preds_for_type_labels=False)
    parser.set_defaults(types_output_zero_init=False)
    parser.set_defaults(probe_types=False)
    parser.set_defaults(merge_plural_classes=True)
    parser.set_defaults(types_num_probe_layers=1)
    parser.set_defaults(types_loss_alpha=0.0)
    parser.set_defaults(base_loss_alpha=0.0)
    parser.add_argument("--k_last_layers", type=int, default=8)
    parser.set_defaults(types_prediction_layer=-1)
    parser.set_defaults(base_prediction_layer=-1)
    parser.set_defaults(rescale_embeds=False)

    parser.add_argument("--skip_multi_token_words", action="store_true", default=False)
    parser.add_argument("--skip_three_token_words", action="store_true", default=False)

    parser.add_argument("--include_derivations", action="store_true", default=False)
    parser.set_defaults(use_type_str_as_type=False)
    parser.set_defaults(allow_multitypes_in_init=False)

    parser.add_argument("--dataset_language", type=str, default=None)
    parser.add_argument("--train_dataset", type=str, default="fineweb-edu")
    parser.add_argument("--train_max_samples", type=int, default=20000)
    parser.add_argument("--train_max_samples_input_distill", type=int, default=20000)
    parser.set_defaults(sft=False)
    parser.add_argument("--eval_ppl", action="store_true", default=False)
    parser.add_argument("--eval_input_only", action="store_true", default=False)
    parser.add_argument("--dont_eval_init_end2end", action="store_true", default=False)
    parser.add_argument("--eval_downstream", action="store_true", default=False)
    parser.add_argument("--eval_baselines_downstream", action="store_true", default=False)
    parser.add_argument("--eval_baselines", action="store_true", default=False)
    parser.add_argument("--eval_init", action="store_true", default=False)
    parser.add_argument("--eval_ft", action="store_true", default=False)
    parser.add_argument("--eval_dataset", type=str, default="wikitext")
    parser.add_argument("--eval_max_samples", type=int, default=None)
    parser.add_argument("--eval_shuffle_samples", action="store_true", default=True)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_max_length", type=int, default=256)
    parser.add_argument("--downstream_limit", type=int, default=5000)
    parser.add_argument("--eval_dataset_split", type=str, default="test")
    parser.add_argument("--eval_dataset_text_col", type=str, default="text")
    parser.add_argument(
        "--pack_documents_with_eos",
        action="store_true",
        default=False,
        help="Pack full documents with EOS separators before chunking sequences.",
    )
    parser.add_argument("--overwrite_eval_cache", action="store_true", default=False)
    parser.add_argument(
        "--overwrite_per_run_cache",
        action="store_true",
        default=False,
        help="Overwrite per-run evaluation cache if it exists",
    )

    args, remaining = parser.parse_known_args(argv)
    hf_parser = HfArgumentParser(TrainingArguments)
    hf_parser.set_defaults(
        num_train_epochs=1.0,
        warmup_ratio=0.03,
        weight_decay=0.0,
        save_strategy="no",
        report_to="none",
    )
    training_args, unknown = hf_parser.parse_args_into_dataclasses(
        args=remaining, return_remaining_strings=True
    )
    if unknown:
        parser.error("unrecognized arguments: " + " ".join(unknown))
    training_args.output_dir = args.output_dir
    training_args.seed = args.seed
    return args, training_args


if __name__ == "__main__":
    args, training_args = parse_args()
    main(args, training_args)
