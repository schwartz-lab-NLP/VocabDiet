"""Portable inference checkpoints for additive models and scaffold tokenizers."""

from pathlib import Path
import json

import torch
from transformers import AutoConfig, PreTrainedTokenizerFast

from .modeling import AdditiveLlamaForCausalLM, AdditiveQwen2ForCausalLM, AdditiveOlmo2ForCausalLM
from .tokenization import ScaffoldTokenizer

FORMAT_VERSION = 1
MAPPING_ATTRIBUTES = (
    "token_id_to_base_id_mapping",
    "token_id_to_orig_first_id_mapping",
    "token_id_is_base_or_inflection_mapping",
    "token_id_is_inflection_mapping",
    "token_id_to_inflection_token_count",
    "types_negative_ids",
)


def model_class_for_config(config):
    classes = {
        "llama": AdditiveLlamaForCausalLM,
        "qwen2": AdditiveQwen2ForCausalLM,
        "olmo2": AdditiveOlmo2ForCausalLM,
    }
    try:
        return classes[config.model_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported model architecture: {config.model_type}; choose llama, qwen2, or olmo2."
        ) from exc


def save_adaptation(model, tokenizer, base_tokenizer, path):
    """Export an inference checkpoint, merging any trained LoRA adapters in place.

    Call after training/evaluation. The exported model includes frozen weights,
    learned offsets, vocabulary mappings, and both tokenizer definitions.
    Training optimizer state is not part of this inference format.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if model.lora_applied:
        model.model = model.model.merge_and_unload()
        model.lora_applied = False
    model.lora_ft = False
    model.lora_trained = False
    model.last_layer_copy = None
    model.set_train_input_types_only(False)
    model.set_distill_to_model_copy(False)
    metadata = {
        "format_version": FORMAT_VERSION,
        "constructor": {
            "types_vocab_size": model.types_vocab_size,
            "model_vocab_size": model.model_vocab_size,
            "extended_vocab_size": model.extended_vocab_size,
            "types_loss_indices": model.types_loss_indices,
            "lora_adapter_toggling": True,
            "clm_loss_alpha": 0.0,
            "types_loss_alpha": 0.0,
            "base_loss_alpha": 0.0,
        },
        "ignored_logits_idx": model.ignored_logits_idx.tolist()
        if model.ignored_logits_idx is not None
        else None,
        "logits_mapper": model._logits_mapper_recipe,
        "mappings": {
            name: getattr(model, name).tolist() if getattr(model, name) is not None else None
            for name in MAPPING_ATTRIBUTES
        },
        "scaffold_vocab": tokenizer.scaffold_vocab,
        "dtype": str(model.dtype).removeprefix("torch."),
    }
    # Probe modules are auxiliary diagnostics; inference uses types_lm_head.
    weights = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if not k.startswith("types_prediction_head.")
    }
    model.config.save_pretrained(path)
    torch.save(weights, path / "weights.pt")
    (path / "adaptation.json").write_text(json.dumps(metadata, indent=2))
    PreTrainedTokenizerFast.save_pretrained(tokenizer, path / "tokenizer")
    base_tokenizer.save_pretrained(path / "base_tokenizer")


def load_adaptation(path, *, device="cpu", dtype=None):
    """Reload a local checkpoint without downloading the original model."""
    path = Path(path)
    metadata = json.loads((path / "adaptation.json").read_text())
    if metadata["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported adaptation checkpoint format")
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    model = model_class_for_config(config)(config, **metadata["constructor"])
    model.to(dtype=dtype or getattr(torch, metadata["dtype"]))
    if metadata["ignored_logits_idx"] is not None:
        model.set_ignored_logits_idx(metadata["ignored_logits_idx"])
    recipe = metadata["logits_mapper"]
    model.set_logits_mapper(
        recipe["base_token_indices"],
        {int(k): (v[0], v[1]) for k, v in recipe["decomposition_map"].items()},
        {int(k): v for k, v in recipe["transformation_int_to_name"].items()},
        recipe["types_loss_indices_map"],
        recipe["untouched_indices"],
    )
    weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not name.startswith("types_prediction_head.") for name in missing):
        raise ValueError(
            f"Checkpoint weights do not match the model: missing={missing}, unexpected={unexpected}"
        )
    for name, values in metadata["mappings"].items():
        if values is not None:
            tensor_dtype = (
                torch.long
                if name
                in {
                    "token_id_to_base_id_mapping",
                    "token_id_to_orig_first_id_mapping",
                    "types_negative_ids",
                }
                else model.dtype
            )
            setattr(model, name, torch.tensor(values, dtype=tensor_dtype))
    base = PreTrainedTokenizerFast.from_pretrained(path / "base_tokenizer", local_files_only=True)
    extended = PreTrainedTokenizerFast.from_pretrained(path / "tokenizer", local_files_only=True)
    tokenizer = ScaffoldTokenizer(extended, base, metadata["scaffold_vocab"])
    model.tokenizer = tokenizer
    model.to(device).eval()
    return model, tokenizer
