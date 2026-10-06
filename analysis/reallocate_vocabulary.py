"""
Freed-slot reallocation tokenization-efficiency analysis.

Workflow:
1) Load post-hoc removed token IDs from the English tokenizer.
2) Train a language-specific BPE tokenizer on in-domain text.
3) Select N distinct new tokens for each language with usable BPE dependencies.
4) Reallocate freed English token IDs in one shared multilingual tokenizer.
5) Evaluate that same tokenizer on each language's held-out in-domain text:
   - bytes per token = total_utf8_bytes / total_tokens
   - token fertility = total_tokens / total_whitespace_words

Example:
python -m analysis.reallocate_vocabulary \
  --base_tokenizer meta-llama/Llama-3.1-8B \
  --removed_tokens_json outputs/adaptation/removed_vocab_inflection_tokens.json \
  --languages ar ru de es \
  --tokens_per_language 2500 \
  --output_root outputs/reallocation
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


DEFAULT_LANGUAGE_SPECS = {
    "ar": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "dataset_config": "arb_Arab",
        "split": "train",
        "text_field": "text",
    },
    "ru": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "dataset_config": "rus_Cyrl",
        "split": "train",
        "text_field": "text",
    },
    "de": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "dataset_config": "deu_Latn",
        "split": "train",
        "text_field": "text",
    },
    "es": {
        "dataset": "HuggingFaceFW/fineweb-2",
        "dataset_config": "spa_Latn",
        "split": "train",
        "text_field": "text",
    },
}


def parse_count(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    if text.endswith("B"):
        return int(float(text[:-1]) * 1e9)
    if text.endswith("M"):
        return int(float(text[:-1]) * 1e6)
    if text.endswith("K"):
        return int(float(text[:-1]) * 1e3)
    return int(text)


def relative_pct(delta: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return (float(delta) / float(baseline)) * 100.0


def unique_tokens(items: List[Optional[str]]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def get_special_tokens(base_tokenizer, fallback_eos: str, fallback_unk: str) -> List[str]:
    if base_tokenizer is None:
        return [fallback_eos, fallback_unk]
    return unique_tokens(
        [
            base_tokenizer.eos_token or fallback_eos,
            base_tokenizer.unk_token or fallback_unk,
            base_tokenizer.pad_token,
            base_tokenizer.bos_token,
            base_tokenizer.sep_token,
            base_tokenizer.cls_token,
            base_tokenizer.mask_token,
        ]
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_merge_pair(merge_entry):
    if isinstance(merge_entry, str):
        parts = merge_entry.split(" ")
        if len(parts) == 2:
            return parts[0], parts[1]
        return None
    if isinstance(merge_entry, (list, tuple)) and len(merge_entry) == 2:
        left, right = merge_entry
        if isinstance(left, str) and isinstance(right, str):
            return left, right
    return None


def format_merge_entry(left: str, right: str, use_list_format: bool):
    if use_list_format:
        return [left, right]
    return f"{left} {right}"


def _is_limit_reached(
    num_examples: int, num_bytes: int, max_examples: Optional[int], max_bytes: Optional[int]
) -> bool:
    if max_examples is not None and num_examples >= max_examples:
        return True
    if max_bytes is not None and num_bytes >= max_bytes:
        return True
    return False


@dataclass
class LanguageSpec:
    dataset: str
    dataset_config: Optional[str]
    split: str
    text_field: str


@dataclass
class TokenizationMetrics:
    total_bytes: int
    total_words: int
    total_tokens: int
    bytes_per_token: float
    token_fertility: float


def collect_train_eval_texts(
    spec: LanguageSpec,
    streaming: bool,
    cache_dir: Optional[str],
    min_text_chars: int,
    train_max_examples: Optional[int],
    train_max_bytes: Optional[int],
    eval_max_examples: Optional[int],
    eval_max_bytes: Optional[int],
) -> Tuple[List[str], List[str], Dict[str, int]]:
    load_kwargs = {
        "path": spec.dataset,
        "split": spec.split,
        "streaming": streaming,
    }
    if spec.dataset_config is not None:
        load_kwargs["name"] = spec.dataset_config
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir
    dataset = load_dataset(**load_kwargs)

    train_texts: List[str] = []
    eval_texts: List[str] = []

    train_examples = 0
    train_bytes = 0
    eval_examples = 0
    eval_bytes = 0

    in_eval_phase = False
    for sample in dataset:
        text = sample.get(spec.text_field)
        if not isinstance(text, str):
            continue
        if len(text) < int(min_text_chars):
            continue
        text_bytes = len(text.encode("utf-8"))
        if text_bytes <= 0:
            continue

        if not in_eval_phase and _is_limit_reached(
            train_examples, train_bytes, train_max_examples, train_max_bytes
        ):
            in_eval_phase = True

        if not in_eval_phase:
            train_texts.append(text)
            train_examples += 1
            train_bytes += text_bytes
            if _is_limit_reached(train_examples, train_bytes, train_max_examples, train_max_bytes):
                in_eval_phase = True
            continue

        if _is_limit_reached(eval_examples, eval_bytes, eval_max_examples, eval_max_bytes):
            break

        eval_texts.append(text)
        eval_examples += 1
        eval_bytes += text_bytes

        if _is_limit_reached(eval_examples, eval_bytes, eval_max_examples, eval_max_bytes):
            break

    stats = {
        "train_examples": train_examples,
        "train_bytes": train_bytes,
        "eval_examples": eval_examples,
        "eval_bytes": eval_bytes,
    }
    return train_texts, eval_texts, stats


def build_tokenizer(base_tokenizer, unk_token: str) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    if base_tokenizer is None:
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        return tokenizer

    backend = base_tokenizer.backend_tokenizer
    if backend.normalizer is not None:
        tokenizer.normalizer = backend.normalizer
    if backend.pre_tokenizer is not None:
        tokenizer.pre_tokenizer = backend.pre_tokenizer
    if backend.post_processor is not None:
        tokenizer.post_processor = backend.post_processor
    if backend.decoder is not None:
        tokenizer.decoder = backend.decoder
    return tokenizer


def _language_bpe_cache_key(
    language: str,
    spec: LanguageSpec,
    base_tokenizer_name: str,
    vocab_size: int,
    min_frequency: int,
    train_examples: Optional[int],
    train_bytes: Optional[int],
) -> str:
    payload = {
        "language": language,
        "dataset": spec.dataset,
        "dataset_config": spec.dataset_config,
        "split": spec.split,
        "text_field": spec.text_field,
        "base_tokenizer": base_tokenizer_name,
        "vocab_size": int(vocab_size),
        "min_frequency": int(min_frequency),
        "train_examples": train_examples,
        "train_bytes": train_bytes,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def train_language_bpe(
    language: str,
    train_texts: Sequence[str],
    spec: LanguageSpec,
    base_tokenizer,
    vocab_size: int,
    min_frequency: int,
    train_max_examples: Optional[int],
    train_max_bytes: Optional[int],
    cache_dir: Optional[str],
    overwrite_cache: bool,
) -> Tuple[PreTrainedTokenizerFast, Dict[str, object], Optional[str]]:
    fallback_eos = "<|endoftext|>"
    fallback_unk = "<|unk|>"
    special_tokens = get_special_tokens(base_tokenizer, fallback_eos, fallback_unk)
    unk_token = (
        base_tokenizer.unk_token if base_tokenizer and base_tokenizer.unk_token else fallback_unk
    )

    cache_path = None
    if cache_dir:
        ensure_dir(cache_dir)
        cache_key = _language_bpe_cache_key(
            language=language,
            spec=spec,
            base_tokenizer_name=str(getattr(base_tokenizer, "name_or_path", "base")),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            train_examples=train_max_examples,
            train_bytes=train_max_bytes,
        )
        cache_path = os.path.join(cache_dir, f"{language}__{cache_key}")
        if os.path.isdir(cache_path) and (not overwrite_cache):
            tok = AutoTokenizer.from_pretrained(cache_path, use_fast=True)
            metadata = {
                "cache_path": cache_path,
                "loaded_from_cache": True,
            }
            return tok, metadata, cache_path

    tokenizer = build_tokenizer(base_tokenizer, unk_token=unk_token)
    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=special_tokens,
        show_progress=True,
    )
    tokenizer.train_from_iterator(train_texts, trainer=trainer, length=len(train_texts))

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=unk_token,
    )
    if base_tokenizer is not None:
        if base_tokenizer.eos_token:
            fast.eos_token = base_tokenizer.eos_token
        if base_tokenizer.bos_token:
            fast.bos_token = base_tokenizer.bos_token
        if base_tokenizer.pad_token:
            fast.pad_token = base_tokenizer.pad_token
        if base_tokenizer.sep_token:
            fast.sep_token = base_tokenizer.sep_token
        if base_tokenizer.cls_token:
            fast.cls_token = base_tokenizer.cls_token
        if base_tokenizer.mask_token:
            fast.mask_token = base_tokenizer.mask_token
        if hasattr(base_tokenizer, "add_prefix_space"):
            fast.add_prefix_space = base_tokenizer.add_prefix_space

    metadata = {
        "cache_path": cache_path,
        "loaded_from_cache": False,
        "vocab_size": len(fast.get_vocab()),
    }
    if cache_path:
        fast.save_pretrained(cache_path)
        with open(os.path.join(cache_path, "training_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "language": language,
                    "dataset": spec.dataset,
                    "dataset_config": spec.dataset_config,
                    "split": spec.split,
                    "text_field": spec.text_field,
                    "train_examples_limit": train_max_examples,
                    "train_bytes_limit": train_max_bytes,
                    "actual_train_examples": len(train_texts),
                    "vocab_size": len(fast.get_vocab()),
                    "min_frequency": min_frequency,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return fast, metadata, cache_path


def load_removed_token_ids(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    removed: List[int] = []
    if isinstance(payload, list):
        for item in payload:
            token_id = None
            if isinstance(item, dict) and "token_id" in item:
                token_id = item["token_id"]
            elif isinstance(item, int):
                token_id = item
            elif isinstance(item, str) and item.strip().isdigit():
                token_id = int(item.strip())
            if token_id is None:
                continue
            removed.append(int(token_id))
    else:
        raise ValueError(f"Unsupported removed-token file format in {path}; expected a JSON list.")

    seen = set()
    deduped = []
    for token_id in removed:
        if token_id in seen:
            continue
        seen.add(token_id)
        deduped.append(token_id)
    return deduped


def collect_special_token_ids(tokenizer) -> List[int]:
    ids = set()
    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    for attr in (
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "unk_token_id",
        "sep_token_id",
        "cls_token_id",
        "mask_token_id",
    ):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            ids.add(value)
    return sorted(ids)


def rank_new_language_tokens(language_tokenizer, base_token_strings: Sequence[str]) -> List[str]:
    base_set = set(base_token_strings)
    special_set = set(getattr(language_tokenizer, "all_special_tokens", []) or [])
    vocab = language_tokenizer.get_vocab()
    ranked = sorted(vocab.items(), key=lambda kv: int(kv[1]))
    new_tokens: List[str] = []
    for token, _ in ranked:
        if token in special_set:
            continue
        if token in base_set:
            continue
        new_tokens.append(token)
    return new_tokens


def _extract_bpe_model(tokenizer_json: Dict[str, object]) -> Tuple[Dict[str, int], List[object]]:
    model = tokenizer_json.get("model", {})
    model_type = str(model.get("type", "")).upper()
    if model_type != "BPE":
        raise ValueError(
            f"Tokenizer must be BPE for this analysis. Got model.type={model.get('type')}"
        )
    vocab = model.get("vocab", {})
    merges = model.get("merges", [])
    if not isinstance(vocab, dict):
        raise ValueError("Invalid tokenizer model.vocab format.")
    if not isinstance(merges, list):
        raise ValueError("Invalid tokenizer model.merges format.")
    vocab_int = {}
    for token, token_id in vocab.items():
        try:
            vocab_int[token] = int(token_id)
        except Exception as exc:
            raise ValueError(f"Non-integer token id for token={token!r}: {token_id}") from exc
    return vocab_int, merges


def select_joint_replacements(base_json, language_jsons, removed_token_ids, slots_per_language):
    """Select unique BPE tokens in rank order, preserving merge reachability.

    Languages consume disjoint slots in the supplied order. Existing tokens and
    tokens selected for an earlier language may serve as merge dependencies but
    do not count toward another language's allocation.
    """
    if slots_per_language <= 0 or not language_jsons:
        raise ValueError("A positive slot budget and at least one language are required.")
    if len(removed_token_ids) != slots_per_language * len(language_jsons):
        raise ValueError("The freed-slot count must equal the combined language budgets.")
    base_vocab, base_merges = _extract_bpe_model(base_json)
    kept = {t for t, i in base_vocab.items() if i not in set(removed_token_ids)}
    reachable = {t for t in kept if len(t) == 1}
    reachable.update(
        t["content"] for t in base_json.get("added_tokens", []) if t["content"] in kept
    )
    for entry in base_merges:
        pair = parse_merge_pair(entry)
        if pair and pair[0] in reachable and pair[1] in reachable and pair[0] + pair[1] in kept:
            reachable.add(pair[0] + pair[1])
    selected, selected_pairs, allocations = [], [], {}
    for language, language_json in language_jsons.items():
        vocab, merges = _extract_bpe_model(language_json)
        producing_pair = {}
        for entry in merges:
            pair = parse_merge_pair(entry)
            if pair:
                producing_pair.setdefault(pair[0] + pair[1], pair)
        allocation = []
        for token in sorted(vocab, key=vocab.get):
            pair = producing_pair.get(token)
            if pair is None or pair[0] not in reachable or pair[1] not in reachable:
                continue
            if token in kept:
                if token not in reachable:
                    selected_pairs.append(pair)
                    reachable.add(token)
                continue
            if token in base_vocab:
                continue
            selected.append(token)
            allocation.append(token)
            selected_pairs.append(pair)
            kept.add(token)
            reachable.add(token)
            if len(allocation) == slots_per_language:
                break
        if len(allocation) != slots_per_language:
            raise ValueError(
                f"Language {language} supplies only {len(allocation)} usable distinct new tokens; "
                f"need {slots_per_language}. Increase the BPE vocabulary or training sample."
            )
        allocations[language] = allocation
    return selected, selected_pairs, allocations


def build_reallocated_tokenizer_json(
    base_tokenizer_json: Dict[str, object],
    language_tokenizer_json: Dict[str, object],
    removed_token_ids: Sequence[int],
    replacement_tokens: Sequence[str],
    replacement_merge_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Tuple[Dict[str, object], Dict[str, int]]:
    if len(removed_token_ids) != len(replacement_tokens):
        raise ValueError("removed_token_ids and replacement_tokens must have equal length.")

    base_vocab, base_merges = _extract_bpe_model(base_tokenizer_json)
    lang_vocab, lang_merges = _extract_bpe_model(language_tokenizer_json)
    _ = lang_vocab

    id_to_base_token = {int(token_id): token for token, token_id in base_vocab.items()}
    removed_ids = [int(token_id) for token_id in removed_token_ids]
    removed_id_set = set(removed_ids)
    if len(removed_id_set) != len(removed_ids):
        raise ValueError(
            "removed_token_ids contains duplicates; each replacement needs a distinct slot."
        )
    protected_ids = {
        int(token["id"])
        for token in base_tokenizer_json.get("added_tokens", [])
        if token.get("special", False)
    }
    if removed_id_set & protected_ids:
        raise ValueError("Cannot reallocate a special-token ID.")

    missing_ids = [token_id for token_id in removed_ids if token_id not in id_to_base_token]
    if missing_ids:
        preview = ", ".join(str(x) for x in missing_ids[:8])
        raise ValueError(f"Some removed IDs are not in base model vocab: {preview}")

    if len(set(replacement_tokens)) != len(replacement_tokens):
        raise ValueError("replacement_tokens contains duplicates.")

    filtered_vocab: Dict[str, int] = {}
    for token, token_id in base_vocab.items():
        if int(token_id) in removed_id_set:
            continue
        filtered_vocab[token] = int(token_id)

    for token, token_id in zip(replacement_tokens, removed_ids):
        if token in filtered_vocab:
            raise ValueError(f"Replacement token already exists in base vocab: {token!r}")
        filtered_vocab[token] = int(token_id)

    kept_tokens = set(filtered_vocab.keys())
    use_list_format = bool(base_merges and isinstance(base_merges[0], (list, tuple)))

    filtered_base_merges: List[object] = []
    existing_pairs = set()
    for merge_entry in base_merges:
        pair = parse_merge_pair(merge_entry)
        if pair is None:
            continue
        left, right = pair
        merged = left + right
        if left in kept_tokens and right in kept_tokens and merged in kept_tokens:
            filtered_base_merges.append(merge_entry)
            existing_pairs.add((left, right))

    additional_merges: List[object] = []
    custom_pairs_added = 0
    if replacement_merge_pairs is not None:
        for pair in replacement_merge_pairs:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            left, right = str(pair[0]), str(pair[1])
            merged = left + right
            if (left, right) in existing_pairs:
                continue
            if left not in kept_tokens or right not in kept_tokens or merged not in kept_tokens:
                continue
            additional_merges.append(
                format_merge_entry(left, right, use_list_format=use_list_format)
            )
            existing_pairs.add((left, right))
            custom_pairs_added += 1
    else:
        produced_by: Dict[str, Tuple[str, str]] = {}
        for merge_entry in lang_merges:
            pair = parse_merge_pair(merge_entry)
            if pair is None:
                continue
            left, right = pair
            merged = left + right
            if merged not in produced_by:
                produced_by[merged] = (left, right)

        needed_tokens = set(replacement_tokens)
        stack = list(replacement_tokens)
        while stack:
            token = stack.pop()
            pair = produced_by.get(token)
            if pair is None:
                continue
            left, right = pair
            for part in (left, right):
                if part not in kept_tokens:
                    continue
                if part in needed_tokens:
                    continue
                needed_tokens.add(part)
                stack.append(part)

        for merge_entry in lang_merges:
            pair = parse_merge_pair(merge_entry)
            if pair is None:
                continue
            left, right = pair
            if (left, right) in existing_pairs:
                continue
            merged = left + right
            if merged not in needed_tokens:
                continue
            if left not in kept_tokens or right not in kept_tokens or merged not in kept_tokens:
                continue
            additional_merges.append(
                format_merge_entry(left, right, use_list_format=use_list_format)
            )
            existing_pairs.add((left, right))

    merged_tokenizer = copy.deepcopy(base_tokenizer_json)
    merged_tokenizer.setdefault("model", {})
    merged_tokenizer["model"]["vocab"] = filtered_vocab
    merged_tokenizer["model"]["merges"] = filtered_base_merges + additional_merges

    reachable = {token for token in filtered_vocab if len(token) == 1}
    reachable.update(
        token["content"]
        for token in merged_tokenizer.get("added_tokens", [])
        if token["content"] in filtered_vocab
    )
    for entry in merged_tokenizer["model"]["merges"]:
        pair = parse_merge_pair(entry)
        if pair and pair[0] in reachable and pair[1] in reachable:
            reachable.add(pair[0] + pair[1])
    unreachable = [token for token in replacement_tokens if token not in reachable]
    if unreachable:
        raise ValueError(
            f"Replacement tokens have no usable BPE merge: {unreachable[:8]!r}. "
            "Allocate their merge dependencies too, or choose another candidate."
        )

    stats = {
        "base_vocab_size": len(base_vocab),
        "new_vocab_size": len(filtered_vocab),
        "removed_token_count": len(removed_ids),
        "filtered_base_merges": len(filtered_base_merges),
        "added_language_merges": len(additional_merges),
        "added_custom_merge_pairs": int(custom_pairs_added),
        "used_custom_merge_pairs": bool(replacement_merge_pairs is not None),
        "total_merges": len(filtered_base_merges) + len(additional_merges),
    }
    return merged_tokenizer, stats


def build_reallocated_tokenizer(
    base_tokenizer,
    language_tokenizer,
    removed_token_ids: Sequence[int],
    replacement_tokens: Sequence[str],
    replacement_merge_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    trust_remote_code: bool = False,
):
    base_json = json.loads(base_tokenizer.backend_tokenizer.to_str())
    lang_json = json.loads(language_tokenizer.backend_tokenizer.to_str())
    merged_json, stats = build_reallocated_tokenizer_json(
        base_tokenizer_json=base_json,
        language_tokenizer_json=lang_json,
        removed_token_ids=removed_token_ids,
        replacement_tokens=replacement_tokens,
        replacement_merge_pairs=replacement_merge_pairs,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        base_tokenizer.save_pretrained(temp_dir)
        tokenizer_json_path = os.path.join(temp_dir, "tokenizer.json")
        with open(tokenizer_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_json, f, indent=2, ensure_ascii=False)

        try:
            new_tokenizer = AutoTokenizer.from_pretrained(
                temp_dir,
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )
        except Exception as exc:
            # Some custom tokenizer configs (e.g., cached ScaffoldTokenizer metadata) can break
            # AutoTokenizer class resolution when loading from a temp directory.
            print(
                f"Warning: AutoTokenizer load failed ({type(exc).__name__}: {exc}). "
                "Falling back to PreTrainedTokenizerFast(tokenizer.json)."
            )
            fallback_unk = (
                base_tokenizer.unk_token
                if getattr(base_tokenizer, "unk_token", None)
                else "<|unk|>"
            )
            new_tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=tokenizer_json_path,
                unk_token=fallback_unk,
            )

        # Preserve special tokens and key tokenizer behavior from the base tokenizer.
        if getattr(base_tokenizer, "eos_token", None):
            new_tokenizer.eos_token = base_tokenizer.eos_token
        if getattr(base_tokenizer, "bos_token", None):
            new_tokenizer.bos_token = base_tokenizer.bos_token
        if getattr(base_tokenizer, "pad_token", None):
            new_tokenizer.pad_token = base_tokenizer.pad_token
        if getattr(base_tokenizer, "unk_token", None):
            new_tokenizer.unk_token = base_tokenizer.unk_token
        if getattr(base_tokenizer, "sep_token", None):
            new_tokenizer.sep_token = base_tokenizer.sep_token
        if getattr(base_tokenizer, "cls_token", None):
            new_tokenizer.cls_token = base_tokenizer.cls_token
        if getattr(base_tokenizer, "mask_token", None):
            new_tokenizer.mask_token = base_tokenizer.mask_token
        if hasattr(base_tokenizer, "add_prefix_space"):
            new_tokenizer.add_prefix_space = bool(getattr(base_tokenizer, "add_prefix_space"))
    return new_tokenizer, stats


def evaluate_tokenizer(tokenizer, texts: Iterable[str], progress_desc: str) -> TokenizationMetrics:
    total_bytes = 0
    total_words = 0
    total_tokens = 0
    for text in tqdm(texts, desc=progress_desc, leave=False):
        total_bytes += len(text.encode("utf-8"))
        total_words += len(text.split())
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(token_ids)

    bytes_per_token = (float(total_bytes) / float(total_tokens)) if total_tokens > 0 else 0.0
    token_fertility = (float(total_tokens) / float(total_words)) if total_words > 0 else 0.0
    return TokenizationMetrics(
        total_bytes=total_bytes,
        total_words=total_words,
        total_tokens=total_tokens,
        bytes_per_token=bytes_per_token,
        token_fertility=token_fertility,
    )


def load_language_specs(path: Optional[str]) -> Dict[str, LanguageSpec]:
    if path is None:
        raw_specs = DEFAULT_LANGUAGE_SPECS
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw_specs = json.load(f)

    result: Dict[str, LanguageSpec] = {}
    if isinstance(raw_specs, dict):
        items = raw_specs.items()
    elif isinstance(raw_specs, list):
        items = []
        for item in raw_specs:
            if not isinstance(item, dict) or "language" not in item:
                raise ValueError(
                    "language_specs_path list entries must include a 'language' field."
                )
            items.append((item["language"], item))
    else:
        raise ValueError("language_specs_path must be a JSON object or list.")

    for language, payload in items:
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid language spec for {language}: expected object.")
        result[str(language)] = LanguageSpec(
            dataset=str(payload["dataset"]),
            dataset_config=payload.get("dataset_config"),
            split=str(payload.get("split", "train")),
            text_field=str(payload.get("text_field", "text")),
        )
    return result


def write_language_rows_csv(path: str, rows: List[Dict[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    fieldnames = [
        "language",
        "dataset",
        "dataset_config",
        "train_examples",
        "train_bytes",
        "eval_examples",
        "eval_bytes",
        "reallocated_slots",
        "candidate_pool_size",
        "baseline_bytes_per_token",
        "reallocated_bytes_per_token",
        "bytes_per_token_delta_abs",
        "bytes_per_token_delta_pct",
        "baseline_token_fertility",
        "reallocated_token_fertility",
        "token_fertility_delta_abs",
        "token_fertility_delta_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _fmt(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{digits}f}"


def _fmt_signed(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "NA"
    return f"{float(x):+.{digits}f}"


def _fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "NA"
    return f"{float(x):+.{digits}f}%"


def _pct_from_row(row: Dict[str, object], pct_key: str, abs_key: str, baseline_key: str) -> float:
    if pct_key in row and row[pct_key] is not None:
        return float(row[pct_key])
    baseline = float(row.get(baseline_key, 0.0))
    if baseline == 0.0:
        return 0.0
    return (float(row.get(abs_key, 0.0)) / baseline) * 100.0


LANGUAGE_DISPLAY_NAMES = {
    "ar": "Arabic",
    "ru": "Russian",
    "de": "German",
    "es": "Spanish",
    "en": "English",
}


def _language_display_name(code: str) -> str:
    key = str(code).strip().lower()
    return LANGUAGE_DISPLAY_NAMES.get(key, str(code))


def build_markdown_report(report: Dict[str, object]) -> str:
    config = report.get("config", {})
    summary = report.get("summary", {})
    macro = summary.get("macro", {})
    micro = summary.get("micro", {})
    per_language = report.get("per_language", [])

    lines: List[str] = []
    lines.append("# Freed-Slot Reallocation Tokenization-Efficiency")
    lines.append("")
    lines.append(
        "Fixed-vocabulary reallocation analysis: removed English slots are reassigned to language-specific BPE tokens, then evaluated on held-out in-domain text."
    )
    lines.append("")
    lines.append(f"- Base tokenizer: `{config.get('base_tokenizer', 'NA')}`")
    lines.append(f"- Removed-token source: `{config.get('removed_tokens_json', 'NA')}`")
    lines.append(
        f"- Reallocated slots used: `{config.get('num_reallocated_tokens', 'NA')}` "
        f"(raw removed IDs: `{config.get('removed_tokens_raw_count', 'NA')}`, usable: `{config.get('removed_tokens_usable_count', 'NA')}`)"
    )
    lines.append(
        "- Metrics: `bytes/token = total_utf8_bytes / total_tokens`; "
        "`token_fertility = total_tokens / total_whitespace_words`."
    )
    lines.append("")
    lines.append(
        "Table 1. Held-out bytes-per-token before vs. after reallocation (per language and macro average)."
    )
    lines.append("")
    lines.append(
        "| Language | Baseline Bytes/Token | Reallocated Bytes/Token | Δ Bytes/Token | Δ Bytes/Token (%) |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for row in per_language:
        bpt_delta_pct = _pct_from_row(
            row=row,
            pct_key="bytes_per_token_delta_pct",
            abs_key="bytes_per_token_delta_abs",
            baseline_key="baseline_bytes_per_token",
        )
        lines.append(
            "| "
            f"{_language_display_name(row.get('language', 'NA'))} | "
            f"{_fmt(row.get('baseline_bytes_per_token'), digits=2)} | "
            f"{_fmt(row.get('reallocated_bytes_per_token'), digits=2)} | "
            f"{_fmt_signed(row.get('bytes_per_token_delta_abs'), digits=2)} | "
            f"{_fmt_pct(bpt_delta_pct, digits=1)} |"
        )
    macro_bpt_delta_pct = _pct_from_row(
        row=macro,
        pct_key="bytes_per_token_delta_pct",
        abs_key="bytes_per_token_delta_abs",
        baseline_key="baseline_bytes_per_token",
    )
    lines.append(
        "| "
        f"average | "
        f"{_fmt(macro.get('baseline_bytes_per_token'), digits=2)} | "
        f"{_fmt(macro.get('reallocated_bytes_per_token'), digits=2)} | "
        f"{_fmt_signed(macro.get('bytes_per_token_delta_abs'), digits=2)} | "
        f"{_fmt_pct(macro_bpt_delta_pct, digits=1)} |"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reallocate freed slots in one multilingual BPE tokenizer."
    )
    parser.add_argument("--base_tokenizer", default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--removed_tokens_json",
        required=True,
        help="JSON list of freed token IDs or objects with a token_id field.",
    )
    parser.add_argument("--tokens_per_language", type=int, default=2500)
    parser.add_argument("--languages", nargs="+", default=["ar", "ru", "de", "es"])
    parser.add_argument("--language_specs_path", default=None)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--train_max_examples", default="200000")
    parser.add_argument("--train_max_bytes", default="200M")
    parser.add_argument("--eval_max_examples", default="20000")
    parser.add_argument("--eval_max_bytes", default="50M")
    parser.add_argument("--language_bpe_vocab_size", type=int, default=65536)
    parser.add_argument("--language_bpe_min_frequency", type=int, default=2)
    parser.add_argument("--overwrite_language_bpe_cache", action="store_true")
    parser.add_argument("--output_root", default="outputs/reallocation")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(set(args.languages)) != len(args.languages):
        raise ValueError("Each language must appear exactly once.")
    if args.tokens_per_language <= 0:
        raise ValueError("tokens_per_language must be positive.")
    language_specs = load_language_specs(args.language_specs_path)
    missing = set(args.languages) - language_specs.keys()
    if missing:
        raise ValueError(f"Missing language specs: {sorted(missing)}")
    caps = {
        key: parse_count(getattr(args, key))
        for key in ("train_max_examples", "train_max_bytes", "eval_max_examples", "eval_max_bytes")
    }
    if any(value is not None and value <= 0 for value in caps.values()):
        raise ValueError("Training and evaluation caps must be positive.")
    if caps["train_max_examples"] is None and caps["train_max_bytes"] is None:
        raise ValueError("Set a finite training cap so held-out documents can be selected.")
    ensure_dir(args.output_root)
    baseline = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    baseline_json = json.loads(baseline.backend_tokenizer.to_str())
    baseline_vocab, _ = _extract_bpe_model(baseline_json)
    raw_ids = load_removed_token_ids(args.removed_tokens_json)
    special = set(collect_special_token_ids(baseline))
    valid_ids = set(baseline_vocab.values())
    usable = [i for i in raw_ids if i not in special and i in valid_ids]
    n_slots = args.tokens_per_language * len(args.languages)
    if len(usable) < n_slots:
        raise ValueError(
            f"Need {n_slots} freed slots; only {len(usable)} usable IDs were provided."
        )
    removed = usable[:n_slots]
    language_tokenizers, language_jsons, evaluation, metadata = {}, {}, {}, {}
    for language in args.languages:
        spec = language_specs[language]
        train, held_out, counts = collect_train_eval_texts(
            spec=spec,
            streaming=args.streaming,
            cache_dir=args.cache_dir,
            min_text_chars=args.min_text_chars,
            **caps,
        )
        if not train or not held_out:
            raise ValueError(f"Need nonempty training and held-out text for {language}.")
        tokenizer, tokenizer_metadata, cache_path = train_language_bpe(
            language=language,
            train_texts=train,
            spec=spec,
            base_tokenizer=baseline,
            vocab_size=args.language_bpe_vocab_size,
            min_frequency=args.language_bpe_min_frequency,
            train_max_examples=caps["train_max_examples"],
            train_max_bytes=caps["train_max_bytes"],
            cache_dir=os.path.join(args.output_root, "language_bpe_cache"),
            overwrite_cache=args.overwrite_language_bpe_cache,
        )
        language_tokenizers[language] = tokenizer
        language_jsons[language] = json.loads(tokenizer.backend_tokenizer.to_str())
        evaluation[language] = held_out
        metadata[language] = {
            "text_counts": counts,
            "tokenizer": tokenizer_metadata,
            "cache_path": cache_path,
        }
        del train
    replacements, merge_pairs, allocation = select_joint_replacements(
        baseline_json, language_jsons, removed, args.tokens_per_language
    )
    joint_tokenizer, allocation_stats = build_reallocated_tokenizer(
        baseline,
        next(iter(language_tokenizers.values())),
        removed,
        replacements,
        replacement_merge_pairs=merge_pairs,
    )
    if len(joint_tokenizer) != len(baseline):
        raise ValueError("Reallocation changed the total tokenizer vocabulary size.")
    joint_tokenizer.save_pretrained(os.path.join(args.output_root, "tokenizer"))
    allocation_report = {
        language: [
            {"token_id": removed[offset + i], "token": token} for i, token in enumerate(tokens)
        ]
        for offset, (language, tokens) in (
            (index * args.tokens_per_language, item)
            for index, item in enumerate(allocation.items())
        )
    }
    with open(os.path.join(args.output_root, "allocation.json"), "w", encoding="utf-8") as f:
        json.dump(allocation_report, f, indent=2, ensure_ascii=False)
    rows = []
    for language in args.languages:
        spec = language_specs[language]
        base = evaluate_tokenizer(baseline, evaluation[language], f"{language}: baseline")
        joint = evaluate_tokenizer(
            joint_tokenizer, evaluation[language], f"{language}: reallocated"
        )
        delta = joint.bytes_per_token - base.bytes_per_token
        fertility_delta = base.token_fertility - joint.token_fertility
        rows.append(
            {
                "language": language,
                "dataset": spec.dataset,
                "dataset_config": spec.dataset_config,
                **metadata[language]["text_counts"],
                "reallocated_slots": args.tokens_per_language,
                "baseline_bytes_per_token": base.bytes_per_token,
                "reallocated_bytes_per_token": joint.bytes_per_token,
                "bytes_per_token_delta_abs": delta,
                "bytes_per_token_delta_pct": relative_pct(delta, base.bytes_per_token),
                "baseline_token_fertility": base.token_fertility,
                "reallocated_token_fertility": joint.token_fertility,
                "token_fertility_delta_abs": fertility_delta,
                "token_fertility_delta_pct": relative_pct(fertility_delta, base.token_fertility),
                "baseline_metrics": asdict(base),
                "reallocated_metrics": asdict(joint),
            }
        )
    macro = {}
    for metric in ("bytes_per_token", "token_fertility"):
        for prefix in ("baseline", "reallocated"):
            key = f"{prefix}_{metric}"
            macro[key] = sum(row[key] for row in rows) / len(rows)
    for metric in ("bytes_per_token", "token_fertility"):
        base_value = macro[f"baseline_{metric}"]
        delta = macro[f"reallocated_{metric}"] - base_value
        if metric == "token_fertility":
            delta = -delta
        macro[f"{metric}_delta_abs"] = delta
        macro[f"{metric}_delta_pct"] = relative_pct(delta, base_value)
    config = {
        **vars(args),
        **caps,
        "num_reallocated_tokens": n_slots,
        "removed_tokens_raw_count": len(raw_ids),
        "removed_tokens_usable_count": len(usable),
        "allocation_mode": "joint",
    }
    report = {
        "config": config,
        "summary": {"macro": macro},
        "per_language": rows,
        "allocation_stats": allocation_stats,
        "language_metadata": metadata,
    }
    with open(os.path.join(args.output_root, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_language_rows_csv(os.path.join(args.output_root, "language_metrics.csv"), rows)
    with open(os.path.join(args.output_root, "report.md"), "w", encoding="utf-8") as f:
        f.write(build_markdown_report(report))
    print(f"Saved one joint tokenizer and held-out metrics to {args.output_root}")


if __name__ == "__main__":
    main()
