"""
Train a BPE tokenizer from a streaming dataset, matching a base tokenizer's
pre-tokenization and special tokens when provided.
"""

import argparse
import os
from typing import Iterable, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def parse_count(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if value.endswith("B"):
        return int(float(value[:-1]) * 1e9)
    if value.endswith("M"):
        return int(float(value[:-1]) * 1e6)
    return int(value)


def parse_limits(args) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    max_tokens = parse_count(args.max_tokens)
    max_examples = parse_count(args.max_examples)
    max_bytes = parse_count(args.max_bytes)
    max_chars = parse_count(args.max_chars)
    return max_tokens, max_examples, max_bytes, max_chars


def format_count(value: Optional[int], suffix: str) -> Optional[str]:
    if value is None:
        return None
    for unit, factor in (("B", 1_000_000_000), ("M", 1_000_000)):
        if value % factor == 0:
            return f"{value // factor}{unit}{suffix}"
    return f"{value}{suffix}"


def build_output_dir(
    args, vocab_size: int, limits: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]
) -> str:
    max_tokens, max_examples, max_bytes, max_chars = limits
    dataset_name = args.dataset.split("/")[-1]
    config = args.dataset_config or "default"
    base_name = (
        os.path.basename(args.base_tokenizer.rstrip(os.sep)) if args.base_tokenizer else "none"
    )

    parts = [
        dataset_name,
        config,
        f"bpe{vocab_size}",
        f"base_{base_name}",
        "stream" if args.streaming else "cached",
    ]

    if max_tokens is not None:
        parts.append(format_count(max_tokens, "tok"))
    if max_bytes is not None:
        parts.append(format_count(max_bytes, "bytes"))
    if max_examples is not None:
        parts.append(format_count(max_examples, "ex"))
    if max_chars is not None:
        parts.append(format_count(max_chars, "chars"))

    return os.path.join(args.output_dir, "__".join(parts))


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


def get_token_counter(base_tokenizer):
    if base_tokenizer is None:
        return None

    def _count(text: str) -> int:
        return len(base_tokenizer.encode(text, add_special_tokens=False))

    return _count


def make_iterator(
    dataset,
    text_field: str,
    limits: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]],
    token_counter,
    batch_size: int,
):
    max_tokens, max_examples, max_bytes, max_chars = limits
    total_tokens = 0
    total_examples = 0
    total_bytes = 0
    total_chars = 0
    buffer: List[str] = []

    def _limits_reached() -> bool:
        if max_tokens is not None and total_tokens >= max_tokens:
            return True
        if max_examples is not None and total_examples >= max_examples:
            return True
        if max_bytes is not None and total_bytes >= max_bytes:
            return True
        if max_chars is not None and total_chars >= max_chars:
            return True
        return False

    for sample in dataset:
        text = sample.get(text_field)
        if not isinstance(text, str):
            continue
        total_examples += 1
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        if max_tokens is not None and token_counter is not None:
            total_tokens += token_counter(text)

        if batch_size > 1:
            buffer.append(text)
            if len(buffer) >= batch_size:
                yield buffer
                buffer = []
        else:
            yield text

        if _limits_reached():
            break

    if buffer:
        yield buffer


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from a streaming dataset")
    parser.add_argument(
        "--dataset", type=str, default="HuggingFaceFW/fineweb", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="sample-10BT", help="Dataset configuration name"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--text_field", type=str, default="text", help="Field name containing text")
    parser.add_argument(
        "--max_tokens",
        type=str,
        default=None,
        help="Maximum tokens to train on (e.g., '10B', '500M')",
    )
    parser.add_argument(
        "--max_examples", type=str, default=None, help="Maximum examples to train on (e.g., '5M')"
    )
    parser.add_argument(
        "--max_bytes", type=str, default=None, help="Maximum UTF-8 bytes to train on (e.g., '50B')"
    )
    parser.add_argument(
        "--max_chars", type=str, default=None, help="Maximum characters to train on (e.g., '50B')"
    )
    parser.add_argument("--vocab_size", type=int, default=None, help="Tokenizer vocab size")
    parser.add_argument("--min_frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tokenizers",
        help="Base output directory (default: tokenizers)",
    )
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        default="openai-community/gpt2",
        help="Base tokenizer to match pre-tokenization and special tokens",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use streaming dataset access (default: false, cache locally)",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Dataset cache directory (optional)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for iterator (reduces Python overhead)",
    )
    parser.add_argument(
        "--rayon_num_threads",
        type=int,
        default=None,
        help="Number of threads for tokenizers (sets RAYON_NUM_THREADS)",
    )
    args = parser.parse_args()

    if args.rayon_num_threads is not None:
        os.environ["RAYON_NUM_THREADS"] = str(args.rayon_num_threads)

    base_tokenizer = (
        AutoTokenizer.from_pretrained(args.base_tokenizer) if args.base_tokenizer else None
    )
    vocab_size = args.vocab_size or (base_tokenizer.vocab_size if base_tokenizer else None)
    if vocab_size is None:
        raise ValueError("vocab_size must be provided when no base_tokenizer is set.")

    fallback_eos = "<|endoftext|>"
    fallback_unk = "<|unk|>"
    special_tokens = get_special_tokens(base_tokenizer, fallback_eos, fallback_unk)
    unk_token = (
        base_tokenizer.unk_token if base_tokenizer and base_tokenizer.unk_token else fallback_unk
    )

    tokenizer = build_tokenizer(base_tokenizer, unk_token)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    load_kwargs = {
        "path": args.dataset,
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.dataset_config is not None:
        load_kwargs["name"] = args.dataset_config
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = args.cache_dir
    dataset = load_dataset(**load_kwargs)

    limits = parse_limits(args)
    max_tokens = limits[0]
    token_counter = get_token_counter(base_tokenizer)
    if max_tokens is not None and token_counter is None:
        raise ValueError("max_tokens requires a base_tokenizer for counting.")

    iterator = make_iterator(dataset, args.text_field, limits, token_counter, args.batch_size)
    tokenizer.train_from_iterator(iterator, trainer=trainer)

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
        if hasattr(base_tokenizer, "add_prefix_space"):
            fast.add_prefix_space = base_tokenizer.add_prefix_space
        if base_tokenizer.model_max_length:
            fast.model_max_length = base_tokenizer.model_max_length

    final_output_dir = build_output_dir(args, vocab_size, limits)
    os.makedirs(final_output_dir, exist_ok=True)
    fast.save_pretrained(final_output_dir)
    print(f"Saved tokenizer to {final_output_dir}")


if __name__ == "__main__":
    main()
