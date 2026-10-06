"""
Dataset tokenization script for language model pretraining.

Examples:
    # Basic usage with default gpt2-large tokenizer, process 1B tokens
    python tokenize_dataset.py --max_tokens 2B

    # Use different tokenizer and process 500M tokens
    python tokenize_dataset.py --tokenizer microsoft/DialoGPT-medium --max_tokens 500M

    # Custom output directory and larger dataset config
    python tokenize_dataset.py --output_dir my_tokens --dataset_config sample-100BT --max_tokens 2B

    # Use different dataset entirely (e.g., C4)
    python tokenize_dataset.py --dataset allenai/c4 --dataset_config en --text_field text --max_tokens 1B

    # Process without token limit (full dataset)
    python tokenize_dataset.py --tokenizer google-t5/t5-small
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# Global variables for multiprocessing
TOKENIZER = None
TEXT_FIELD = None


def init_worker(tokenizer_name, text_field):
    """Initialize worker process with tokenizer."""
    global TOKENIZER, TEXT_FIELD
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TEXT_FIELD = text_field


def tokenize(doc):
    """Tokenize a document using the global tokenizer."""
    # Add EOS token at the start of each document (for document separation)
    if hasattr(TOKENIZER, "eos_token_id") and TOKENIZER.eos_token_id is not None:
        tokens = [TOKENIZER.eos_token_id]
    else:
        # TODO need cleaner solution here for tokenizers that don't have this special token
        tokens = TOKENIZER.encode("<|endoftext|>", add_special_tokens=False)

    # Encode the document text
    doc_tokens = TOKENIZER.encode(doc[TEXT_FIELD], add_special_tokens=False)
    tokens.extend(doc_tokens)

    tokens_np = np.array(tokens, dtype=np.uint16)
    assert (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np


def parse_token_count(s):
    """Parse token count string like '1B', '800M' into integer."""
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    else:
        return int(s)


def write_datafile(filename, toks, dtype=np.uint16):
    assert len(toks) < 2**31, "token count too large"
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1  # version
    header[2] = len(toks)

    if not isinstance(toks, np.ndarray):
        if dtype == np.uint16:
            assert all(0 <= t < 2**16 for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=dtype)
    else:
        toks_np = toks
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())


def get_dataset(dataset_name, dataset_config, streaming=True):
    """Load dataset - modular for future dataset changes."""
    return load_dataset(dataset_name, name=dataset_config, split="train", streaming=streaming)


def iter_dataset_until_stop(dataset, stop_event):
    """Yield dataset items until a stop_event is set (for graceful pool shutdown)."""
    for item in dataset:
        if stop_event.is_set():
            break
        yield item


def main():
    parser = argparse.ArgumentParser(description="Dataset tokenization for pretraining")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="openai-community/gpt2-large",
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--dataset", type=str, default="HuggingFaceFW/fineweb", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--dataset_config", type=str, default="sample-10BT", help="Dataset configuration"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory (defaults to tokenizer name)"
    )
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes for tokenization (default: 8)",
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
    args = parser.parse_args()
    mp.set_start_method(args.mp_start_method, force=True)

    # Setup output directory - extract name from tokenizer path (handles HF Hub and local paths)
    if args.output_dir is None:
        args.output_dir = os.path.basename(args.tokenizer.rstrip(os.sep))
        if not args.output_dir:  # Handle edge case
            args.output_dir = args.tokenizer.replace(os.sep, "_").replace("/", "_")

    DATA_CACHE_DIR = os.path.join(
        os.path.dirname(__file__),
        "data",
        f"{args.dataset.split('/')[-1]}_{args.dataset_config}",
        args.output_dir,
    )
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    fw = get_dataset(args.dataset, args.dataset_config, streaming=True)

    # Parse max tokens
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
    nprocs = args.num_workers
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
    early_stop = False
    ctx = mp.get_context(args.mp_start_method)
    stop_event = ctx.Event()

    with ctx.Pool(
        nprocs,
        initializer=init_worker,
        initargs=(args.tokenizer, args.text_field),
        maxtasksperchild=args.maxtasksperchild,
    ) as pool:
        split_buffers = {
            "val": {
                "tokens": np.empty((args.shard_size,), dtype=np.uint16),
                "count": 0,
                "shard_index": 0,
                "tokens_written": 0,
            },
            "train": {
                "tokens": np.empty((args.shard_size,), dtype=np.uint16),
                "count": 0,
                "shard_index": 0,
                "tokens_written": 0,
            },
        }

        def flush_split(split_name, force=False):
            state = split_buffers[split_name]
            if state["count"] == 0:
                return
            if not force and state["count"] < args.shard_size:
                return
            n = state["count"] if force else args.shard_size
            filename = os.path.join(
                DATA_CACHE_DIR,
                f"data_{split_name}_{state['shard_index']:06d}.bin",
            )
            write_datafile(filename, state["tokens"][:n])
            state["shard_index"] += 1
            state["tokens_written"] += n
            state["count"] = 0

        def append_to_split(split_name, tokens):
            state = split_buffers[split_name]
            pos = 0
            while pos < len(tokens):
                remaining = args.shard_size - state["count"]
                take = min(remaining, len(tokens) - pos)
                state["tokens"][state["count"] : state["count"] + take] = tokens[pos : pos + take]
                state["count"] += take
                pos += take
                if state["count"] == args.shard_size:
                    flush_split(split_name, force=False)

        def append_legacy_split(tokens):
            """Legacy behavior: fill first val shard by tokens, then route everything to train."""
            pos = 0
            while pos < len(tokens):
                val_filled = split_buffers["val"]["tokens_written"] + split_buffers["val"]["count"]
                if val_filled < args.shard_size:
                    split_name = "val"
                    cap = args.shard_size - val_filled
                else:
                    split_name = "train"
                    cap = len(tokens) - pos
                take = min(cap, len(tokens) - pos)
                append_to_split(split_name, tokens[pos : pos + take])
                pos += take

        def split_token_count(split_name):
            state = split_buffers[split_name]
            return state["tokens_written"] + state["count"]

        progress_total = None if max_tokens == float("inf") else int(max_tokens)
        progress_bar = tqdm(total=progress_total, unit="tokens", desc="Tokenizing")
        doc_index = 0

        for tokens in pool.imap(tokenize, iter_dataset_until_stop(fw, stop_event), chunksize=16):
            if early_stop:
                continue

            if args.val_docs > 0:
                split = "val" if doc_index < args.val_docs else "train"
                append_to_split(split, tokens)
            else:
                append_legacy_split(tokens)
            doc_index += 1
            if args.val_docs > 0 and min_val_tokens is not None and doc_index == args.val_docs:
                final_val_tokens = split_token_count("val")
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

        flush_split("val", force=True)
        flush_split("train", force=True)
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

    print(f"Total tokens processed: {total_tokens_processed:,}")


if __name__ == "__main__":
    main()
