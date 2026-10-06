import re
from datasets import load_dataset, Dataset, DatasetDict
from typing import Union, Optional, Callable, List
import itertools
from itertools import chain
from tqdm import tqdm
from collections import Counter
from accelerate import Accelerator

WIKI40B_LANGUAGES_TO_DECODE_FROM_BYTES = [
    "he",
    "de",
    "ja",
    "fa",
    "tr",
    "fr",
    "uk",
    "es",
    "it",
    "nl",
    "ru",
    "pt",
]
STREAMING_DATASETS = ["fineweb-edu", "fineweb-2"]


def load_pg19_val_and_test(num_examples=None):
    # Load the dataset in streaming mode
    streaming_dataset = load_dataset(
        "deepmind/pg19", split=None, streaming=True, trust_remote_code=True, revision="main"
    )

    # Extract test and validation splits with optional limit
    test_split = list(itertools.islice(streaming_dataset["test"], num_examples))
    validation_split = list(itertools.islice(streaming_dataset["validation"], num_examples))

    # Convert them into regular datasets
    test_dataset = Dataset.from_list(test_split)
    validation_dataset = Dataset.from_list(validation_split)

    return DatasetDict({"validation": validation_dataset, "test": test_dataset})


def load_pubmed(n_samples=10000):
    # Load the dataset in streaming mode
    streaming_dataset = load_dataset("MedRAG/pubmed", streaming=True, revision="main")

    # Extract test and validation splits
    data = list(streaming_dataset["train"].take(n_samples * 4))
    train = data[: 2 * n_samples]
    validation = data[2 * n_samples : 3 * n_samples]
    test = data[3 * n_samples :]
    # Convert them into regular datasets
    train = Dataset.from_list(train)
    validation = Dataset.from_list(validation)
    test = Dataset.from_list(test)
    dataset = DatasetDict({"train": train, "validation": validation, "test": test})
    dataset = dataset.rename_column("content", "text")
    return dataset


def load_sampled_dataset(
    dataset_name: str,
    subset_name: str,
    num_samples: int = 60000,
    split: Optional[str] = None,
    seed: Optional[int] = None,
) -> Union[Dataset, DatasetDict]:
    """
    Load a sample from the HuggingFaceFW/fineweb-2 dataset using streaming mode,
    then convert back to a regular dataset.

    Args:
        dataset_name: dataset
        subset_name: Language code for the dataset
        num_samples: Number of samples to take from each split
        split: Specific split to load (e.g., 'train'). If None, loads all splits
        seed: Random seed for reproducible sampling

    Returns:
        Dataset if split is specified, DatasetDict if split is None
    """

    # Load the dataset in streaming mode to get split information
    streaming_dataset = load_dataset(
        dataset_name, subset_name, split=None, streaming=True, revision="main"
    )

    def sample_from_split(dataset_split, n_samples: int) -> Dataset:
        """Sample n_samples from a streaming dataset split"""
        # Take the first n_samples from the streaming dataset
        samples = []
        for i, example in enumerate(dataset_split):
            if i >= n_samples:
                break
            samples.append(example)

        # Convert list of samples back to Dataset
        if samples:
            # Get all keys from the first example
            keys = samples[0].keys()
            # Create a dictionary with lists for each key
            data_dict = {key: [sample[key] for sample in samples] for key in keys}
            return Dataset.from_dict(data_dict)
        else:
            # Return empty dataset if no samples
            return Dataset.from_dict({})

    # If a specific split is requested
    if split is not None:
        if split not in streaming_dataset:
            raise ValueError(
                f"Split '{split}' not found in dataset. Available splits: {list(streaming_dataset.keys())}"
            )

        dataset_split = streaming_dataset[split]
        if seed is not None:
            dataset_split = dataset_split.shuffle(seed=seed)

        return sample_from_split(dataset_split, num_samples)

    # If no specific split requested, sample from all splits
    sampled_splits = {}

    for split_name, dataset_split in streaming_dataset.items():
        print(f"Sampling {num_samples} examples from split: {split_name}")

        if seed is not None:
            dataset_split = dataset_split.shuffle(seed=seed)

        sampled_splits[split_name] = sample_from_split(dataset_split, num_samples)

    return DatasetDict(sampled_splits)


def load_lm_dataset(dataset_name, language="english", split=None, num_examples=None):
    """
    Loads a popular pretraining or perplexity evaluation dataset by name and language.

    Args:
        dataset_name (str): The name of the dataset to load. Options include:
            - 'wikitext' (wikitext-2, smaller WikiText dataset)
            - 'wikitext-103' (larger WikiText dataset)
            - 'pg19' (Project Gutenberg dataset for long-context modeling)
            - 'c4' (Common Crawl-based English corpus)
            - 'wiki40b' (Wikipedia dataset in multiple languages)
            - 'mc4' (Multilingual C4 dataset in various languages)
        language (str): Language code for datasets that support multilingual options (e.g., 'en' for English).
                        Defaults to 'en'.

    Returns:
        Dataset: Loaded Hugging Face dataset.
    """
    if dataset_name.lower() == "wikitext":
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split=split, revision="main"
        )
    elif dataset_name.lower() == "fineweb-edu":
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", revision="main")
    elif dataset_name.lower() == "fineweb-2":
        lang_to_code = {
            "turkish": "tur_Latn",
            "russian": "rus_Cyrl",
            "hebrew": "heb_Hebr",
            "dutch": "nld_Latn",
            "arabic": "arb_Arab",
            "french": "fra_Latn",
            "spanish": "spa_Latn",
            "german": "deu_Latn",
            "portuguese": "por_Latn",
            "ukrainian": "ukr_Cyrl",
        }
        lang_code = lang_to_code[language]
        dataset = load_sampled_dataset("HuggingFaceFW/fineweb-2", lang_code, split=split)
    elif dataset_name.lower() == "wikitext-103":
        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split, revision="main"
        )
    elif dataset_name.lower() == "cord19":
        dataset = load_dataset(
            "allenai/cord19", "fulltext", trust_remote_code=True, revision="main"
        )
    elif dataset_name.lower() == "pubmed":
        return load_pubmed(num_examples)
    elif dataset_name.lower() == "wikilingua":
        dataset = load_dataset("GEM/wiki_lingua", trust_remote_code=True, revision="main")
        dataset = dataset.filter(
            lambda ex: (ex["source_language"] == "en") & (ex["target_language"] == "en")
        )
        dataset = dataset.rename_column("source", "text")
        dataset = dataset.rename_column("target", "summary")
    elif dataset_name.lower() == "xsum":
        dataset = load_dataset("EdinburghNLP/xsum", revision="main")
        dataset = dataset.rename_column("document", "text")
    elif dataset_name.lower() == "cnn":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", revision="main")
        dataset = dataset.rename_column("article", "text")
        dataset = dataset.rename_column("highlights", "summary")
        dataset = dataset.map(lambda example: {"text": example["text"].replace("(CNN)", "")})
    elif dataset_name.lower() == "pg19":
        return load_pg19_val_and_test(num_examples)
    elif dataset_name.lower() == "wiki40b":
        lang_code = language[:2]
        dataset = load_dataset("google/wiki40b", lang_code, split=split, revision="main")
        if lang_code in WIKI40B_LANGUAGES_TO_DECODE_FROM_BYTES:
            dataset = dataset.map(
                lambda x: {
                    "text": bytes(x["text"][2:-1], "utf-8")
                    .decode("unicode_escape")
                    .encode("latin1")
                    .decode("utf-8")
                    .replace("_NEWLINE_", "\n")
                }
            )
    else:
        raise ValueError(
            "Dataset not recognized. Available options: 'wikitext-2', 'wikitext-103', 'pg19', 'c4', 'wiki40b', 'mc4'."
        )

    if num_examples is not None:
        # Limit the number of examples if specified
        limited_dataset = {
            split: Dataset.from_list(list(itertools.islice(dataset[split], num_examples)))
            for split in dataset
        }
        return DatasetDict(limited_dataset)
    else:
        return dataset


def _get_bos_token_id(tokenizer):
    if hasattr(tokenizer, "orig_bos_token_id"):
        bos_token_id = tokenizer.orig_bos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
    return bos_token_id


def _get_eos_token_id(tokenizer):
    if hasattr(tokenizer, "orig_eos_token_id"):
        return tokenizer.orig_eos_token_id
    return tokenizer.eos_token_id


def get_group_texts_func(
    block_size=1024,
    add_start_token=False,
    start_token_id=None,
    add_end_token=False,
    end_token_id=None,
    eos_token_id=None,
    pack_documents_with_eos=False,
):
    def group_texts(examples):
        if block_size is None or block_size <= 0:
            raise ValueError("block_size must be a positive integer for grouping.")

        feature_names = list(examples.keys())
        start_ids = [start_token_id] if add_start_token and start_token_id is not None else []
        start_one = [1] if start_ids else []
        start_zero = [0] if start_ids else []
        start_label = [-100] * len(start_ids)

        result = dict()
        feature_start = {}
        feature_end = {}

        end_ids = [end_token_id] if add_end_token and end_token_id is not None else []
        end_one = [1] if end_ids else []
        end_zero = [0] if end_ids else []
        end_label = end_ids

        for k in feature_names:
            if k == "input_ids":
                feature_start[k] = start_ids
                feature_end[k] = end_ids
            elif k == "attention_mask":
                feature_start[k] = start_one
                feature_end[k] = end_one
            elif k == "labels":
                feature_start[k] = start_label
                feature_end[k] = end_label
            else:
                feature_start[k] = start_zero
                feature_end[k] = end_zero

        if pack_documents_with_eos:
            concatenated_examples = {k: [] for k in feature_names}
            eos_tokens = {k: [] for k in feature_names}
            if eos_token_id is not None:
                for k in feature_names:
                    if k == "input_ids":
                        eos_tokens[k] = [eos_token_id]
                    elif k == "attention_mask":
                        eos_tokens[k] = [1]
                    elif k == "labels":
                        eos_tokens[k] = [eos_token_id]
                    else:
                        eos_tokens[k] = [0]

            num_sequences = len(examples[feature_names[0]])
            for idx in range(num_sequences):
                for key in feature_names:
                    concatenated_examples[key].extend(examples[key][idx])
                if eos_token_id is not None:
                    for key in feature_names:
                        concatenated_examples[key].extend(eos_tokens[key])

            total_length = len(concatenated_examples[feature_names[0]])
            total_length = (total_length // block_size) * block_size

            for k, t in concatenated_examples.items():
                t = t[:total_length]
                chunks = []
                for i in range(0, total_length, block_size):
                    core = t[i : i + block_size]
                    chunks.append(feature_start.get(k, []) + core)
                result[k] = chunks
        else:
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[feature_names[0]])
            total_length = (total_length // block_size) * block_size

            for k, t in concatenated_examples.items():
                chunks = []
                for i in range(0, total_length, block_size):
                    core = t[i : i + block_size]
                    chunks.append(feature_start.get(k, []) + core + feature_end.get(k, []))
                result[k] = chunks

        if "labels" not in result and "input_ids" in result:
            result["labels"] = result["input_ids"].copy()
        return result

    return group_texts


def get_tokenize_func(tokenizer, text_col_name):
    def _tokenize(examples):
        output = tokenizer(
            examples[text_col_name],
            add_special_tokens=False,
        )
        return output

    return _tokenize


def get_sft_tokenize_func(tokenizer):
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("SFT mode requires a tokenizer with `apply_chat_template` support.")

    eos_token_id = _get_eos_token_id(tokenizer)

    def _tokenize(examples):
        input_ids_batch: List[List[int]] = []
        attention_masks_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []

        for messages in examples["messages"]:
            if not messages or messages[-1]["role"] != "assistant":
                input_ids_batch.append([])
                attention_masks_batch.append([])
                labels_batch.append([])
                continue

            full_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            full_ids = list(full_ids) if full_ids is not None else []
            if not full_ids:
                input_ids_batch.append([])
                attention_masks_batch.append([])
                labels_batch.append([])
                continue

            if len(messages) > 1:
                prompt_ids = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                prompt_len = len(prompt_ids)
            else:
                prompt_len = 0

            prompt_len = min(prompt_len, len(full_ids))
            attention_mask = [1] * len(full_ids)
            labels = list(full_ids)
            if prompt_len:
                labels[:prompt_len] = [-100] * prompt_len
            if eos_token_id is not None:
                labels = [
                    token if token == eos_token_id else label
                    for token, label in zip(full_ids, labels)
                ]

            input_ids_batch.append(full_ids)
            attention_masks_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_masks_batch,
            "labels": labels_batch,
        }

    return _tokenize


def tokenize_and_prepare_dataset(
    dataset,
    tokenizer,
    accelerator=None,
    text_col_name: str = "text",
    max_length: int = 256,
    max_samples: int = None,
    preprocessing_num_workers: int = 4,
    overwrite_cache: bool = False,
    add_start_token_after_grouping: bool = False,
    pack_documents_with_eos: bool = False,
    tokenize_function: Optional[Callable] = None,
):
    eos_token_id = _get_eos_token_id(tokenizer)
    bos_token_id = _get_bos_token_id(tokenizer)

    add_end_token_after_grouping = False

    if not (
        tokenizer.bos_token is not None
        and bos_token_id is not None
        and max_length
        and add_start_token_after_grouping
    ):
        add_start_token_after_grouping = False

    max_tokenized_len = max_length
    if max_tokenized_len is None:
        raise ValueError("max_length must be specified for dataset preparation.")

    if add_start_token_after_grouping:
        max_tokenized_len -= 1

    if not pack_documents_with_eos and eos_token_id is not None and max_tokenized_len:
        max_tokenized_len -= 1
        add_end_token_after_grouping = True

    if max_tokenized_len <= 0:
        raise ValueError("max_length is too small once special tokens are accounted for.")

    if tokenize_function is None:
        tokenize_function = get_tokenize_func(tokenizer, text_col_name)

    column_names = dataset.column_names

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    group_texts = get_group_texts_func(
        block_size=max_tokenized_len,
        add_start_token=add_start_token_after_grouping,
        start_token_id=bos_token_id if add_start_token_after_grouping else None,
        add_end_token=add_end_token_after_grouping,
        end_token_id=eos_token_id if add_end_token_after_grouping else None,
        eos_token_id=eos_token_id,
        pack_documents_with_eos=pack_documents_with_eos,
    )
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
    )

    if max_samples and max_samples < len(lm_dataset):
        lm_dataset = lm_dataset.select(range(max_samples))

    return lm_dataset
