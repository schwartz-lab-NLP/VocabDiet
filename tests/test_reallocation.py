"""Exercise slot accounting, BPE merges, and held-out UTF-8 metrics."""

import copy
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from analysis import reallocate_vocabulary as reallocation


@pytest.fixture
def tokenizers():
    baseline = Tokenizer(
        BPE(vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "<eos>": 4}, merges=[("a", "b")])
    )
    baseline.add_special_tokens(["<eos>"])
    language = Tokenizer(BPE(vocab={"a": 0, "b": 1, "c": 2, "bc": 3}, merges=[("b", "c")]))
    return (
        PreTrainedTokenizerFast(tokenizer_object=baseline, eos_token="<eos>"),
        PreTrainedTokenizerFast(tokenizer_object=language),
    )


def test_reallocation_preserves_budget_ids_special_tokens_and_source(tokenizers):
    baseline, language = tokenizers
    original = json.loads(baseline.backend_tokenizer.to_str())
    language_json = json.loads(language.backend_tokenizer.to_str())
    unchanged = copy.deepcopy(original)
    result, stats = reallocation.build_reallocated_tokenizer_json(
        original, language_json, [3], ["bc"]
    )
    assert original == unchanged
    assert result["model"]["vocab"] == {"a": 0, "b": 1, "c": 2, "bc": 3, "<eos>": 4}
    assert stats["new_vocab_size"] == stats["base_vocab_size"]
    tokenizer = Tokenizer.from_str(json.dumps(result))
    assert tokenizer.encode("bc").ids == [3]
    assert tokenizer.encode("ab").ids == [0, 1]
    assert tokenizer.token_to_id("<eos>") == 4


@pytest.mark.parametrize(
    "removed,replacements,message",
    [
        ([3, 3], ["bc", "cc"], "duplicates"),
        ([4], ["bc"], "special"),
        ([100], ["bc"], "not in base"),
        ([3], ["a"], "already exists"),
        ([3], [], "equal length"),
        ([3], ["zz"], "no usable BPE merge"),
    ],
)
def test_invalid_slot_allocations_fail(tokenizers, removed, replacements, message):
    baseline, language = tokenizers
    with pytest.raises(ValueError, match=message):
        reallocation.build_reallocated_tokenizer_json(
            json.loads(baseline.backend_tokenizer.to_str()),
            json.loads(language.backend_tokenizer.to_str()),
            removed,
            replacements,
        )


def test_metrics_count_utf8_bytes_and_exclude_special_tokens():
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text)

    metrics = reallocation.evaluate_tokenizer(CharacterTokenizer(), ["é", "猫 a"], "test")
    assert metrics.total_bytes == 7
    assert metrics.total_tokens == 4
    assert metrics.total_words == 3
    assert metrics.bytes_per_token == 7 / 4


def test_training_and_evaluation_use_separate_documents(monkeypatch):
    monkeypatch.setattr(
        reallocation,
        "load_dataset",
        lambda **kwargs: iter(
            [
                {"text": "training one"},
                {"text": "training two"},
                {"text": "held out"},
                {"text": "unused"},
            ]
        ),
    )
    train, evaluation, stats = reallocation.collect_train_eval_texts(
        reallocation.LanguageSpec("local", None, "train", "text"), True, None, 1, 2, None, 1, None
    )
    assert train == ["training one", "training two"]
    assert evaluation == ["held out"]
    assert stats["eval_bytes"] == len("held out".encode("utf-8"))
