import json
from argparse import Namespace

import pytest

from analysis import reallocate_vocabulary as rv


def _tokenizer_json(vocab, merges):
    return {"model": {"type": "BPE", "vocab": vocab, "merges": merges}, "added_tokens": []}


def test_select_joint_replacements_uses_rank_order_dependencies_and_disjoint_slots():
    base = _tokenizer_json({"a": 0, "b": 1, "c": 2, "ab": 3}, [["a", "b"]])
    # `bad` ranks first but needs a dependency that is not reachable. `ac` is
    # the first usable new token. The duplicate `ac` is skipped for language
    # two in favor of `bc`.
    lang_one = _tokenizer_json(
        {"a": 0, "b": 1, "c": 2, "bad": 3, "ac": 4},
        [["missing", "z"], ["a", "c"]],
    )
    lang_two = _tokenizer_json(
        {"a": 0, "b": 1, "c": 2, "ac": 3, "bc": 4},
        [["a", "c"], ["b", "c"]],
    )
    selected, pairs, allocation = rv.select_joint_replacements(
        base, {"one": lang_one, "two": lang_two}, [3, 4], slots_per_language=1
    )
    assert allocation == {"one": ["ac"], "two": ["bc"]}
    assert selected == ["ac", "bc"]
    assert pairs == [("a", "c"), ("b", "c")]
    assert len(set(selected)) == 2


def test_select_joint_replacements_fails_when_language_cannot_fill_distinct_quota():
    base = _tokenizer_json({"a": 0, "b": 1, "ab": 2}, [["a", "b"]])
    language = _tokenizer_json({"a": 0, "b": 1, "ab": 2}, [["a", "b"]])
    with pytest.raises(ValueError, match="only 0 usable distinct new tokens"):
        rv.select_joint_replacements(base, {"one": language}, [2], slots_per_language=1)


class _Backend:
    def __init__(self, payload):
        self.payload = payload

    def to_str(self):
        return json.dumps(self.payload)


class _Tokenizer:
    def __init__(self, payload, vocab_size=6):
        self.backend_tokenizer = _Backend(payload)
        self.all_special_tokens = []
        self.eos_token_id = self.bos_token_id = self.pad_token_id = None
        self.unk_token_id = self.sep_token_id = self.cls_token_id = self.mask_token_id = None
        self.vocab_size = vocab_size

    def __len__(self):
        return self.vocab_size

    def save_pretrained(self, path):
        return (path,)


def test_main_evaluates_one_joint_tokenizer_on_every_language(monkeypatch, tmp_path):
    languages = ["aa", "bb"]
    base_json = _tokenizer_json({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5}, [])
    baseline = _Tokenizer(base_json)
    joint = _Tokenizer(base_json)
    specs = {lang: rv.LanguageSpec("toy", lang, "train", "text") for lang in languages}
    args = Namespace(
        languages=languages,
        tokens_per_language=2,
        language_specs_path=None,
        train_max_examples="2",
        train_max_bytes="20",
        eval_max_examples="1",
        eval_max_bytes="10",
        streaming=True,
        cache_dir=None,
        min_text_chars=1,
        output_root=str(tmp_path),
        base_tokenizer="toy",
        removed_tokens_json="unused.json",
        language_bpe_vocab_size=32,
        language_bpe_min_frequency=1,
        overwrite_language_bpe_cache=False,
    )
    monkeypatch.setattr(rv, "parse_args", lambda: args)
    monkeypatch.setattr(rv, "load_language_specs", lambda _: specs)
    monkeypatch.setattr(rv, "load_removed_token_ids", lambda _: [2, 3, 4, 5])
    monkeypatch.setattr(
        rv,
        "AutoTokenizer",
        type("Auto", (), {"from_pretrained": staticmethod(lambda *a, **k: baseline)}),
    )
    monkeypatch.setattr(
        rv,
        "collect_train_eval_texts",
        lambda spec, **kwargs: (
            [f"train-{spec.dataset_config}"],
            [f"eval-{spec.dataset_config}"],
            {"train_examples": 1, "train_bytes": 1, "eval_examples": 1, "eval_bytes": 1},
        ),
    )
    monkeypatch.setattr(
        rv, "train_language_bpe", lambda **kwargs: (_Tokenizer(base_json), {}, None)
    )
    monkeypatch.setattr(
        rv,
        "select_joint_replacements",
        lambda *a, **k: (
            ["aa1", "aa2", "bb1", "bb2"],
            [],
            {"aa": ["aa1", "aa2"], "bb": ["bb1", "bb2"]},
        ),
    )
    monkeypatch.setattr(
        rv, "build_reallocated_tokenizer", lambda *a, **k: (joint, {"new_vocab_size": 6})
    )
    evaluated = []

    def fake_evaluate(tokenizer, texts, desc):
        evaluated.append((tokenizer, texts, desc))
        return rv.TokenizationMetrics(1, 1, 1, 1.0, 1.0)

    monkeypatch.setattr(rv, "evaluate_tokenizer", fake_evaluate)

    rv.main()

    assert [entry[1] for entry in evaluated] == [["eval-aa"], ["eval-aa"], ["eval-bb"], ["eval-bb"]]
    assert evaluated[0][0] is baseline and evaluated[2][0] is baseline
    assert evaluated[1][0] is joint and evaluated[3][0] is joint
    allocation = json.loads((tmp_path / "allocation.json").read_text())
    assert [item["token_id"] for item in allocation["aa"]] == [2, 3]
    assert [item["token_id"] for item in allocation["bb"]] == [4, 5]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["config"]["num_reallocated_tokens"] == 4
    assert report["config"]["allocation_mode"] == "joint"
