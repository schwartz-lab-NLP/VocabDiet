import pandas as pd
import pytest

from analysis.aggregate_patchscopes import exemplar_correlations, summarize_resolution


def test_summarize_resolution_uses_single_type_and_token_count_splits():
    frame = pd.DataFrame(
        [
            {
                "word": "walks",
                "type": "V;PRS",
                "word_n_tokens": 1,
                "is_single_type": True,
                "first_layer_additive": 0,
            },
            {
                "word": "walked",
                "type": "V;PRS",
                "word_n_tokens": 2,
                "is_single_type": True,
                "first_layer_additive": 3,
            },
            {
                "word": "walkeded",
                "type": "V;PRS",
                "word_n_tokens": 2,
                "is_single_type": False,
                "first_layer_additive": 0,
            },
            {
                "word": "unwalk",
                "type": "un-",
                "word_n_tokens": 2,
                "is_single_type": True,
                "first_layer_additive": None,
            },
        ]
    )
    by_type, by_class = summarize_resolution(frame)
    inflection = by_type[by_type["type"] == "V;PRS"].iloc[0]
    assert inflection["iv_n"] == 1
    assert inflection["iv_embed_success"] == 1
    assert inflection["oov_n"] == 1
    assert inflection["oov_embed_success"] == 0
    assert inflection["oov_detok_success"] == 1
    derivation = by_class[by_class["category"] == "derivation"].iloc[0]
    assert derivation["oov_n"] == 1
    assert derivation["oov_detok_success"] == 0


def test_resolution_schema_errors_are_actionable():
    with pytest.raises(ValueError, match="missing columns"):
        summarize_resolution(pd.DataFrame({"word": ["x"]}))


def test_exemplar_correlations_use_iv_count_and_oov_case_requirements():
    per_type = pd.DataFrame(
        [
            {
                "type": f"V;T{i}",
                "category": "inflection",
                "iv_n": i,
                "iv_embed_success": i / 4,
                "oov_n": 1 if i < 4 else 0,
                "oov_embed_success": i / 4 if i < 4 else float("nan"),
            }
            for i in range(1, 5)
        ]
    )
    result = exemplar_correlations(per_type)
    overall_iv = result[(result["category"] == "all") & (result["target_split"] == "iv")].iloc[0]
    overall_oov = result[(result["category"] == "all") & (result["target_split"] == "oov")].iloc[0]
    assert overall_iv["n_transformations"] == 4
    assert overall_iv["spearman_rho"] == 1
    assert overall_oov["n_transformations"] == 3


def test_detokenization_requires_match_in_first_ten_layers():
    frame = pd.DataFrame(
        [
            {
                "word": str(layer),
                "type": "V;PST",
                "word_n_tokens": 1,
                "is_single_type": True,
                "first_layer_additive": layer,
            }
            for layer in (-1, 0, 10, 11)
        ]
    )
    by_type, _ = summarize_resolution(frame)
    assert by_type.iloc[0].iv_detok_success == 0.5
