from analysis.aggregate_downstream import flatten_results


def test_flatten_downstream_standard_and_nested_evaluator_metrics():
    payload = {
        "boolq": {
            "baseline": {"acc,none": 0.61},
            "finetuned": {"acc,none": 0.63, "alias": "adapted"},
        },
        "mmlu": {
            "baseline": {"global_mmlu_full_en": {"acc_norm,none": 0.42}},
        },
    }
    rows = flatten_results(payload, model="model-x", language="en")
    assert len(rows) == 3
    base = rows[(rows["task"] == "boolq") & (rows["method"] == "baseline")].iloc[0]
    assert base["metric"] == "acc,none"
    assert base["value"] == 0.61
    nested = rows[rows["task"] == "mmlu/global_mmlu_full_en"].iloc[0]
    assert nested["metric"] == "acc_norm,none"
    assert nested["value"] == 0.42


def test_empty_payload_returns_stable_columns():
    rows = flatten_results({}, model="model-x", language="en")
    assert list(rows.columns) == ["model", "language", "task", "method", "metric", "value"]
    assert rows.empty
