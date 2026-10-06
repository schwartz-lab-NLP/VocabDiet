import math

import pytest

from eval_utils import (
    compute_bytes_per_token_metrics,
    compute_factorized_bytes_per_token_metrics,
)
from modeling_compositional import ConditionedTransformationHead, TypeConfig


def test_factorized_bpb_uses_joint_base_and_group_nll():
    metrics = compute_factorized_bytes_per_token_metrics(
        base_loss_per_token=2.0,
        type_loss_per_group=0.5,
        num_groups=4,
        avg_bytes_per_token=2.0,
    )

    assert metrics["joint_nll"] == pytest.approx(4.0)
    assert metrics["bpb"] == pytest.approx(4.0 / math.log(2.0) / 2.0)


@pytest.mark.parametrize("invalid_bytes", [0.0, -1.0])
def test_bpb_rejects_nonpositive_byte_denominator(invalid_bytes):
    with pytest.raises(ValueError, match="avg_bytes_per_token must be positive"):
        compute_bytes_per_token_metrics(avg_loss_per_token=1.0, avg_bytes_per_token=invalid_bytes)


def test_empty_composition_map_uses_head_device_without_cuda_side_effect():
    head = ConditionedTransformationHead(
        model_dim=8,
        type_vocab_size=2,
        base_vocab_size=4,
        type_config=TypeConfig({"inflection": 2}),
    )

    head.set_decomposition_map({})

    assert head.composition_lemma_base_indices.device.type == "cpu"
    assert head.base_rep_source == "unembedding"


def test_validation_bytes_count_prediction_targets_and_exclude_special_strings():
    import torch
    from eval_utils import compute_validation_tokenization_stats

    class Tokenizer:
        def decode(self, ids, skip_special_tokens):
            assert skip_special_tokens
            return "".join({0: "", 1: "é", 2: "a"}[i] for i in ids)

    batch = (torch.tensor([2, 1, 0]), torch.tensor([1, 0, 2]))
    total_bytes, _ = compute_validation_tokenization_stats(Tokenizer(), iter([batch]), 1)
    assert total_bytes == 3


def test_inapplicable_groups_contribute_zero_to_per_token_joint_nll():
    import torch
    from modeling_compositional import CompositionalGPT

    model = CompositionalGPT(
        4, 4, TypeConfig({"inflections": 2}), 0, 1, 8, 32, 128, use_linear_cross_entropy=False
    )
    logits = torch.zeros(2, 2)
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # second row is not applicable
    loss = model.compute_type_losses(labels, logits)
    assert loss == pytest.approx(math.log(2) / 2)
