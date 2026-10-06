import sys
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, Qwen2Config, Olmo2Config, PreTrainedTokenizerFast

from posthoc.checkpoint import load_adaptation, model_class_for_config, save_adaptation
from posthoc.initialization import mean_offsets
from posthoc.tokenization import ScaffoldTokenizer


def tiny_adaptation(config_class):
    config = config_class(
        vocab_size=8,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        pad_token_id=0,
        bos_token_id=0,
        eos_token_id=0,
    )
    config._attn_implementation = "eager"
    groups = {"prefix": (0, 2), "capitalization": (2, 4), "inflections": (4, 6)}
    model = model_class_for_config(config)(
        config,
        types_vocab_size=6,
        model_vocab_size=8,
        extended_vocab_size=9,
        types_loss_indices=groups,
        types_loss_alpha=0,
        base_loss_alpha=0,
    )
    decomposition = {1: (1, [0, 2, 4]), 8: (1, [0, 2, 5])}
    model.set_token_id_to_base_id_mapping({8: 1})
    model.set_token_id_to_orig_first_id_mapping({8: 1})
    model.set_token_id_is_base_or_inflection_token([1], [8], [2])
    model.set_token_to_type_ids(decomposition, [0, 2, 4])
    model.set_token_types_filter(decomposition.keys(), [v[1] for v in decomposition.values()])
    model.set_negative_type_ids([0, 2, 4])
    model.set_ignored_logits_idx([7])
    model.set_logits_mapper(
        [1], decomposition, {i: str(i) for i in range(6)}, groups, [0, 2, 3, 4, 5, 6]
    )
    with torch.no_grad():
        model.embed_types.weight.zero_()
        model.types_lm_head.weight.zero_()
        model.embed_types.weight[5].fill_(0.02)
        model.types_lm_head.weight[5].fill_(0.03)
    base = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({f"t{i}": i for i in range(8)}, unk_token="t0")
        ),
        unk_token="t0",
        pad_token="t0",
        eos_token="t0",
    )
    extended = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({f"t{i}": i for i in range(9)}, unk_token="t0")
        ),
        unk_token="t0",
        pad_token="t0",
        eos_token="t0",
    )
    tokenizer = ScaffoldTokenizer(extended, base, {})
    return model.eval(), tokenizer, base


@pytest.mark.parametrize("config_class", [LlamaConfig, Qwen2Config, Olmo2Config])
def test_additive_forward_and_checkpoint_roundtrip(config_class, tmp_path):
    model, tokenizer, base = tiny_adaptation(config_class)
    inputs = torch.tensor([[1, 8, 2]])
    with torch.no_grad():
        expected = model(inputs, use_cache=False).logits
    assert expected.shape == (1, 3, 9)
    assert torch.isfinite(expected).all()
    save_adaptation(model, tokenizer, base, tmp_path)
    restored, restored_tokenizer = load_adaptation(tmp_path)
    with torch.no_grad():
        actual = restored(inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
    assert restored_tokenizer.encode("t8") == tokenizer.encode("t8")
    assert restored.generate(inputs, max_new_tokens=1, do_sample=False).shape == (1, 4)


def test_lora_checkpoint_merges_without_changing_logits(tmp_path):
    model, tokenizer, base = tiny_adaptation(LlamaConfig)
    model.k_last_layers = 1
    model.lora_ft = True
    model.lora_adapter_toggling = True
    model.apply_lora_to_last_layer(lora_r=2, lora_alpha=2)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.01)
    model.set_lora_trainable(True)
    model.eval()
    inputs = torch.tensor([[1, 8, 2]])
    with torch.no_grad():
        expected = model(inputs, use_cache=False).logits
    save_adaptation(model, tokenizer, base, tmp_path)
    restored, _ = load_adaptation(tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(inputs, use_cache=False).logits, expected, atol=1e-5, rtol=1e-5
        )


def test_output_is_base_plus_offset_with_one_surface_softmax():
    model, _, _ = tiny_adaptation(LlamaConfig)
    base_logits = torch.tensor([[0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, -18.0]])
    type_logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 3.0]])
    scores = model.logits_mapper(base_logits, type_logits)
    assert scores[0, 1] == 2
    assert scores[0, 8] == 5
    torch.testing.assert_close(
        scores.softmax(-1)[0, 8] / scores.softmax(-1)[0, 1], torch.exp(torch.tensor(3.0))
    )


def test_initialization_excludes_oov_and_multiple_transformations():
    inputs = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 6.0], [9.0, 9.0]])
    outputs = inputs * 2
    decomp = {2: (1, [0, 1]), 3: (1, [1, 2]), 4: (1, [1])}
    in_offsets, out_offsets = mean_offsets(decomp, inputs, outputs, 3, [0])
    torch.testing.assert_close(in_offsets, torch.tensor([[0.0, 0.0], [2.0, 4.0], [0.0, 0.0]]))
    torch.testing.assert_close(out_offsets, 2 * in_offsets)


def test_distillation_all_oov_or_ignored_labels_has_finite_zero_loss():
    model, _, _ = tiny_adaptation(LlamaConfig)
    model.train()
    inputs = torch.tensor([[8, 8, 8]])
    output = model(inputs, labels=inputs, use_cache=False)
    assert output.loss == 0
    output.loss.backward()
    assert torch.isfinite(model.types_lm_head.weight.grad).all()


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pretraining"))
from modeling_gpt import GPT, GPTForCausalLM, convert_to_hf_model


def test_pretraining_export_keeps_real_vocab_and_softcap(tmp_path):
    model = GPT(9, 1, 1, 32, 64, 128, use_linear_cross_entropy=False, eos_token_id=8)
    with torch.no_grad():
        model.lm_head.weight[9:].fill_(1000)
        model.lm_head.weight[:9].normal_()
    hf = convert_to_hf_model(model).eval()
    assert hf.config.valid_vocab_size == 9
    assert hf.config.eos_token_id == 8
    inputs = torch.tensor([[1, 2, 8]])
    with torch.no_grad():
        expected = hf(inputs, use_cache=False).logits
    assert expected.shape[-1] == 9
    assert expected.abs().max() <= 30
    hf.save_pretrained(tmp_path)
    restored = GPTForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(inputs, use_cache=False).logits, expected)


def test_document_mask_uses_selected_eos():
    model = GPT(9, 0, 1, 32, 64, 256, eos_token_id=8)
    inputs = torch.ones(256, dtype=torch.long)
    inputs[128] = 8
    long_mask, _ = model.create_blockmasks(inputs, torch.tensor(2))
    assert not long_mask.mask_mod(0, 0, 129, 1)
    assert long_mask.mask_mod(0, 0, 129, 128)


def test_pretraining_cached_and_uncached_attention_match():
    model = GPT(9, 2, 2, 32, 64, 128, use_linear_cross_entropy=False)
    # Nonzero projections make the test sensitive to attention layout and RoPE position.
    with torch.no_grad():
        for block in model.blocks:
            block.attn.c_proj.weight.normal_(std=0.03)
        model.lm_head.weight.normal_(std=0.03)
    hf = convert_to_hf_model(model).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full = hf(ids, use_cache=False).logits
        prefix = hf(ids[:, :3], use_cache=True)
        final = hf(ids[:, 3:], past_key_values=prefix.past_key_values, use_cache=True)
    torch.testing.assert_close(final.logits[:, 0], full[:, -1])
    assert hf.generate(
        ids,
        max_new_tokens=2,
        do_sample=False,
        eos_token_id=[],
        pad_token_id=0,
        attention_mask=torch.ones_like(ids),
    ).shape == (1, 6)


def test_compositional_pretraining_excludes_padded_predictions_and_losses():
    from modeling_compositional import CompositionalGPT, TypeConfig

    model = CompositionalGPT(
        9,
        10,
        TypeConfig({"prefix": 2, "capitalization": 2, "inflections": 3}),
        0,
        1,
        32,
        64,
        128,
        use_linear_cross_entropy=False,
        ignore_na_types=False,
        eos_token_id=0,
    )
    decomposition = {i: (i, [0, 2, 4]) for i in range(9)}
    decomposition[9] = (1, [0, 2, 5])
    model.set_token_mappings(
        list(range(9)), {i: v[0] for i, v in decomposition.items()}, list(range(10))
    )
    model.set_token_to_type_ids(decomposition)
    model.conditioned_transformation_head.set_decomposition_map(decomposition)
    with torch.no_grad():
        model.lm_head.weight[9:].fill_(1000.0)
        model.lm_head.weight[:9].normal_(std=0.03)
    # Exercise the scoring and joint loss without compiling the CUDA attention kernel.
    model.create_blockmasks = lambda *args: (None, None)
    model.eval()
    outputs = model(torch.tensor([1, 9, 2]), torch.tensor([9, 2, 3]), torch.tensor(1))
    assert outputs.base_logits.shape[-1] == 9
    assert torch.isfinite(outputs.loss)
    assert outputs.top1_extended_idx.max() < 10
    torch.testing.assert_close(
        outputs.loss, outputs.base_loss + outputs.type_loss * len(model.type_config.type_groups)
    )


def test_native_checkpoint_restores_compositional_mappings(tmp_path):
    from modeling_compositional import CompositionalGPT, TypeConfig
    from checkpoint_utils import model_metadata, restore_model

    model = CompositionalGPT(
        9,
        10,
        TypeConfig({"inflections": 3}),
        0,
        1,
        32,
        64,
        128,
        use_linear_cross_entropy=False,
        ignore_na_types=False,
    )
    decomposition = {i: (i, [0]) for i in range(9)}
    decomposition[9] = (1, [1])
    model.set_token_to_type_ids(decomposition)
    model.set_token_mappings(
        list(range(9)), {i: v[0] for i, v in decomposition.items()}, list(range(10))
    )
    payload = {"model_metadata": model_metadata(model), "model_state_dict": model.state_dict()}
    torch.save(payload, tmp_path / "checkpoint.pt")
    restored = restore_model(torch.load(tmp_path / "checkpoint.pt", weights_only=True))
    model.create_blockmasks = restored.create_blockmasks = lambda *args: (None, None)
    model.eval()
    inputs, targets = torch.tensor([1, 9, 2]), torch.tensor([9, 2, 3])
    with torch.no_grad():
        expected = model(inputs, targets, torch.tensor(1))
        actual = restored(inputs, targets, torch.tensor(1))
    torch.testing.assert_close(actual.base_logits, expected.base_logits)
    torch.testing.assert_close(actual.type_logits, expected.type_logits)
    torch.testing.assert_close(actual.loss, expected.loss)


def test_exported_attention_masks_padding_for_a_batch():
    model = GPT(9, 1, 2, 32, 64, 128, use_linear_cross_entropy=False, eos_token_id=8)
    with torch.no_grad():
        model.blocks[0].attn.c_proj.weight.normal_(std=0.03)
        model.lm_head.weight.normal_(std=0.03)
    hf = convert_to_hf_model(model).eval()
    with torch.no_grad():
        single = hf(torch.tensor([[1, 2]]), use_cache=False).logits
        batch = hf(
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
            use_cache=False,
        ).logits
    assert torch.isfinite(batch).all()
    torch.testing.assert_close(batch[0, 1:], single[0])
