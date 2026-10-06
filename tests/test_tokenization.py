import torch
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from posthoc.tokenization import ScaffoldTokenizer


def scaffold_tokenizer():
    backend = Tokenizer(
        models.BPE(
            {"[UNK]": 0, "a": 1, "b": 2, "ab": 3, "abab": 4},
            [("a", "b"), ("ab", "ab")],
            unk_token="[UNK]",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]"
    )
    return ScaffoldTokenizer(tokenizer, tokenizer, {"abab": 4})


def test_scaffold_expansion_precedes_batch_padding_and_truncation():
    tokenizer = scaffold_tokenizer()
    encoded = tokenizer(["abab", "a"], padding=True, return_tensors="pt")
    torch.testing.assert_close(encoded.input_ids, torch.tensor([[3, 3], [1, 0]]))
    torch.testing.assert_close(encoded.attention_mask, torch.tensor([[1, 1], [1, 0]]))
    assert encoded.token_type_ids.shape == encoded.input_ids.shape
    assert tokenizer.encode("abab", truncation=True, max_length=1) == [3]
    assert tokenizer.encode("abab", return_tensors="pt").shape == (1, 2)


def test_shortest_scaffold_path_preserves_special_tokens():
    tokenizer = scaffold_tokenizer()
    assert tokenizer.scaffold_id_map[4] == [3, 3]
    tokenizer.special_token_ids.add(4)
    assert tokenizer._replace_scaffold_tokens([4, 1])[0] == [4, 1]
