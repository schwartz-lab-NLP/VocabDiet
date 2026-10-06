"""Tokenize with the additive vocabulary while replacing removed scaffold tokens."""

from transformers import PreTrainedTokenizerFast, BatchEncoding


class ScaffoldTokenizer(PreTrainedTokenizerFast):
    """A fast tokenizer that expands removed BPE merge scaffolds into kept tokens.

    Intermediate BPE merges can still be needed to create longer kept tokens.
    A removed merge that survives tokenization is replaced by its shortest
    segmentation using kept vocabulary tokens. Special tokens stay intact.
    """

    def __init__(self, extended_tokenizer, base_tokenizer, scaffold_vocab=None, **kwargs):
        super().__init__(
            tokenizer_object=extended_tokenizer.backend_tokenizer,
            added_tokens_decoder=base_tokenizer.added_tokens_decoder.copy(),
            **kwargs,
        )
        self.__dict__.update(extended_tokenizer.__dict__)
        self.scaffold_vocab = scaffold_vocab or {}
        self.scaffold_id_map = {}
        self.scaffold_token_ids = set()
        self.special_token_ids = set(self.all_special_ids)
        self._build_scaffold_mappings()

    def _build_scaffold_mappings(self):
        vocab = self.get_vocab()
        self.scaffold_id_map.clear()
        self.scaffold_token_ids.clear()
        for text, token_id in self.scaffold_vocab.items():
            if text not in vocab or token_id in self.special_token_ids:
                continue
            # Keep one shortest path per prefix instead of enumerating all paths.
            paths = [None] * (len(text) + 1)
            paths[0] = []
            for end in range(1, len(text) + 1):
                for start in range(end):
                    part = text[start:end]
                    if paths[start] is None or part not in vocab:
                        continue
                    if part in self.scaffold_vocab and len(part) > 1:
                        continue
                    candidate = paths[start] + [vocab[part]]
                    if paths[end] is None or len(candidate) < len(paths[end]):
                        paths[end] = candidate
            self.scaffold_token_ids.add(token_id)
            self.scaffold_id_map[token_id] = paths[-1] or [token_id]

    def _replace_scaffold_tokens(self, token_ids, attention_mask=None):
        if token_ids and isinstance(token_ids[0], list):
            pairs = [
                self._replace_scaffold_tokens(
                    ids, attention_mask[i] if attention_mask is not None else None
                )
                for i, ids in enumerate(token_ids)
            ]
            return [ids for ids, _ in pairs], [
                mask for _, mask in pairs
            ] if attention_mask is not None else None
        ids, mask = [], []
        for i, token_id in enumerate(token_ids):
            replacement = (
                self.scaffold_id_map.get(token_id, [token_id])
                if token_id not in self.special_token_ids
                else [token_id]
            )
            ids.extend(replacement)
            mask.extend([attention_mask[i] if attention_mask is not None else 1] * len(replacement))
        return ids, mask

    def __call__(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        # Apply length limits and padding after expansion so every field aligns.
        encoding = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_tensors=None,
            **kwargs,
        )
        original = encoding["input_ids"]
        batched = bool(original and isinstance(original[0], list))
        rows = original if batched else [original]
        for key, values in list(encoding.items()):
            field_rows = values if batched else [values]
            expanded = []
            for ids, field in zip(rows, field_rows):
                if key == "input_ids":
                    row, _ = self._replace_scaffold_tokens(ids)
                elif len(field) == len(ids):
                    row = [
                        value
                        for token_id, value in zip(ids, field)
                        for _ in range(
                            len(self.scaffold_id_map.get(token_id, [token_id]))
                            if token_id not in self.special_token_ids
                            else 1
                        )
                    ]
                else:
                    row = field
                if truncation:
                    limit = max_length if max_length is not None else self.model_max_length
                    row = row[-limit:] if self.truncation_side == "left" else row[:limit]
                expanded.append(row)
            encoding[key] = expanded if batched else expanded[0]
        padded = self.pad(
            encoding, padding=padding, max_length=max_length if padding == "max_length" else None
        )
        return BatchEncoding(padded, tensor_type=return_tensors, prepend_batch_axis=not batched)

    def encode(
        self,
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        return self(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs,
        )["input_ids"]
