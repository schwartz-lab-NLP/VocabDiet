import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import default_data_collator
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import re
from datasets import load_dataset, Dataset, DatasetDict
from itertools import chain
from collections import Counter

binary_online_metrics = {
    "AUPRC": "BinaryAUPRC",
    "precision": "BinaryPrecision",
}
multiclass_online_metrics = {
    "AUPRC": "MulticlassAUPRC",
    "precision": "MulticlassPrecision",
    "confusion_matrix": "MulticlassConfusionMatrix",
}


def _load_torcheval_metric(name):
    try:
        from torcheval import metrics

        return getattr(metrics, name)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Type-level evaluation requires the optional `torcheval` package."
        ) from exc


def _is_online_metric(metric_name):
    return any([metric_type in metric_name for metric_type in binary_online_metrics.keys()]) or any(
        [metric_type in metric_name for metric_type in multiclass_online_metrics.keys()]
    )


def _init_binary_online_metric(metric_name, **kwargs):
    for k, metric_module in binary_online_metrics.items():
        if k in metric_name:
            return _load_torcheval_metric(metric_module)(**kwargs)


def _init_online_metric(metric_name, **kwargs):
    for k, metric_module in multiclass_online_metrics.items():
        if k in metric_name:
            return _load_torcheval_metric(metric_module)(**kwargs)


def _choose_or_ignore_replaced_words_in_attention_mask(
    attention_mask,
    labels,
    base_token_ids=None,
    inflection_token_ids=None,
    replaced_token_seqs_by_len=None,
):
    background_mask, base_token_mask, target_mask, subsequent_mask = None, None, None, None

    # tokens that appear in their base form
    if base_token_ids is not None:
        ignore_mask = torch.isin(labels, base_token_ids)
        base_token_mask = attention_mask * ignore_mask.long()
        base_token_mask = base_token_mask[..., 1:].contiguous()

    # choose token_ids of new vocabulary words in labels
    background_mask = attention_mask.clone()
    if inflection_token_ids is not None:
        ignore_mask = torch.isin(labels, inflection_token_ids)
        background_mask = background_mask * (~ignore_mask).long()

    # Ignore multi-token sequences of that were replaced with a single token
    if replaced_token_seqs_by_len is not None:
        # Create a mask that will be updated where sequences match
        ignore_mask = attention_mask.clone()  # Clone the attention mask to modify it
        # Loop over sequences in skip_token_seqs
        for seq_len, seqs in replaced_token_seqs_by_len.items():
            # Create a sliding window of the same size as the skip_seq and check for matches
            for i in range(labels.size(1) - seq_len + 1):
                # Check if the sequence matches at position i
                window = labels[:, i : i + seq_len]
                curr_mask = torch.all(window.unsqueeze(1) == seqs.unsqueeze(0), dim=-1)
                if curr_mask.any():
                    # Zero out the ignore mask for the length of the sequence
                    ignore_mask[curr_mask.any(dim=-1), i : i + seq_len] = 0
        # Apply the ignore mask to the attention mask
        background_mask *= ignore_mask

    if inflection_token_ids is not None or replaced_token_seqs_by_len is not None:
        subsequent_mask = get_last_zero_in_every_seq_mask(background_mask)
        subsequent_mask = subsequent_mask[..., :-1].contiguous()
        target_mask = get_first_zero_in_every_seq_mask(background_mask)
        target_mask = target_mask[..., 1:].contiguous()

    background_mask = background_mask[..., 1:].contiguous()

    return background_mask, base_token_mask, target_mask, subsequent_mask


# TODO make clearer what these functions are for
def get_last_zero_in_every_seq_mask(tensor):
    # Find where consecutive zeros end
    zero_mask = tensor == 0
    diff = torch.diff(zero_mask.int(), dim=1)
    last_zero_mask = (
        torch.cat([diff, torch.ones(tensor.size(0), 1, dtype=diff.dtype).to(tensor.device)], dim=1)
        == -1
    )

    # Create the output
    output = 1 - tensor
    output[zero_mask & ~last_zero_mask] = 0
    return output


def get_first_zero_in_every_seq_mask(tensor):
    # Identify where consecutive zeros begin
    zero_mask = tensor == 0
    diff = torch.diff(
        zero_mask.int(),
        dim=1,
        prepend=torch.zeros(tensor.size(0), 1, dtype=torch.int).to(tensor.device),
    )
    first_zero_mask = diff == 1  # Marks the beginning of each sequence of zeros

    # Create the output
    output = 1 - tensor
    output[zero_mask & ~first_zero_mask] = 0
    return output


def count_tokens_in_dataset(dataset, tokenizer, text_column="text"):
    def tokenize_and_count(examples):
        return {"num_tokens": [len(tokenizer(ex).input_ids) for ex in examples[text_column]]}

    tokenized_dataset = dataset.map(
        tokenize_and_count, batched=True, remove_columns=dataset.column_names
    )

    total_tokens = sum(tokenized_dataset["num_tokens"])
    return total_tokens


def _add_start_token(batch, tokenizer):
    if hasattr(tokenizer, "orig_bos_token_id"):
        bos_token_id = tokenizer.orig_bos_token_id
    else:
        bos_token_id = tokenizer.bos_token_id
    bos_tokens_tensor = torch.tensor([[bos_token_id]] * batch["input_ids"].size(dim=0)).to(
        batch["input_ids"].device
    )
    batch["input_ids"] = torch.cat([bos_tokens_tensor, batch["input_ids"]], dim=1)
    batch["labels"] = torch.cat([bos_tokens_tensor, batch["labels"]], dim=1)
    batch["attention_mask"] = torch.cat(
        [
            torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(
                batch["attention_mask"].device
            ),
            batch["attention_mask"],
        ],
        dim=1,
    )
    if "original_input_ids" in batch:
        batch["original_input_ids"] = torch.cat(
            [bos_tokens_tensor, batch["original_input_ids"]], dim=1
        )
    if "type_ids" in batch:
        batch["type_ids"] = torch.cat(
            [
                torch.zeros(
                    (bos_tokens_tensor.shape[0], batch["type_ids"].shape[1], 1), dtype=torch.int64
                ).to(batch["type_ids"].device),
                batch["type_ids"],
            ],
            dim=-1,
        )
    return batch


def compute_type_metrics(
    model,
    type_logits,
    type_labels,
    types_attn_mask,
    pred_ids,
    attention_mask,
    type2name,
    num_probe_layers,
):
    """
    Compute type-specific metrics for a given set of type logits and labels.
    Handles both single-layer (tensor) and multi-layer (dict) cases.

    Args:
        model: The model instance with type probing logic.
        type_logits: Tensor (num_probe_layers == 1) or dict (num_probe_layers > 1) of type logits.
        type_labels: Tensor of type labels.
        attention_mask: Attention mask to filter valid positions.
        type2name: Mapping from type indices to names.
        num_probe_layers: Number of layers being probed.

    Returns:
        Dict of type metrics, potentially prefixed by layer names if num_probe_layers > 1.
    """
    results = {}
    attention_mask_bool = (attention_mask == 1) & (types_attn_mask == 1)

    # Helper function to process a single layer's type logits
    def process_type_logits(curr_type_logits, layer_prefix=""):
        split_type_logits = model.split_types_to_groups(curr_type_logits)
        split_type_labels = model.split_types_to_groups(type_labels)
        layer_results = {}

        for type_group in split_type_labels.keys():
            curr_type_labels = split_type_labels[type_group][attention_mask_bool]
            curr_type_labels = curr_type_labels.view(-1, curr_type_labels.shape[-1])
            curr_type_logits = split_type_logits[type_group][attention_mask_bool].view(
                -1, curr_type_labels.shape[-1]
            )

            if "multilabel" in type_group:
                curr_type_probs = F.sigmoid(curr_type_logits)
                for type_i in range(curr_type_probs.shape[-1]):
                    type_name = type2name[type_i]
                    type_i_probs = curr_type_probs[..., type_i].view(-1)
                    type_i_labels = curr_type_labels[..., type_i].view(-1)
                    if torch.any(type_i_labels > 0):
                        # layer_results[f"types_{type_group}_{type_name}_recall"] = (type_i_probs, type_i_labels)
                        layer_results[f"types_{type_group}_{type_name}_precision"] = (
                            type_i_probs,
                            type_i_labels,
                        )
                        layer_results[f"types_{type_group}_{type_name}_AUPRC"] = (
                            type_i_probs,
                            type_i_labels,
                        )
                    else:
                        # layer_results[f"types_{type_group}_{type_name}_recall"] = None
                        layer_results[f"types_{type_group}_{type_name}_precision"] = None
                        layer_results[f"types_{type_group}_{type_name}_AUPRC"] = None
            else:
                curr_type_probs = F.softmax(curr_type_logits, dim=-1)
                # layer_results[f"types_{type_group}_recall"] = (curr_type_probs, curr_type_labels.argmax(-1))
                layer_results[f"types_{type_group}_precision"] = (
                    curr_type_probs,
                    curr_type_labels.argmax(-1),
                )
                layer_results[f"types_{type_group}_AUPRC"] = (
                    curr_type_probs,
                    curr_type_labels.argmax(-1),
                )
                layer_results[f"types_{type_group}_confusion_matrix"] = (
                    curr_type_probs,
                    curr_type_labels.argmax(-1),
                )

        # Prefix results with layer name if applicable
        if layer_prefix:
            return {f"{layer_prefix}_{k}": v for k, v in layer_results.items()}
        return layer_results

    if num_probe_layers == 1:
        # Single-layer case: type_logits is a tensor
        results.update(process_type_logits(type_logits))
    else:
        # Multi-layer case: type_logits is a dict
        for layer_name, layer_type_logits in type_logits.items():
            layer_results = process_type_logits(layer_type_logits, layer_prefix=layer_name)
            results.update(layer_results)

    return results


def compute_metrics(
    model,
    logits,
    labels,
    original_logits,
    original_labels,
    attention_mask,
    type_logits=None,
    type_labels=None,
    types_attn_mask=None,
    type2name=None,
    output_type_metrics=False,
    top_ks=[10],
    debug=False,
):
    results = dict()

    attention_mask_bool = attention_mask == 1

    results["counts"] = attention_mask_bool.to(torch.float).sum().to(torch.int).item()

    results["perplexity"] = (
        torch.exp(
            (
                F.cross_entropy(logits.transpose(1, 2), labels, reduction="none") * attention_mask
            ).sum(1)
            / attention_mask.sum(1)
        )
        .mean()
        .detach()
        .to(torch.float)
        .cpu()
        .numpy()
    )
    top1 = logits.argmax(dim=-1)
    results["top1_acc"] = ((labels == top1)[attention_mask_bool]).detach().cpu().numpy()
    for top_k in top_ks:
        topk = logits.topk(top_k, dim=-1).indices
        results[f"top{top_k}_acc"] = (
            ((topk == labels.unsqueeze(-1)).any(dim=-1)[attention_mask_bool]).detach().cpu().numpy()
        )

    if original_labels is not None and original_logits is not None:
        results["original_perplexity"] = (
            torch.exp(
                (
                    F.cross_entropy(
                        original_logits.transpose(1, 2), original_labels, reduction="none"
                    )
                    * attention_mask
                ).sum(1)
                / attention_mask.sum(1)
            )
            .mean()
            .detach()
            .to(torch.float)
            .cpu()
            .numpy()
        )
        original_top1 = original_logits.argmax(dim=-1)
        results["original_top1_acc"] = (
            ((original_labels == original_top1)[attention_mask_bool]).detach().cpu().numpy()
        )
        for top_k in top_ks:
            original_topk = original_logits.topk(top_k, dim=-1).indices
            results[f"original_top{top_k}_acc"] = (
                ((original_topk == original_labels.unsqueeze(-1)).any(dim=-1)[attention_mask_bool])
                .detach()
                .cpu()
                .numpy()
            )

    if type_logits is not None and output_type_metrics:
        if isinstance(type_logits, dict):
            num_probe_layers = len(type_logits)
        else:
            num_probe_layers = 1
        type_results = compute_type_metrics(
            model,
            type_logits,
            type_labels,
            types_attn_mask,
            top1,
            attention_mask,
            type2name,
            num_probe_layers,
        )
        results.update(type_results)

    return results


def eval_next_word_prediction(
    model,
    tokenizer,
    lm_dataset,
    accelerator=None,
    ignored_logits_idx=None,
    ignored_logits_value=-10,
    token_ids_to_eval=None,
    type2name=None,
    map_labels_to_base_token=False,
    map_labels_to_orig_first_token=False,
    base_token_ids=None,
    inflection_token_ids=None,
    inflection_replaced_sequences=None,
    compute_inflecion_base_token_metrics=True,
    compute_inflection_1st_token_metrics=True,
    compute_inflection_subsequent_token_metrics=True,
    compute_background_token_metrics=True,
    batch_size: int = 4,
    max_length: int = 256,
    drop_last: bool = True,
    max_samples: int = None,
    shuffle_samples: bool = False,
    eval_type_probes: bool = False,
    use_lm_preds_for_type_labels: bool = True,
    use_original_logits_as_target: bool = True,
    reduction="mean",
):
    if accelerator is None:
        accelerator = Accelerator()
    model.eval()
    if tokenizer.bos_token is not None and max_length:
        add_start_token = True
    else:
        add_start_token = False

    data_collator = default_data_collator

    if max_samples:
        eval_idx = range(0, min(max_samples, len(lm_dataset)))
        if shuffle_samples:
            eval_idx = np.random.choice(len(lm_dataset), min(max_samples, len(lm_dataset)))
        lm_dataset = lm_dataset.select(eval_idx)

    # Create data loaders
    eval_dataloader = DataLoader(
        lm_dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=False,
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    model.eval()

    # set up evaluation masks
    if token_ids_to_eval is not None:
        if not isinstance(token_ids_to_eval, torch.Tensor):
            token_ids_to_eval = (
                torch.Tensor(list(token_ids_to_eval)).to(torch.long).to(model.device)
            )

    if base_token_ids is not None:
        if not isinstance(base_token_ids, torch.Tensor):
            base_token_ids = torch.Tensor(list(base_token_ids)).to(torch.long).to(model.device)
    if inflection_token_ids is not None:
        if not isinstance(inflection_token_ids, torch.Tensor):
            inflection_token_ids = (
                torch.Tensor(list(inflection_token_ids)).to(torch.long).to(model.device)
            )
    if inflection_replaced_sequences is not None:
        if not isinstance(inflection_replaced_sequences[0], torch.Tensor):
            inflection_replaced_sequences = [
                torch.Tensor(list(seq)).to(torch.long).to(model.device)
                for seq in inflection_replaced_sequences
                if len(seq) > 1
            ]
        inflection_replaced_seqs_by_length = defaultdict(list)
        for seq in inflection_replaced_sequences:
            inflection_replaced_seqs_by_length[len(seq)].append(seq)
        for seq_len, seqs in inflection_replaced_seqs_by_length.items():
            inflection_replaced_seqs_by_length[seq_len] = torch.stack(seqs)
    else:
        inflection_replaced_seqs_by_length = None

    metrics = defaultdict(list)
    # run eval and compute metrics
    for batch_i, batch in tqdm(
        enumerate(eval_dataloader), total=len(eval_dataloader), miniters=10, desc="Evaluating..."
    ):
        if add_start_token:
            batch = _add_start_token(batch, tokenizer)
        labels = batch["input_ids"]
        input_ids = batch["input_ids"]
        if "labels" in batch:
            batch.pop("labels")
        attn_mask = batch["attention_mask"]

        eval_types = eval_type_probes or "type_ids" in batch
        if eval_types:
            if hasattr(model, "token_to_type_ids"):
                type_labels = model.token_to_type_ids(input_ids)
            else:
                type_labels = batch["type_ids"] if "type_ids" in batch else None
            if "type_labels" in batch:
                batch.pop("type_labels")

        with torch.no_grad():
            outputs = model(**batch)
        out_logits = outputs.logits
        shift_logits = out_logits[..., :-1, :].contiguous()
        if ignored_logits_idx is not None:
            shift_logits[..., ignored_logits_idx] = ignored_logits_value

        if use_original_logits_as_target:
            labels = outputs.original_logits.argmax(-1)
        if map_labels_to_base_token:
            with torch.no_grad():
                labels = model.map_input_ids_to_base_ids(labels)
        elif map_labels_to_orig_first_token:
            with torch.no_grad():
                labels = model.map_input_ids_to_orig_first_ids(labels)
        if use_original_logits_as_target:
            shift_labels = labels[..., :-1].contiguous()
        else:
            shift_labels = labels[..., 1:].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        # set up type labels and logits
        shift_type_logits = None
        shift_type_labels = None
        types_attn_mask = None
        if eval_types:
            if hasattr(outputs, "all_type_logits"):
                out_type_logits = outputs.all_type_logits
            else:
                out_type_logits = outputs.type_logits
            if type_labels is not None:
                with torch.no_grad():
                    if use_lm_preds_for_type_labels:
                        if hasattr(outputs, "original_logits"):
                            logits_for_type_labels = outputs.original_logits
                        else:
                            logits_for_type_labels = out_logits
                        types_attn_mask = model.map_input_ids_to_is_base_or_inflection_token(
                            logits_for_type_labels.argmax(-1)[..., :-1]
                        )
                        shift_type_labels = model.token_to_type_ids(
                            logits_for_type_labels.argmax(-1)[..., :-1]
                        ).contiguous()  # .transpose(2, 1)
                    else:
                        types_attn_mask = model.map_input_ids_to_is_base_or_inflection_token(
                            shift_labels
                        )
                        shift_type_labels = type_labels[..., 1:, :].contiguous()  # .transpose(2, 1)

                    if not isinstance(out_type_logits, dict):
                        shift_type_logits = out_type_logits[..., :-1, :].contiguous()
                    else:
                        shift_type_logits = {
                            k: v[..., :-1, :].contiguous() for k, v in out_type_logits.items()
                        }
        eval_original_labels = hasattr(outputs, "original_logits")
        shift_original_labels = None
        shift_original_logits = None
        if eval_original_labels:
            shift_original_logits = outputs.original_logits[..., :-1, :].contiguous()
            if "original_input_ids" in batch:
                shift_original_labels = batch["original_input_ids"][..., 1:].contiguous()

        # compute metrics over all tokens
        results = compute_metrics(
            model,
            shift_logits,
            shift_labels,
            shift_original_logits,
            shift_original_labels,
            shift_attention_mask_batch,
            shift_type_logits,
            shift_type_labels,
            types_attn_mask,
            type2name,
        )
        for metric_name, metric_value in results.items():
            if metric_value is None:
                continue
            metrics[metric_name].append(metric_value)

        background_mask, base_token_mask, target_mask, subsequent_mask = (
            _choose_or_ignore_replaced_words_in_attention_mask(
                attn_mask,
                input_ids,
                base_token_ids=base_token_ids,
                inflection_token_ids=inflection_token_ids,
                replaced_token_seqs_by_len=inflection_replaced_seqs_by_length,
            )
        )
        if compute_inflecion_base_token_metrics and base_token_mask is not None:
            results = compute_metrics(
                model,
                shift_logits,
                shift_labels,
                shift_original_logits,
                shift_original_labels,
                base_token_mask,
                shift_type_logits,
                shift_type_labels,
                types_attn_mask,
                type2name,
            )

            for metric_name, metric_value in results.items():
                if metric_value is None:
                    continue
                metrics[f"base_tokens_{metric_name}"].append(metric_value)

        if compute_inflection_1st_token_metrics and target_mask is not None:
            results = compute_metrics(
                model,
                shift_logits,
                shift_labels,
                shift_original_logits,
                shift_original_labels,
                target_mask,
                shift_type_logits,
                shift_type_labels,
                types_attn_mask,
                type2name,
            )

            for metric_name, metric_value in results.items():
                if metric_value is None:
                    continue
                metrics[f"target_tokens_{metric_name}"].append(metric_value)

        if compute_inflection_subsequent_token_metrics and subsequent_mask is not None:
            results = compute_metrics(
                model,
                shift_logits,
                shift_labels,
                shift_original_logits,
                shift_original_labels,
                subsequent_mask,
                shift_type_logits,
                shift_type_labels,
                types_attn_mask,
                type2name,
            )

            for metric_name, metric_value in results.items():
                if metric_value is None:
                    continue
                metrics[f"subsequent_tokens_{metric_name}"].append(metric_value)

        if compute_background_token_metrics and background_mask is not None:
            results = compute_metrics(
                model,
                shift_logits,
                shift_labels,
                shift_original_logits,
                shift_original_labels,
                background_mask,
                shift_type_logits,
                shift_type_labels,
                types_attn_mask,
                type2name,
            )

            for metric_name, metric_value in results.items():
                if metric_value is None:
                    continue
                metrics[f"background_tokens_{metric_name}"].append(metric_value)

        type_group2ids = None
        if eval_types:
            type_group2ids = model.get_split_type_ids()

            def _handle_type_results(results, prefix="additive_"):
                for metric_name, metric_value in results.items():
                    if metric_value is None:
                        continue
                    elif _is_online_metric(metric_name):
                        input, target = metric_value
                        input = input.to(torch.float)
                        target = target.to(torch.long)
                        if f"{prefix}{metric_name}" not in metrics:
                            if "multilabel" in metric_name:
                                metrics[f"{prefix}{metric_name}"] = _init_binary_online_metric(
                                    metric_name
                                )
                            else:
                                if "confusion_matrix" in metric_name:
                                    metrics[f"{prefix}{metric_name}"] = _init_online_metric(
                                        metric_name,
                                        normalize=None,
                                        num_classes=input.shape[-1],
                                        device=input.device,
                                    )
                                else:
                                    metrics[f"{prefix}{metric_name}"] = _init_online_metric(
                                        metric_name, average=None, num_classes=input.shape[-1]
                                    )
                        metrics[f"{prefix}{metric_name}"].update(input, target)
                    else:
                        metrics[f"{prefix}{metric_name}"].append(metric_value)

            shift_additive_tokens_mask = torch.any(shift_type_labels > 0, -1).to(
                shift_attention_mask_batch.dtype
            )
            base_word_results = compute_metrics(
                model,
                shift_logits,
                shift_labels,
                shift_original_logits,
                shift_original_labels,
                shift_attention_mask_batch * shift_additive_tokens_mask,
                shift_type_logits,
                shift_type_labels,
                types_attn_mask,
                type2name,
                output_type_metrics=True,
            )
            _handle_type_results(base_word_results)

    def _concat_func(x):
        if isinstance(x, np.ndarray) and len(x.shape) > 1:
            x = np.concat(x)
        elif isinstance(x, (list, tuple)) and len(x) > 1:
            if isinstance(x[0], np.ndarray) and len(x[0].shape) == 0:
                x = np.array(x)
            else:
                try:
                    x = np.concat(x)
                except:
                    x = np.array(x)
        return x

    # apply reduction
    reduce_func = _concat_func
    if reduction == "mean":

        def reduce_func(x):
            return np.mean(_concat_func(x)).item()
    elif reduction == "weighted_mean":

        def reduce_func(x, counts):
            return np.average(_concat_func(x), weights=counts).item()

    final_metrics = dict()
    for metric_name, metric_value in metrics.items():
        if _is_online_metric(metric_name):
            result = metrics[metric_name].compute()
            if result.dim() == 0:
                final_metrics[metric_name] = metrics[metric_name].compute().item()
            elif result.dim() == 1:
                for k, type_ids in type_group2ids.items():
                    if k in metric_name:
                        type_names = [type2name[type_id] for type_id in type_ids]
                        for i, type_name in enumerate(type_names):
                            final_metrics[f"{metric_name}_({type_name})"] = result[i].item()
                        break
            elif result.dim() == 2:
                for k, type_ids in type_group2ids.items():
                    if k in metric_name:
                        type_names = [type2name[type_id] for type_id in type_ids]
                        final_metrics[metric_name] = pd.DataFrame(
                            result.detach().cpu().numpy(), columns=type_names, index=type_names
                        )
                        break
        else:
            curr_counts = None
            if "counts" in metric_name:
                final_metrics[metric_name] = np.sum(_concat_func(metric_value)).item()
                curr_counts = _concat_func(metric_value)
            if reduction == "weighted_mean":
                final_metrics[metric_name] = reduce_func(metric_value, curr_counts)
            else:
                final_metrics[metric_name] = reduce_func(metric_value)

    return final_metrics


def serialize_metrics(metrics):
    """Convert confusion matrices (DataFrames) in metrics to lists for JSON compatibility."""
    serialized = {}
    for exp_name, results in metrics.items():
        serialized[exp_name] = {
            k: (v.values.tolist() if isinstance(v, pd.DataFrame) else v) for k, v in results.items()
        }
    return serialized
