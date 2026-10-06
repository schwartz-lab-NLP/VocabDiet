from lm_eval.evaluator import simple_evaluate
from lm_eval import tasks
from lm_eval.models.huggingface import HFLM
import numpy as np
import torch
import os
import json
from collections import defaultdict
import math

GENERATIVE_TASKS = ["squadv2", "triviaqa", "xquad"]
SAVE_PREDS_TO_FILE_TASKS = GENERATIVE_TASKS

task_to_limited_samples = defaultdict(lambda: 100)

generation_tasks = []


def compute_validation_tokenization_stats(
    tokenizer, val_data_generator, val_steps, dual_stream_tokenizer=None
):
    """
    Compute total bytes and word counts in validation dataset by decoding tokens back to text.
    Only needs to be called once at the start of training.
    """
    total_bytes = 0
    total_words = 0

    for i in range(val_steps):
        inputs, targets = next(val_data_generator)
        input_modifiers = None
        target_modifiers = None

        if isinstance(inputs, (tuple, list)) and len(inputs) == 2 and torch.is_tensor(inputs[0]):
            inputs, input_modifiers = inputs
        if isinstance(targets, (tuple, list)) and len(targets) == 2 and torch.is_tensor(targets[0]):
            targets, target_modifiers = targets

        def _decode(token_tensor, modifier_tensor):
            if dual_stream_tokenizer is not None and modifier_tensor is not None:
                if hasattr(modifier_tensor, "dim") and modifier_tensor.dim() == 2:
                    return dual_stream_tokenizer.decode_with_modifiers(
                        token_tensor.cpu().tolist(),
                        modifier_tensor.cpu().numpy(),
                    )
            return tokenizer.decode(token_tensor.cpu().tolist(), skip_special_tokens=True)

        if inputs.dim() == 1:
            full_sequence = targets if targets is not None else inputs
            full_modifiers = target_modifiers if targets is not None else input_modifiers
            text = _decode(full_sequence, full_modifiers)
            total_bytes += len(text.encode("utf-8"))
            total_words += len(text.split())
        else:
            batch_size, seq_len = inputs.shape
            for batch_idx in range(batch_size):
                full_sequence = targets[batch_idx]
                full_modifiers = None
                if input_modifiers is not None and target_modifiers is not None:
                    if input_modifiers.dim() == 3 and target_modifiers.dim() == 3:
                        full_modifiers = torch.cat(
                            [input_modifiers[batch_idx], target_modifiers[batch_idx, -1:, :]],
                            dim=0,
                        )
                text = _decode(full_sequence, full_modifiers)
                total_bytes += len(text.encode("utf-8"))
                total_words += len(text.split())

    return total_bytes, total_words


def compute_bytes_per_token_metrics(avg_loss_per_token, avg_bytes_per_token):
    """
    Compute bytes-per-token metrics for fair tokenizer comparison.

    Args:
        avg_loss_per_token: Average validation loss per token
        avg_bytes_per_token: Average number of UTF-8 bytes per token

    Returns:
        dict: Dictionary with 'bpb' metric
    """
    if avg_bytes_per_token <= 0:
        raise ValueError(f"avg_bytes_per_token must be positive, got {avg_bytes_per_token!r}")

    # Bits per byte: loss_per_token / log(2) / bytes_per_token
    bpb = avg_loss_per_token / math.log(2) / avg_bytes_per_token

    return {"bpb": bpb}


def compute_factorized_bytes_per_token_metrics(
    base_loss_per_token, type_loss_per_group, num_groups, avg_bytes_per_token
):
    """Compute BPB from the joint NLL of a base and factorized type prediction."""
    if num_groups < 0:
        raise ValueError(f"num_groups must be nonnegative, got {num_groups!r}")
    joint_loss = base_loss_per_token
    if type_loss_per_group is not None:
        joint_loss += type_loss_per_group * num_groups
    metrics = compute_bytes_per_token_metrics(joint_loss, avg_bytes_per_token)
    return {"joint_nll": joint_loss, **metrics}


def evaluate_downstream(
    model,
    tokenizer,
    limited=True,
    task_names=None,
    output_path=None,
    examples_limit=None,
    chat_format=False,
):
    """
    Evaluate model on multiple benchmarks using lm-evaluation-harness.
    """
    curr_task_to_limited_samples = task_to_limited_samples
    if examples_limit is not None:
        curr_task_to_limited_samples = defaultdict(lambda: examples_limit)

    task_configs = {
        "copa": {
            "num_fewshot": 0,
        },
        "piqa": {
            "num_fewshot": 0,
            "gen_kwargs": "do_sample=False,temperature=0.0,top_p=1.0,max_new_tokens=32",
        },
        "lambada_standard": {
            "num_fewshot": 0,
        },
        "tinyHellaswag": {
            "num_fewshot": 0,
        },
        "tinyWinogrande": {
            "num_fewshot": 0,
        },
        "tinyMMLU": {
            "num_fewshot": 0,
        },
        "tinyArc": {
            "num_fewshot": 0,
            "gen_kwargs": "max_tokens=100",
        },
    }

    if task_names is not None:
        task_configs = {task_name: task_configs[task_name] for task_name in task_names}

    model.eval()
    adapted_model = HFLM(
        model,
        tokenizer=tokenizer,
        trust_remote_code=True,
    )

    # Evaluate the model on each task
    results = {}
    for task_name, config in task_configs.items():
        # Evaluate the model on the task
        if output_path:
            curr_output_path = os.path.join(output_path, task_name)
            os.makedirs(curr_output_path, exist_ok=True)
        with torch.no_grad():
            result = simple_evaluate(
                model=adapted_model,
                tasks=[task_name],
                limit=curr_task_to_limited_samples[task_name] if limited else None,
                batch_size=1,
                num_fewshot=config.get("num_fewshot", None),
                gen_kwargs=config.get("gen_kwargs", None),
                log_samples=True if output_path else False,
            )

        # Store base results
        if len(result["results"].keys()) > 1:
            results[task_name] = dict()
            for subtask_name in result["results"].keys():
                results[task_name][subtask_name] = result["results"][subtask_name]
        else:
            results[task_name] = result["results"][task_name]

    return results
