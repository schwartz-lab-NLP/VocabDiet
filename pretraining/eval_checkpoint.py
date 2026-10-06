#!/usr/bin/env python3

import os
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer

from modeling_gpt import GPT, convert_to_hf_model
from checkpoint_utils import restore_model
from eval_utils import evaluate_downstream


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to checkpoint file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(**checkpoint["args"])

    if checkpoint.get("model_metadata", {}).get("model_type") == "compositional":
        raise ValueError(
            "This downstream evaluator expects a baseline GPT checkpoint. Use checkpoint_utils.restore_model for native compositional scoring."
        )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if "model_metadata" in checkpoint:
        return restore_model(checkpoint), checkpoint["step"], tokenizer
    model = GPT(
        vocab_size=len(tokenizer),
        eos_token_id=tokenizer.eos_token_id,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        model_dim=args.model_dim,
        intermediate_dim=args.intermediate_dim,
        num_kv_heads=args.num_kv_heads,
        max_seq_len=max(args.train_seq_len, args.val_seq_len),
        use_gated_proj=True,
        use_rms_norm=False,
        use_gqa=True,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        rope_scaling_type="linear",
        reorder_norms=args.reorder_norms,
        init_strategy=args.init_strategy,
        init_std=args.init_std,
        base_dim=args.base_dim,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint["step"], tokenizer


def main():
    args = parse_args()

    print(f"Loading checkpoint from: {args.checkpoint_path}")
    model, step, hf_tokenizer = load_model_from_checkpoint(args.checkpoint_path)

    print(f"Converting to HF model (step {step})")
    hf_model = convert_to_hf_model(model)

    print("Running downstream evaluation")
    downstream_metrics = evaluate_downstream(hf_model, hf_tokenizer)

    # Determine output directory
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint_path)
        checkpoint_name = os.path.basename(args.checkpoint_path).replace(".pt", "")
        args.output_dir = os.path.join(checkpoint_dir, f"eval_{checkpoint_name}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save results
    results_path = os.path.join(args.output_dir, "downstream_metrics.json")
    with open(results_path, "w") as f:
        json.dump(downstream_metrics, f, indent=4)

    print(f"Results saved to: {results_path}")

    # Print results
    for task_name, task_results in downstream_metrics.items():
        print(f"{task_name}: {task_results}")


if __name__ == "__main__":
    main()
