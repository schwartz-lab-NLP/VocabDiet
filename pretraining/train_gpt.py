import os
import sys
from pathlib import Path

code = Path(__file__).read_text()  # read the code of this file ASAP, for logging
import uuid
import time
import math
from dataclasses import dataclass
from functools import lru_cache
import json
import argparse

try:
    import wandb
except ImportError:  # Optional for local runs and command-line help.
    wandb = None
import copy
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# Keep module imports and `--help` usable on CPU-only hosts. CUDA setup is done
# after argument parsing in the training entry point.
from torch import Tensor, nn
import torch.distributed as dist

from modeling_gpt import GPT, convert_to_hf_model, next_multiple_of_n
from checkpoint_utils import unwrap_model, model_metadata, effective_args
from optimizer import DistAdam, Muon
from data_loader import distributed_data_generator
from eval_utils import compute_validation_tokenization_stats, compute_bytes_per_token_metrics
from logging_utils import LossLogger, RunTracker
from gpu_utils import get_combined_gpu_stats

rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
master_process = rank == 0  # this process will do logging, checkpointing etc.


BASE_EMBED_LR = 0.0006
BASE_LM_HEAD_LR = 0.002
BASE_HIDDEN_LR = 0.005
MODEL_SIZE_LR_MUL = 1.0

MUON_MOMENTUM = 0.95
ADAM_BETA1 = 0.8
ADAM_BETA2 = 0.95
MOMENTUM_WARMUP_GAP = 0.1
CLIP_GRADIENT = 1.0

TRAIN_ITERS_MODIFIER = 1
TRAIN_BATCH_MODIFIER = 1
VAL_BATCH_MODIFIER = 1


@dataclass
class Hyperparameters:
    # data
    dataset = "HuggingFaceFW/fineweb"  # HuggingFace dataset name
    dataset_config = "sample-10BT"  # Dataset configuration
    tokenizer = "openai-community/gpt2-large"  # Can be HuggingFace Hub name or local path
    data_dir = None  # Will be constructed from dataset/config/tokenizer if not provided
    val_tokens = 9830400
    train_seq_len = int(TRAIN_BATCH_MODIFIER * 48) * 1024
    val_seq_len = int(VAL_BATCH_MODIFIER * 48) * 1024
    # optimization
    num_iterations = int(TRAIN_ITERS_MODIFIER * 1750)
    weight_decay = 0.1
    cooldown_frac = 0.315
    lr_warmup_steps = -1
    momentum_warmup_steps = 300
    gradient_accumulation_steps = 1
    # optimizer hyperparameters
    base_embed_lr: float = BASE_EMBED_LR
    base_lm_head_lr: float = BASE_LM_HEAD_LR
    base_hidden_lr: float = BASE_HIDDEN_LR
    model_size_lr_mul: float = MODEL_SIZE_LR_MUL
    muon_momentum: float = MUON_MOMENTUM
    adam_beta1: float = ADAM_BETA1
    adam_beta2: float = ADAM_BETA2
    momentum_warmup_gap: float = MOMENTUM_WARMUP_GAP
    clip_gradient: float = CLIP_GRADIENT
    # evaluation and logging
    val_loss_every = 250
    save_checkpoint_every = 0  # 0 means no intermediate checkpoints
    save_final_checkpoint = True
    # resume training
    resume_from_checkpoint = None
    # lr decay
    lr_decay_type = "linear"  # "linear" or "cosine"
    # wandb
    wandb_project = None
    wandb_entity = None
    wandb_run_name = None
    throughput_window_frac = 0.67
    use_linear_cross_entropy = True
    benchmark_regular_ce_memory = False
    # Small
    num_layers = 12
    num_heads = 6
    model_dim = 768
    intermediate_dim = int(2 / 3 * 4 * 768)
    num_kv_heads = 2


def get_base_parser():
    """Returns the base argument parser that can be extended."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb')",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset configuration (e.g., 'sample-10BT')",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name or local path to trained tokenizer",
    )
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--int32_data", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--val_tokens", type=int, default=None)
    parser.add_argument("--train_seq_len", type=int, default=None)
    parser.add_argument("--val_seq_len", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--cooldown_frac", type=float, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--momentum_warmup_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--reorder_norms", action="store_true", default=True)
    parser.add_argument(
        "--init_strategy", type=str, choices=["default", "olmo2", "minicpm"], default="olmo2"
    )
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--base_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--intermediate_dim", type=int, default=None)
    parser.add_argument("--val_loss_every", type=int, default=None)
    parser.add_argument("--save_checkpoint_every", type=int, default=None)
    parser.add_argument("--save_final_checkpoint", action="store_true", default=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lr_decay_type", type=str, choices=["linear", "cosine"], default=None)
    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--throughput_window_frac",
        type=float,
        default=None,
        help="Fraction of trailing training iterations used for train_tokens_per_sec (0 < frac <= 1).",
    )
    parser.add_argument(
        "--use_linear_cross_entropy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use cut/linear cross entropy loss (default: true).",
    )
    parser.set_defaults(benchmark_regular_ce_memory=False)
    parser.add_argument(
        "--disable_local_logging",
        action="store_true",
        default=False,
        help="Disable local file logging (loss logs, logits stats, runs CSV)",
    )
    parser.add_argument("--dont_compile", action="store_true", default=False)

    return parser


def parse_args():
    parser = get_base_parser()
    parsed_args = parser.parse_args()
    args = Hyperparameters()
    for key, value in vars(parsed_args).items():
        if value is not None or not hasattr(args, key):
            setattr(args, key, value)
    if not (0.0 < args.throughput_window_frac <= 1.0):
        raise ValueError(
            f"throughput_window_frac must be in (0, 1], got {args.throughput_window_frac}"
        )
    if args.benchmark_regular_ce_memory and args.use_linear_cross_entropy:
        print("benchmark_regular_ce_memory enabled: forcing --no-use_linear_cross_entropy")
        args.use_linear_cross_entropy = False
    return args


def get_data_files(data_dir):
    """Construct train and val file patterns from data directory."""
    data_fn_prefix = "data"
    train_files = os.path.join(data_dir, f"{data_fn_prefix}_train_*.bin")
    val_files = os.path.join(data_dir, f"{data_fn_prefix}_val_*.bin")
    return train_files, val_files


def save_checkpoint(step, model, optimizers, args, run_id, current_lr):
    if not master_process:
        return

    checkpoint_dir = f"{args.output_dir}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save full checkpoint
    checkpoint = {
        "step": step,
        "model_state_dict": unwrap_model(model).state_dict(),
        "model_metadata": model_metadata(model),
        "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
        "args": effective_args(args),
        "run_id": str(run_id),
    }
    torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_step_{step:06d}.pt")

    # Save config and lr info
    with open(f"{checkpoint_dir}/config_step_{step:06d}.json", "w") as f:
        json.dump(effective_args(args), f, indent=2)

    with open(f"{checkpoint_dir}/lr_step_{step:06d}.txt", "w") as f:
        f.write(f"Step: {step}\nLearning Rate: {current_lr:.8f}\n")


def load_checkpoint(checkpoint_path, model, optimizers):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    for opt, opt_state in zip(optimizers, checkpoint["optimizer_state_dicts"]):
        opt.load_state_dict(opt_state)
    return checkpoint["step"]


def save_hf_model(model, run_id, output_dir):
    if not master_process:
        return
    hf_model = convert_to_hf_model(unwrap_model(model))
    checkpoint_dir = os.path.join(output_dir, f"{run_id}/hf")
    os.makedirs(checkpoint_dir, exist_ok=True)
    hf_model.save_pretrained(checkpoint_dir)
    return hf_model


def extract_dataset_name(dataset_path):
    """Extract dataset name from HuggingFace format (org/dataset) or simple name."""
    if "/" in dataset_path:
        # HuggingFace Hub format (e.g., "HuggingFaceFW/fineweb" -> "fineweb")
        name = dataset_path.split("/")[-1]
    else:
        # Simple name (e.g., "fineweb")
        name = dataset_path
    return name


def get_tokenizer_name(tokenizer_path):
    """Extract a short name from tokenizer path for run naming."""
    if "/" in tokenizer_path:
        # HuggingFace Hub format (e.g., "openai-community/gpt2-large")
        name = tokenizer_path.split("/")[-1]
    else:
        # Local path (e.g., "tokenizers/custom_bpe_10k")
        name = os.path.basename(tokenizer_path)
    return name


def get_run_name(args, model_type="gpt"):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dataset_name = extract_dataset_name(args.dataset)
    tokenizer_name = get_tokenizer_name(args.tokenizer)
    run_name = f"{dataset_name}_{tokenizer_name}_{model_type}_lr{MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR:.4f}_cooldown{args.cooldown_frac:.2f}{args.lr_decay_type}_bs{world_size * args.train_seq_len}_d{args.model_dim}_ff{args.intermediate_dim}_l{args.num_layers}_h{args.num_heads}"
    run_name = run_name + "_bf16" if args.bf16 else run_name
    run_name = (
        run_name + f"_accum{args.gradient_accumulation_steps}steps"
        if args.gradient_accumulation_steps > 1
        else run_name
    )
    run_name = run_name + "_torchce" if not args.use_linear_cross_entropy else run_name
    run_name = run_name + "_membench" if args.benchmark_regular_ce_memory else run_name
    return run_name


def get_wandb_run_name(args, model_type="gpt"):
    if args.wandb_run_name:
        return args.wandb_run_name

    return get_run_name(args, model_type)


def nvidia_smi():
    import subprocess  # avoid top level import

    return subprocess.run(
        ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ).stdout


if __name__ == "__main__":
    args = parse_args()
    if args.wandb_project and wandb is None:
        raise RuntimeError(
            "Weights & Biases logging was requested, but wandb is not installed. Install wandb or pass --wandb_project ''."
        )

    # Construct data_dir if not provided
    if args.data_dir is None:
        dataset_name = extract_dataset_name(args.dataset)
        tokenizer_name = get_tokenizer_name(args.tokenizer)
        args.data_dir = f"data/{dataset_name}_{args.dataset_config}/{tokenizer_name}"

    # Load tokenizer to get vocab size and for validation metrics
    from transformers import AutoTokenizer

    try:
        # Try loading from local path or HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer from '{args.tokenizer}': {e}")
        raise

    vocab_size = len(tokenizer)
    print(f"Loaded tokenizer '{args.tokenizer}' with vocab_size={vocab_size:,}")

    # Verify tokenizer has required special tokens
    if tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer does not have an eos_token_id. "
            f"eos_token: {tokenizer.eos_token}, "
            f"bos_token: {tokenizer.bos_token}, "
            f"Tokenizer: {args.tokenizer}"
        )
    args.eos_token_id = tokenizer.eos_token_id
    args.valid_vocab_size = vocab_size
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

    # Construct file paths from data directory
    train_files, val_files = get_data_files(args.data_dir)
    print(f"train_files: '{train_files}'")
    print(f"val_files: '{val_files}'")

    # torchrun sets these env variables
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        # Multi-GPU distributed setup
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
        distributed = True
    else:
        # Single GPU setup
        world_size = 1
        rank = 0
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        distributed = False

    # begin logging
    logfile = None
    run_id = None
    run_name = None
    if master_process:
        run_id = uuid.uuid4()
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)

    def print0(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)

    # begin by printing this file (the Python code)
    print0(code)
    print0("=" * 100)
    # log information about the hardware/software environment this is running on
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(nvidia_smi())
    print0("=" * 100)

    model: nn.Module = GPT(
        vocab_size=vocab_size,
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
        use_linear_cross_entropy=args.use_linear_cross_entropy,
    ).cuda()

    model.lm_head.bfloat16()
    for m in model.modules():
        if args.bf16:
            m.bfloat16()
        elif isinstance(m, nn.Embedding):
            m.bfloat16()
    if distributed:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
    total_parameter_count = sum(p.numel() for p in model.parameters())
    trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print0(
        f"Model parameter count (start): total={total_parameter_count:,} trainable={trainable_parameter_count:,}",
        console=True,
    )

    # Initialize wandb
    if master_process and args.wandb_project:
        # Extract model hyperparameters
        model_hparams = {
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "model_dim": model.model_dim,
            "intermediate_dim": model.intermediate_dim,
            "num_kv_heads": model.num_kv_heads,
            "max_seq_len": model.max_seq_len,
            "use_gated_proj": model.use_gated_proj,
            "use_rms_norm": model.use_rms_norm,
            "use_gqa": model.use_gqa,
            "use_rope_scaling": model.use_rope_scaling,
            "rope_scaling_factor": model.rope_scaling_factor,
            "rope_scaling_type": model.rope_scaling_type,
            "use_linear_cross_entropy": model.use_linear_cross_entropy,
        }

        # Merge args and model hyperparams into a single config
        wandb_config = vars(args).copy()
        wandb_config.update(model_hparams)
        # Add dataset info for easy tracking
        wandb_config["dataset_name"] = extract_dataset_name(args.dataset)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=get_wandb_run_name(args),
            config=wandb_config,
        )

    if master_process:
        run_name = get_run_name(args)
        run_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Initialize logging utilities
        if args.disable_local_logging:
            loss_logger = logits_stats_collector = None
        else:
            loss_logger = LossLogger(os.path.join(run_dir, "loss_logs"), master_process)
            logits_stats_collector = None
            run_tracker = RunTracker("runs_log.csv")
            run_tracker.log_run(args, "gpt", run_id, run_dir, logfile)
    else:
        loss_logger = logits_stats_collector = None

    # collect the parameters to optimize
    hidden_matrix_params = [
        p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n
    ]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    # init the optimizer(s)
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = DistAdam(
        scalar_params + embed_params,
        lr=args.model_size_lr_mul * args.base_embed_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
        weight_decay=args.weight_decay,
    )
    optimizer1b = DistAdam(
        head_params,
        lr=args.model_size_lr_mul * args.base_lm_head_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-10,
        weight_decay=args.weight_decay,
    )
    optimizer2 = Muon(
        hidden_matrix_params,
        lr=args.model_size_lr_mul * args.base_hidden_lr,
        momentum=args.muon_momentum,
        weight_decay=args.weight_decay,
    )

    optimizers = [optimizer1, optimizer1b, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from_checkpoint:
        print0(f"Resuming from checkpoint: {args.resume_from_checkpoint}", console=True)
        start_step = load_checkpoint(args.resume_from_checkpoint, model, optimizers)
        print0(f"Resumed from step {start_step}", console=True)

    # learning rate schedule: stable then decay
    def get_lr(step: int):
        x = step / args.num_iterations
        assert 0 <= x <= 1
        if args.lr_warmup_steps > 0 and (x < (args.lr_warmup_steps / args.num_iterations)):
            w = (1 - x) / (args.lr_warmup_steps / args.num_iterations)
            return w * 0.1 + (1 - w) * 1.0
        elif x < 1 - args.cooldown_frac:
            return 1.0
        else:
            # Decay phase
            decay_progress = (x - (1 - args.cooldown_frac)) / args.cooldown_frac
            if args.lr_decay_type == "cosine":
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * decay_progress))
            else:  # linear
                return 1.0 - 0.9 * decay_progress

    # attention window size schedule: linearly increase
    @lru_cache(1)
    def get_window_size_blocks_helper(window_size: int):
        return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )

    def get_window_size_blocks(step: int):
        x = step / args.num_iterations  # progress in training
        assert 0 <= x <= 1
        # Linearly increase the block-wise sliding window size over training 128 -> 1792
        # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
        window_size = next_multiple_of_n(1728 * x, n=128)
        return get_window_size_blocks_helper(window_size)

    if not args.dont_compile:
        model: nn.Module = torch.compile(model, dynamic=False)
    hf_model = None

    ########################################
    #            Warmup kernels            #
    ########################################

    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    warmup_steps = 10
    initial_state = dict(
        model=copy.deepcopy(model.state_dict()),
        optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers],
    )  # save the initial state
    train_loader = distributed_data_generator(
        train_files,
        world_size * args.train_seq_len,
        align_to_bos=True,
        dtype=torch.int32 if args.int32_data else torch.uint16,
        bos_token_id=tokenizer.eos_token_id,
    )
    for _ in range(warmup_steps):
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(1)).backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
        opt.load_state_dict(opt_state)
    del train_loader, initial_state

    ########################################
    #        Training and validation       #
    ########################################

    train_loader = distributed_data_generator(
        train_files,
        world_size * args.train_seq_len,
        align_to_bos=True,
        dtype=torch.int32 if args.int32_data else torch.uint16,
        bos_token_id=tokenizer.eos_token_id,
    )

    # Compute validation tokenization stats once (tokenizer already loaded earlier)
    avg_bytes_per_token = None
    token_fertility = None
    if args.val_loss_every > 0:
        # Compute total bytes and word counts in validation set once (distributed across all processes)
        if master_process:
            print0("Computing validation tokenization metrics...", console=True)
        val_batch_size = world_size * args.val_seq_len
        val_steps = args.val_tokens // val_batch_size
        val_loader_for_bytes = distributed_data_generator(
            val_files, val_batch_size, align_to_bos=False, bos_token_id=tokenizer.eos_token_id
        )
        local_bytes, local_words = compute_validation_tokenization_stats(
            tokenizer, val_loader_for_bytes, val_steps
        )
        del val_loader_for_bytes

        # Sum bytes and words across all processes
        total_val_bytes_tensor = torch.tensor(local_bytes, device="cuda", dtype=torch.float64)
        total_val_words_tensor = torch.tensor(local_words, device="cuda", dtype=torch.float64)
        if distributed:
            dist.all_reduce(total_val_bytes_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_val_words_tensor, op=dist.ReduceOp.SUM)
        total_val_bytes = total_val_bytes_tensor.item()
        total_val_words = total_val_words_tensor.item()
        avg_bytes_per_token = total_val_bytes / args.val_tokens
        token_fertility = (args.val_tokens / total_val_words) if total_val_words > 0 else 0.0

        if master_process:
            print0(
                f"Total validation bytes: {total_val_bytes}, avg bytes per token: {avg_bytes_per_token:.2f}, "
                f"token fertility: {token_fertility:.4f}",
                console=True,
            )

    training_time_ms = 0
    train_tokens_per_iteration = world_size * args.train_seq_len * args.gradient_accumulation_steps
    throughput_window_steps = max(
        1, int(math.ceil(args.num_iterations * args.throughput_window_frac))
    )
    throughput_window_start_step = max(start_step, args.num_iterations - throughput_window_steps)
    throughput_window_start_time_ms = None
    if master_process:
        print0(
            f"Throughput window: last {throughput_window_steps}/{args.num_iterations} iterations "
            f"(frac={args.throughput_window_frac:.2f}, start_step={throughput_window_start_step})",
            console=True,
        )
    memory_benchmark_active = bool(
        args.benchmark_regular_ce_memory and not args.use_linear_cross_entropy
    )
    memory_benchmark_summary = {
        "steps": 0,
        "max_peak_alloc_mb": 0.0,
        "max_peak_reserved_mb": 0.0,
        "sum_peak_alloc_mb": 0.0,
        "sum_peak_reserved_mb": 0.0,
    }
    if memory_benchmark_active and master_process:
        print0("GPU memory benchmark enabled (regular torch cross entropy)", console=True)
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    train_steps = args.num_iterations
    for step in range(start_step, train_steps + 1):
        last_step = step == train_steps

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0 and step > 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            val_loader = distributed_data_generator(
                val_files, val_batch_size, align_to_bos=False, bos_token_id=tokenizer.eos_token_id
            )
            val_loss = 0
            val_acc = 0
            logits_for_stats = None
            with torch.no_grad():
                for i in range(val_steps):
                    inputs, targets = next(val_loader)
                    loss = model(inputs, targets, get_window_size_blocks(step))
                    val_loss += loss

                    # Compute top-1 accuracy
                    logits = model(inputs, None, get_window_size_blocks(step))
                    preds = logits.argmax(dim=-1)
                    correct = (preds == targets).float()
                    val_acc += correct.mean().item()

                    # Collect logits for statistics on the first validation batch
                    if i == 0 and master_process and logits_stats_collector:
                        logits_for_stats = {"logits": logits.detach()}
            val_loss /= val_steps
            val_acc /= val_steps
            del val_loader
            if distributed:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
                val_acc_tensor = torch.tensor(val_acc, device=val_loss.device)
                dist.all_reduce(val_acc_tensor, op=dist.ReduceOp.AVG)
                val_acc = val_acc_tensor.item()

            current_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
            train_steps_completed = max(step - start_step, 0)
            train_tokens_seen = train_steps_completed * train_tokens_per_iteration
            global_steps_completed = start_step + train_steps_completed
            if (
                throughput_window_start_time_ms is None
                and global_steps_completed >= throughput_window_start_step
            ):
                throughput_window_start_time_ms = training_time_ms
                if global_steps_completed > throughput_window_start_step:
                    throughput_window_start_step = global_steps_completed
            throughput_window_steps_completed = max(
                0, global_steps_completed - throughput_window_start_step
            )
            throughput_window_time_ms = (
                0.0
                if throughput_window_start_time_ms is None
                else max(0.0, training_time_ms - throughput_window_start_time_ms)
            )
            train_tokens_per_sec = 0.0
            if throughput_window_steps_completed > 0 and throughput_window_time_ms > 0:
                train_tokens_per_sec = (
                    throughput_window_steps_completed * train_tokens_per_iteration
                ) / (throughput_window_time_ms / 1000.0)

            byte_metrics = None
            # Prepare validation log message
            val_msg = f"step:{step}/{train_steps} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}"
            if avg_bytes_per_token is not None:
                byte_metrics = compute_bytes_per_token_metrics(val_loss, avg_bytes_per_token)
                val_msg += f" bpb:{byte_metrics['bpb']:.4f}"
            val_msg += f" train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step - start_step, 1):.2f}ms tok_s:{train_tokens_per_sec:.1f} lr:{current_lr:.6f}"

            print0(val_msg, console=True)

            if master_process:
                # Log validation loss to file
                if loss_logger:
                    additional_losses = {
                        "val_accuracy": val_acc,
                        "train_tokens_seen": train_tokens_seen,
                        "train_tokens_per_sec": train_tokens_per_sec,
                    }
                    if byte_metrics is not None:
                        additional_losses["bpb"] = byte_metrics["bpb"]
                        additional_losses["bpb_baseline"] = byte_metrics["bpb"]
                    if avg_bytes_per_token is not None:
                        additional_losses["avg_bytes_per_token"] = avg_bytes_per_token
                    if token_fertility is not None:
                        additional_losses["token_fertility"] = token_fertility
                    loss_logger.log_val_loss(step, val_loss, training_time_ms, additional_losses)

                # Collect and log logits statistics
                logits_wandb_logs = {}
                if logits_stats_collector and logits_for_stats:
                    logits_stats = logits_stats_collector.collect_stats(logits_for_stats, step)
                    logits_wandb_logs = logits_stats_collector.get_wandb_logs(logits_stats)

                # Log to wandb
                if args.wandb_project:
                    wandb_logs = {
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "lr": current_lr,
                        "step": step,
                        "train_tokens_seen": train_tokens_seen,
                        "train_tokens_per_sec": train_tokens_per_sec,
                    }
                    wandb_logs.update(logits_wandb_logs)

                    # Add bytes-per-token metrics if available
                    if byte_metrics is not None:
                        wandb_logs.update(byte_metrics)
                        wandb_logs["val_bpb"] = byte_metrics["bpb"]
                        wandb_logs["val_bpb_baseline"] = byte_metrics["bpb"]
                    if avg_bytes_per_token is not None:
                        wandb_logs["avg_bytes_per_token"] = avg_bytes_per_token
                    if token_fertility is not None:
                        wandb_logs["token_fertility"] = token_fertility

                    # Add GPU utilization metrics
                    gpu_stats = get_combined_gpu_stats()
                    wandb_logs.update(gpu_stats)

                    wandb.log(wandb_logs)

            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Save intermediate checkpoints
        if (
            args.save_checkpoint_every > 0
            and step > 0
            and step % args.save_checkpoint_every == 0
            and not last_step
        ):
            current_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
            save_checkpoint(step, model, optimizers, args, run_name, current_lr)

        if last_step:
            if args.save_final_checkpoint:
                current_lr = get_lr(step) * MODEL_SIZE_LR_MUL * BASE_HIDDEN_LR
                save_checkpoint(step, model, optimizers, args, run_name, current_lr)
                hf_model = save_hf_model(model, run_name, args.output_dir)
                tokenizer.save_pretrained(os.path.join(args.output_dir, run_name, "hf"))
            break

        # --------------- TRAINING SECTION -----------------
        train_loss = 0
        if memory_benchmark_active:
            torch.cuda.reset_peak_memory_stats()
        for micro_step in range(args.gradient_accumulation_steps):
            inputs, targets = next(train_loader)
            with torch.amp.autocast("cuda", enabled=False):  # Keep bfloat16
                loss = model(inputs, targets, get_window_size_blocks(step))
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()

        # set optimization hyperparameters
        current_lr = get_lr(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * current_lr
        if args.momentum_warmup_steps > 0:
            frac = min(step / args.momentum_warmup_steps, 1)
            for group in optimizer2.param_groups:
                group["momentum"] = (1 - frac) * (
                    args.muon_momentum - args.momentum_warmup_gap
                ) + frac * args.muon_momentum
            for group in optimizer1.param_groups:
                group["betas"] = (
                    (1 - frac) * (args.adam_beta1 - args.momentum_warmup_gap)
                    + frac * args.adam_beta1,
                    args.adam_beta2,
                )
            for group in optimizer1b.param_groups:
                group["betas"] = (
                    (1 - frac) * (args.adam_beta1 - args.momentum_warmup_gap)
                    + frac * args.adam_beta1,
                    args.adam_beta2,
                )

        # gradient clipping
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)

        # step the optimizers
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)

        current_learning_rate = current_lr * args.model_size_lr_mul * args.base_hidden_lr
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        train_steps_completed = step + 1 - start_step
        train_tokens_seen = train_steps_completed * train_tokens_per_iteration
        global_steps_completed = start_step + train_steps_completed
        if (
            throughput_window_start_time_ms is None
            and global_steps_completed >= throughput_window_start_step
        ):
            throughput_window_start_time_ms = approx_training_time_ms
            if global_steps_completed > throughput_window_start_step:
                throughput_window_start_step = global_steps_completed
        throughput_window_steps_completed = max(
            0, global_steps_completed - throughput_window_start_step
        )
        throughput_window_time_ms = (
            0.0
            if throughput_window_start_time_ms is None
            else max(0.0, approx_training_time_ms - throughput_window_start_time_ms)
        )
        train_tokens_per_sec = 0.0
        if throughput_window_steps_completed > 0 and throughput_window_time_ms > 0:
            train_tokens_per_sec = (
                throughput_window_steps_completed * train_tokens_per_iteration
            ) / (throughput_window_time_ms / 1000.0)
        memory_metrics = {}
        if memory_benchmark_active:
            torch.cuda.synchronize()
            peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
            peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
            memory_metrics = {
                "gpu_peak_allocated_mb": peak_alloc_mb,
                "gpu_peak_reserved_mb": peak_reserved_mb,
                "gpu_end_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_end_reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            }
            memory_benchmark_summary["steps"] += 1
            memory_benchmark_summary["max_peak_alloc_mb"] = max(
                memory_benchmark_summary["max_peak_alloc_mb"], peak_alloc_mb
            )
            memory_benchmark_summary["max_peak_reserved_mb"] = max(
                memory_benchmark_summary["max_peak_reserved_mb"], peak_reserved_mb
            )
            memory_benchmark_summary["sum_peak_alloc_mb"] += peak_alloc_mb
            memory_benchmark_summary["sum_peak_reserved_mb"] += peak_reserved_mb

        if master_process:
            additional_losses = {
                "train_tokens_seen": train_tokens_seen,
                "train_tokens_per_sec": train_tokens_per_sec,
            }
            additional_losses.update(memory_metrics)
            # Log to file
            if loss_logger:
                loss_logger.log_train_loss(
                    step, train_loss, current_learning_rate, additional_losses
                )
            # Log to wandb
            if args.wandb_project:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "lr": current_learning_rate,
                        "step": step,
                        "train_tokens_seen": train_tokens_seen,
                        "train_tokens_per_sec": train_tokens_per_sec,
                        **memory_metrics,
                    }
                )

        print0(
            f"step:{step + 1}/{train_steps} train_loss:{train_loss:.4f} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / max(train_steps_completed, 1):.2f}ms tok_s:{train_tokens_per_sec:.1f}"
            + (
                f" mem_peak_alloc:{memory_metrics['gpu_peak_allocated_mb']:.1f}MB mem_peak_res:{memory_metrics['gpu_peak_reserved_mb']:.1f}MB"
                if memory_metrics
                else ""
            ),
            console=True,
        )

    if memory_benchmark_active and memory_benchmark_summary["steps"] > 0:
        steps = memory_benchmark_summary["steps"]
        avg_peak_alloc_mb = memory_benchmark_summary["sum_peak_alloc_mb"] / steps
        avg_peak_reserved_mb = memory_benchmark_summary["sum_peak_reserved_mb"] / steps
        print0(
            f"memory benchmark summary: steps={steps} "
            f"max_peak_alloc={memory_benchmark_summary['max_peak_alloc_mb']:.1f}MB "
            f"max_peak_reserved={memory_benchmark_summary['max_peak_reserved_mb']:.1f}MB "
            f"avg_peak_alloc={avg_peak_alloc_mb:.1f}MB "
            f"avg_peak_reserved={avg_peak_reserved_mb:.1f}MB",
            console=True,
        )
    else:
        print0(
            f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
            f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB",
            console=True,
        )

    if master_process and args.wandb_project:
        # Log final GPU utilization stats
        final_gpu_stats = get_combined_gpu_stats()
        if final_gpu_stats:
            print0(f"Final GPU stats: {final_gpu_stats}", console=True)
            wandb.log({"final_" + k: v for k, v in final_gpu_stats.items()})
        if memory_benchmark_active and memory_benchmark_summary["steps"] > 0:
            steps = memory_benchmark_summary["steps"]
            wandb.log(
                {
                    "final_memory_benchmark_steps": steps,
                    "final_memory_benchmark_max_peak_allocated_mb": memory_benchmark_summary[
                        "max_peak_alloc_mb"
                    ],
                    "final_memory_benchmark_max_peak_reserved_mb": memory_benchmark_summary[
                        "max_peak_reserved_mb"
                    ],
                    "final_memory_benchmark_avg_peak_allocated_mb": memory_benchmark_summary[
                        "sum_peak_alloc_mb"
                    ]
                    / steps,
                    "final_memory_benchmark_avg_peak_reserved_mb": memory_benchmark_summary[
                        "sum_peak_reserved_mb"
                    ]
                    / steps,
                }
            )
        wandb.finish()
    print0(
        f"Model parameter count (end): total={total_parameter_count:,} trainable={trainable_parameter_count:,}",
        console=True,
    )

    if distributed:
        dist.destroy_process_group()
