import os
import copy
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
import torch.distributed as dist


OLD_WORLD_SIZE = 8


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


NEW_WORLD_SIZE = get_world_size()

# -----------------------------------------------------------------------------
# Distributed data loader


def _load_data_shard(file: Path, dtype=torch.uint16):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    bytes_per_tok = 4 if dtype == torch.int32 else 2
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(
            num_tokens, dtype=dtype, pin_memory=torch.cuda.is_available()
        )  # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == bytes_per_tok * num_tokens, "number of tokens read does not match header"
    return tokens


def _load_2d_modifier_shard(file: Path, num_groups: int, dtype: Optional[torch.dtype] = None):
    """Load a 2D modifier shard where each token has num_groups modifier values.

    The file format is:
    - Header: 256 int32 values (magic number, version, num_tokens, num_groups, ...)
    - Data: num_tokens * num_groups integer values (flattened row-major)

    Args:
        file: Path to the modifier file.
        num_groups: Number of transformation groups per token.
        dtype: Optional torch dtype override. If None, inferred from header.

    Returns:
        Tensor of shape (num_tokens, num_groups).
    """
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the modifier .bin file"
    version = int(header[1])

    if version == 1:
        # Old format: 1D modifiers (single scalar per token)
        num_tokens = int(header[2])
        bytes_per_val = 2  # uint16 for old format
        with file.open("rb", buffering=0) as f:
            modifiers = torch.empty(
                num_tokens, dtype=torch.uint16, pin_memory=torch.cuda.is_available()
            )
            f.seek(256 * 4)
            nbytes = f.readinto(modifiers.numpy())
            assert nbytes == bytes_per_val * num_tokens, (
                "number of modifiers read does not match header"
            )
        # Return as 1D for backward compatibility
        return modifiers
    elif version == 2:
        # New format: 2D modifiers (num_groups per token)
        num_tokens = int(header[2])
        stored_num_groups = int(header[3])
        assert stored_num_groups == num_groups, (
            f"Expected {num_groups} groups but file has {stored_num_groups}"
        )

        bytes_per_val = int(header[4]) if int(header[4]) > 0 else 1
        dtype_code = int(header[5]) if int(header[5]) > 0 else bytes_per_val
        code_to_torch = {
            1: torch.uint8,
            2: torch.uint16,
            4: torch.uint32,
            8: torch.uint64,
        }
        inferred_dtype = code_to_torch.get(dtype_code)
        if inferred_dtype is None:
            inferred_dtype = code_to_torch.get(bytes_per_val)
        if inferred_dtype is None:
            raise ValueError(f"Unsupported modifier dtype code={dtype_code} bytes={bytes_per_val}")
        load_dtype = inferred_dtype if dtype is None else dtype
        if load_dtype not in code_to_torch.values():
            raise ValueError(f"Unsupported modifier dtype override: {load_dtype}")
        if torch.empty((), dtype=load_dtype).element_size() != bytes_per_val:
            raise ValueError(
                f"Modifier dtype size mismatch: header has {bytes_per_val} bytes, override dtype={load_dtype}"
            )
        with file.open("rb", buffering=0) as f:
            modifiers = torch.empty(
                num_tokens * num_groups, dtype=load_dtype, pin_memory=torch.cuda.is_available()
            )
            f.seek(256 * 4)
            nbytes = f.readinto(modifiers.numpy())
            assert nbytes == bytes_per_val * num_tokens * num_groups, (
                "number of modifiers read does not match header"
            )
        # Reshape to (num_tokens, num_groups)
        return modifiers.reshape(num_tokens, num_groups)
    else:
        raise ValueError(f"Unsupported modifier file version: {version}")


def _get_modifier_file_path(ids_file: Path) -> Path:
    """Get the corresponding modifier file path for an ids file."""
    # Replace _ids.bin with _modifiers.bin
    return ids_file.parent / ids_file.name.replace("_ids.bin", "_modifiers.bin")


def save_1d_token_shard(file_path: Path, tokens: np.ndarray):
    """Save a 1D token array to disk with header.

    Args:
        file_path: Path to save the token file.
        tokens: NumPy array of token IDs (1D, uint16).
    """
    assert tokens.ndim == 1, f"Expected 1D array, got shape {tokens.shape}"

    if tokens.dtype.kind not in ("i", "u") or (
        len(tokens) and (tokens.min() < 0 or tokens.max() > 65535)
    ):
        raise ValueError(
            "Token shard requires integer IDs in the uint16 range; choose int32 preprocessing for larger vocabularies"
        )
    num_tokens = len(tokens)

    # Create header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # Magic number
    header[1] = 1  # Version 1 for tokens
    header[2] = num_tokens

    # Save to file
    with open(file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


# find world_size starting indices, such that each begins with a BOS token and local_batches don't overlap
def find_batch_starts(
    tokens: Tensor, pos: int, local_batch_size: int, max_batch_span: int, bos_token_id: int = 50256
):
    if bos_token_id is None:
        raise ValueError(
            "bos_token_id cannot be None. Make sure your tokenizer has an eos_token_id set. "
            "Check that the tokenizer was initialized with proper special tokens."
        )
    boundary_mask = tokens[pos : pos + max_batch_span] == bos_token_id
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    starts = []
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item()
        if end - start >= local_batch_size:
            starts.append(start)  # append start once end pos is confirmed
            if len(starts) == get_world_size():
                return starts, end - pos
            start = end
    assert False  # increase max_batch_span if necessary


def distributed_data_generator(
    filename_pattern: str,
    batch_size: int,
    align_to_bos: bool,
    dtype="uint16",
    bos_token_id: int = 50256,
    bundle=None,
    num_modifier_groups: int = None,
):
    """Generate batches of training data.

    Args:
        filename_pattern: Glob pattern for data files (e.g., "data/data_train_*.bin" or "data/data_train_*_ids.bin")
        batch_size: Total batch size across all GPUs
        align_to_bos: Whether to align batches to BOS tokens
        dtype: Data type for token IDs ('uint16' or 'int32')
        bos_token_id: BOS token ID for alignment
        bundle: Optional CompositionalTokenizerBundle for dual-stream mode
        num_modifier_groups: Number of transformation groups for 2D modifiers (if None, uses bundle info or 1D format)

    Yields:
        (inputs, targets): Tensors of shape [local_batch_size] with token IDs
        OR ((inputs, input_modifiers), (targets, target_modifiers)): In dual-stream mode
           where modifiers are shape [local_batch_size] (1D) or [local_batch_size, num_groups] (2D)
    """
    rank = get_rank()
    world_size = get_world_size()
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]

    # Detect dual-stream mode
    is_dual_stream = False
    is_2d_modifiers = False

    if bundle and bundle.tokenization_mode == "dual_stream":
        # Check for modifier_map (old 1D format) or unified_modifier_array (new 2D format)
        if hasattr(bundle, "unified_modifier_array") and bundle.unified_modifier_array:
            is_dual_stream = True
            is_2d_modifiers = True
            if num_modifier_groups is None:
                num_modifier_groups = bundle.unified_modifier_array.num_groups
        elif bundle.modifier_map:
            is_dual_stream = True
            is_2d_modifiers = False

        # Check if modifier files exist for the first file
        if files and "_ids.bin" in str(files[0]):
            modifier_file = _get_modifier_file_path(files[0])
            if not modifier_file.exists():
                print(
                    f"Warning: Bundle is in dual_stream mode but no modifier file found at {modifier_file}"
                )
                print(f"  Falling back to single_id mode")
                is_dual_stream = False

    # Override with explicit num_modifier_groups if provided
    if num_modifier_groups is not None and num_modifier_groups > 1:
        is_2d_modifiers = True

    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)

    # Load initial shard
    current_file = next(file_iter)
    tokens = _load_data_shard(current_file, dtype)
    modifiers = None

    if is_dual_stream:
        modifier_file = _get_modifier_file_path(current_file)
        if modifier_file.exists():
            if is_2d_modifiers and num_modifier_groups:
                modifiers = _load_2d_modifier_shard(modifier_file, num_modifier_groups)
                if modifiers.dim() == 2:
                    assert len(tokens) == modifiers.shape[0], "Token and modifier counts must match"
                else:
                    # Fallback to 1D if file is old format
                    assert len(tokens) == len(modifiers), "Token and modifier counts must match"
                    is_2d_modifiers = False
            else:
                modifiers = _load_data_shard(modifier_file, torch.uint16)
                assert len(tokens) == len(modifiers), "Token and modifier counts must match"

    pos = 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size
    max_batch_span = max_batch_span * (
        OLD_WORLD_SIZE // NEW_WORLD_SIZE
    )  # HACK for changing world size

    while True:
        if pos + max_batch_span + 1 >= len(tokens):
            # Load next shard
            current_file = next(file_iter)
            tokens = _load_data_shard(current_file, dtype)

            if is_dual_stream:
                modifier_file = _get_modifier_file_path(current_file)
                if modifier_file.exists():
                    if is_2d_modifiers and num_modifier_groups:
                        modifiers = _load_2d_modifier_shard(modifier_file, num_modifier_groups)
                        if modifiers.dim() == 2:
                            assert len(tokens) == modifiers.shape[0], (
                                "Token and modifier counts must match"
                            )
                    else:
                        modifiers = _load_data_shard(modifier_file, torch.uint16)
                        assert len(tokens) == len(modifiers), "Token and modifier counts must match"
                else:
                    modifiers = None

            pos = 0

        if align_to_bos:
            batch_starts, batch_span = find_batch_starts(
                tokens, pos, local_batch_size, max_batch_span, bos_token_id
            )
            start_idx = batch_starts[rank]
        else:
            batch_span = batch_size
            start_idx = pos + rank * local_batch_size

        buf_tokens = tokens[start_idx:][: local_batch_size + 1]
        inputs = buf_tokens[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf_tokens[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)

        # Handle modifier_ids if in dual-stream mode
        if is_dual_stream and modifiers is not None:
            if is_2d_modifiers and modifiers.dim() == 2:
                # 2D modifiers: shape (num_tokens, num_groups)
                buf_modifiers = modifiers[start_idx:][: local_batch_size + 1]
                input_modifiers = buf_modifiers[:-1].to(
                    device="cuda", dtype=torch.int32, non_blocking=True
                )
                target_modifiers = buf_modifiers[1:].to(
                    device="cuda", dtype=torch.int64, non_blocking=True
                )
            else:
                # 1D modifiers: shape (num_tokens,)
                buf_modifiers = modifiers[start_idx:][: local_batch_size + 1]
                input_modifiers = buf_modifiers[:-1].to(
                    device="cuda", dtype=torch.int32, non_blocking=True
                )
                target_modifiers = buf_modifiers[1:].to(
                    device="cuda", dtype=torch.int64, non_blocking=True
                )
        else:
            input_modifiers = None
            target_modifiers = None

        pos += batch_span

        # Yield in format: ((inputs, input_modifiers), (targets, target_modifiers))
        if is_dual_stream:
            yield (inputs, input_modifiers), (targets, target_modifiers)
        else:
            yield inputs, targets
