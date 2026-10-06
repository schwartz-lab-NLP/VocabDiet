"""Portable native checkpoints, including compositional vocabulary mappings."""

import numpy as np
import torch


def unwrap_model(model):
    while hasattr(model, "_orig_mod") or isinstance(
        model, torch.nn.parallel.DistributedDataParallel
    ):
        model = model._orig_mod if hasattr(model, "_orig_mod") else model.module
    return model


def primitive(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {primitive(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return type(value)(primitive(v) for v in value)
    return value


def effective_args(args):
    return {
        name: primitive(getattr(args, name))
        for name in dir(args)
        if not name.startswith("_") and not callable(getattr(args, name))
    }


def model_metadata(model):
    model = unwrap_model(model)
    constructor = model._checkpoint_constructor.copy()
    compositional = hasattr(model, "type_config")
    metadata = {"format_version": 1, "model_type": "compositional" if compositional else "baseline"}
    if compositional:
        config = constructor.pop("type_config")
        metadata["type_config"] = {
            "type_groups": config.type_groups,
            "loss_weights": config.loss_weights,
            "types_loss_indices_map": config.offsets,
            "transformation_names_to_int": {
                v: k for k, v in config.transformation_int_to_names.items()
            },
        }
        metadata["token_mappings"] = model._checkpoint_token_mappings
        metadata["decomposition"] = model._checkpoint_decomposition
    metadata["constructor"] = constructor
    return primitive(metadata)


def restore_model(checkpoint, *, device="cpu"):
    """Restore a versioned native checkpoint loaded with weights_only=True."""
    from modeling_gpt import GPT
    from modeling_compositional import CompositionalGPT, TypeConfig

    metadata = checkpoint["model_metadata"]
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported pretraining checkpoint format")
    constructor = metadata["constructor"].copy()
    if metadata["model_type"] == "compositional":
        constructor["type_config"] = TypeConfig(**metadata["type_config"])
        model = CompositionalGPT(**constructor)
        decomp = metadata["decomposition"]
        model.set_token_to_type_ids(decomp["map"], decomp["negative_type_ids"])
        model.set_token_mappings(**metadata["token_mappings"])
    elif metadata["model_type"] == "baseline":
        model = GPT(**constructor)
    else:
        raise ValueError("Unknown pretraining model type")
    weights = checkpoint["model_state_dict"]
    if model.scalars.shape != weights["scalars"].shape:
        model.scalars = torch.nn.Parameter(torch.empty_like(weights["scalars"]))
    # Preserve mixed parameter dtypes without breaking shared module references.
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to(dtype=weights[name].dtype)
    model.load_state_dict(weights)
    return model.to(device).eval()
