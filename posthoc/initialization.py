"""Initialize transformation vectors from clean in-vocabulary pairs."""

import torch


def mean_offsets(decomposition_map, input_vectors, output_vectors, n_types, identity_type_ids=()):
    """Average variant-minus-base offsets separately in the two vector spaces.

    Only pairs whose base and variant have original vocabulary rows and exactly
    one nonidentity transformation contribute. Identity offsets remain zero.
    """
    identity = set(identity_type_ids)
    inputs = input_vectors.new_zeros(n_types, input_vectors.shape[-1])
    outputs = output_vectors.new_zeros(n_types, output_vectors.shape[-1])
    counts = input_vectors.new_zeros(n_types)
    for variant, (base, transforms) in decomposition_map.items():
        active = set(transforms) - identity
        if len(active) != 1 or not (
            0 <= base < len(input_vectors) and 0 <= variant < len(input_vectors)
        ):
            continue
        if base >= len(output_vectors) or variant >= len(output_vectors):
            continue
        transform = next(iter(active))
        inputs[transform] += input_vectors[variant] - input_vectors[base]
        outputs[transform] += output_vectors[variant] - output_vectors[base]
        counts[transform] += 1
    denominator = counts.clamp_min(1).unsqueeze(-1)
    return inputs / denominator, outputs / denominator
