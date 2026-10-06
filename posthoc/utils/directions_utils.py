import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import tempfile
import json
import re
import os
import math
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, pipeline

# for clustering category groups
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

pass
from sklearn.metrics import silhouette_score


def rms_norm(x, eps=1e-8):
    """
    Applies RMS (Root Mean Square) normalization to a 2D tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, features)
        eps (float): Small value added for numerical stability

    Returns:
        torch.Tensor: Normalized tensor of the same shape as input
    """
    # Ensure input is 2D
    assert x.dim() == 2, "Input tensor must be 2D (batch_size, features)"

    # Calculate RMS: sqrt(mean(x^2))
    rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + eps)

    # Normalize
    return rms
