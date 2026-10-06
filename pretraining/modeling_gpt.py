import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import RMSNorm
import torch.distributed as dist
import math

# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention

# Add linear cross-entropy
from cut_cross_entropy import linear_cross_entropy

# For HF wrapper
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from typing import Optional, Tuple, Union

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model
# NORM_EPSILON, norm, Rotary, CausalSelfAttention, MLP, Block, next_multiple_of_n, GPT


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


NORM_EPSILON = 1e-6


def norm(x: Tensor, norm_layer=None):
    if norm_layer is not None:
        return norm_layer(x)
    return F.rms_norm(x, (x.size(-1),))


class Rotary(nn.Module):
    def __init__(
        self,
        dim: int,
        max_seq_len: int,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        rope_scaling_type="linear",
    ):
        super().__init__()
        self.use_rope_scaling = use_rope_scaling
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type

        # Base frequency - can be adjusted for longer contexts
        base_freq = 1024 if not use_rope_scaling else 1024 * rope_scaling_factor

        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / base_freq) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)

        if use_rope_scaling and rope_scaling_type == "dynamic":
            # Dynamic NTK scaling
            scale = (rope_scaling_factor * max_seq_len / max_seq_len) ** (
                torch.arange(0, dim // 2, 2).float() / (dim // 2)
            )
            angular_freq = angular_freq / scale.repeat(2)
        elif use_rope_scaling and rope_scaling_type == "linear":
            # Linear scaling
            t = t / rope_scaling_factor

        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor, position_offset: int = 0):
        assert self.cos.size(0) >= position_offset + x_BTHD.size(-3)
        cos, sin = (
            self.cos[None, position_offset : position_offset + x_BTHD.size(-3), None, :],
            self.sin[None, position_offset : position_offset + x_BTHD.size(-3), None, :],
        )
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


# Attention module


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int = 128,
        layer_idx: int = None,
        use_gqa=False,
        num_kv_heads=None,
        use_rms_norm: bool = False,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        rope_scaling_type="linear",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.use_gqa = use_gqa

        if use_gqa:
            self.num_kv_heads = num_kv_heads if num_kv_heads is not None else max(1, num_heads // 2)
            assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
            self.num_kv_groups = num_heads // self.num_kv_heads
        else:
            self.num_kv_heads = num_heads
            self.num_kv_groups = 1

        q_dim = self.num_heads * head_dim
        kv_dim = self.num_kv_heads * head_dim

        std = 0.5 * (dim**-0.5)
        bound = (3**0.5) * std

        if use_gqa:
            # Separate Q, K, V projections for GQA
            self.q_proj = nn.Parameter(torch.empty(q_dim, dim).uniform_(-bound, bound))
            self.k_proj = nn.Parameter(torch.empty(kv_dim, dim).uniform_(-bound, bound))
            self.v_proj = nn.Parameter(torch.empty(kv_dim, dim).uniform_(-bound, bound))
        else:
            # Original merged QKV weights
            self.qkv_w = nn.Parameter(torch.empty(3, q_dim, dim).uniform_(-bound, bound))

        self.rotary = Rotary(
            head_dim, max_seq_len, use_rope_scaling, rope_scaling_factor, rope_scaling_type
        )
        self.c_proj = nn.Linear(q_dim, dim, bias=False)
        self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12

        # k, q norms
        self.k_norm = RMSNorm(self.head_dim, NORM_EPSILON) if use_rms_norm else None
        self.q_norm = RMSNorm(self.head_dim, NORM_EPSILON) if use_rms_norm else None

    def forward(
        self,
        x: Tensor,
        block_mask: BlockMask = None,
        past_key_value: Cache = None,
        use_cache: bool = False,
        attention_mask: Tensor = None,
    ):
        B, T = x.size(0), x.size(1)
        if block_mask is not None and B != 1:
            raise ValueError("FlexAttention training requires batch size one")

        if self.use_gqa:
            # Separate projections for GQA
            q = F.linear(x, self.q_proj.type_as(x)).view(B, T, self.num_heads, self.head_dim)
            k = F.linear(x, self.k_proj.type_as(x)).view(B, T, self.num_kv_heads, self.head_dim)
            v = F.linear(x, self.v_proj.type_as(x)).view(B, T, self.num_kv_heads, self.head_dim)
        else:
            # Original merged QKV
            q, k, v = (
                F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
                .view(B, T, 3 * self.num_heads, self.head_dim)
                .chunk(3, dim=-2)
            )

        q, k = norm(q, self.q_norm), norm(k, self.k_norm)
        past_length = (
            past_key_value.get_seq_length(self.layer_idx)
            if past_key_value is not None and use_cache
            else 0
        )
        q, k = self.rotary(q, past_length), self.rotary(k, past_length)

        # Handle cache
        if past_key_value is not None and use_cache:
            k, v = past_key_value.update(k.transpose(1, 2), v.transpose(1, 2), self.layer_idx)
            k, v = k.transpose(1, 2), v.transpose(1, 2)

        # Handle GQA by repeating k,v heads
        if self.use_gqa and self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=-2)
            v = v.repeat_interleave(self.num_kv_groups, dim=-2)

        if block_mask is not None and B == 1:
            y = flex_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
                scale=self.attn_scale,
            ).transpose(1, 2)
        else:
            # Standard attention for generation
            scale = self.attn_scale or (self.head_dim**-0.5)
            q_heads, k_heads, v_heads = (t.transpose(1, 2) for t in (q, k, v))
            scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
            T_k = k.size(-3)
            causal_mask = (
                torch.arange(T_k, device=x.device)[None, :]
                <= (torch.arange(T, device=x.device) + past_length)[:, None]
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            if attention_mask is not None:
                scores = scores.masked_fill(
                    ~attention_mask[:, None, None, :T_k].bool(), float("-inf")
                )
            attn_weights = F.softmax(scores, dim=-1).nan_to_num(0.0)
            y = torch.matmul(attn_weights, v_heads).transpose(1, 2)

        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, dim: int, hdim: int, use_gated_proj: bool = False):
        super().__init__()
        self.use_gated_proj = use_gated_proj

        hdim = next_multiple_of_n(hdim, n=128)  # Round to multiple of 128 for efficiency
        if use_gated_proj:
            # Modern architecture with gate projection (like Llama, Qwen, OLMo)
            self.gate_proj = nn.Linear(dim, hdim, bias=False)
            self.up_proj = nn.Linear(dim, hdim, bias=False)
            self.down_proj = nn.Linear(hdim, dim, bias=False)
            self.down_proj.weight.detach().zero_()  # zero init
        else:
            self.c_fc = nn.Linear(dim, hdim, bias=False)
            self.c_proj = nn.Linear(hdim, dim, bias=False)
            self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        if self.use_gated_proj:
            # SwiGLU activation: gate * silu(up)
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            x = F.silu(gate) * up  # SwiGLU
            x = self.down_proj(x)
        else:
            # Original squared ReLU
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        max_seq_len: int,
        layer_idx: int,
        use_gated_proj: bool = False,
        use_rms_norm: bool = False,
        use_gqa: bool = False,
        num_kv_heads: int = None,
        use_rope_scaling: bool = False,
        rope_scaling_factor: float = 1.0,
        rope_scaling_type: str = "linear",
        reorder_norms: bool = True,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.reorder_norms = reorder_norms

        # Create norm layers
        self.input_layernorm = RMSNorm(dim, NORM_EPSILON) if use_rms_norm else None
        self.post_attention_layernorm = RMSNorm(dim, NORM_EPSILON) if use_rms_norm else None

        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            max_seq_len,
            head_dim=128,
            layer_idx=layer_idx,
            use_gqa=use_gqa,
            num_kv_heads=num_kv_heads,
            use_rms_norm=use_rms_norm,
            use_rope_scaling=use_rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            rope_scaling_type=rope_scaling_type,
        )

        self.mlp = MLP(dim, intermediate_dim, use_gated_proj)

    def forward(
        self,
        x: Tensor,
        block_mask: BlockMask = None,
        past_key_value: Cache = None,
        use_cache: bool = False,
        attention_mask: Tensor = None,
    ):
        if self.reorder_norms:
            if self.attn is not None:
                x = x + norm(
                    self.attn(x, block_mask, past_key_value, use_cache, attention_mask),
                    self.input_layernorm,
                )
            x = x + norm(self.mlp(x), self.post_attention_layernorm)
        else:
            if self.attn is not None:
                normed_x = norm(x, self.input_layernorm)
                x = x + self.attn(normed_x, block_mask, past_key_value, use_cache, attention_mask)
            normed_x = norm(x, self.post_attention_layernorm)
            x = x + self.mlp(normed_x)
        return x


# -----------------------------------------------------------------------------
# The main model


def init_weights(module, strategy="default", init_std=0.02, model_dim=None, base_dim=None):
    if base_dim is None:
        base_dim = model_dim

    for name, param in module.named_parameters():
        if strategy == "olmo2":
            std = init_std
        elif strategy == "minicpm":
            if param.ndim >= 2:
                if model_dim is not None and base_dim is not None:
                    std = init_std / math.sqrt(model_dim / base_dim)
                else:
                    std = init_std
            else:
                std = 0.1
        else:  # torch default init
            continue

        with torch.no_grad():
            param.uniform_(-std, std)


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        model_dim: int,
        intermediate_dim: int,
        max_seq_len: int,
        use_gated_proj: bool = False,
        use_rms_norm: bool = False,
        use_gqa: bool = False,
        num_kv_heads: int = None,
        use_rope_scaling: bool = False,
        rope_scaling_factor: float = 1.0,
        rope_scaling_type: str = "linear",
        reorder_norms: bool = True,
        init_strategy: str = "default",
        init_std: float = 0.02,
        base_dim: int = None,
        use_linear_cross_entropy: bool = True,
        eos_token_id: int = 50256,
    ):
        super().__init__()
        self._checkpoint_constructor = {
            k: v for k, v in locals().items() if k not in {"self", "__class__"}
        }
        self.valid_vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.use_rms_norm = use_rms_norm
        self.num_layers = num_layers
        self.use_linear_cross_entropy = use_linear_cross_entropy

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.use_rope_scaling = use_rope_scaling
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type
        self.model_dim = model_dim
        self.intermediate_dim = intermediate_dim
        self.max_seq_len = max_seq_len
        self.use_gated_proj = use_gated_proj
        self.use_rms_norm = use_rms_norm
        self.use_gqa = use_gqa
        self.reorder_norms = reorder_norms
        self.embed = nn.Embedding(vocab_size, model_dim)

        # Embedding and last norm layer
        self.embedding_norm = None
        self.final_norm = None
        self.embedding_norm = RMSNorm(model_dim, NORM_EPSILON) if use_rms_norm else None
        self.final_norm = RMSNorm(model_dim, NORM_EPSILON) if use_rms_norm else None

        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    intermediate_dim,
                    num_heads,
                    max_seq_len,
                    i,
                    use_gated_proj,
                    use_rms_norm,
                    use_gqa,
                    num_kv_heads,
                    use_rope_scaling,
                    rope_scaling_factor,
                    rope_scaling_type,
                    reorder_norms,
                )
                for i in range(num_layers)
            ]
        )

        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight.detach().zero_()

        # Apply custom initialization
        if init_strategy in ["olmo2", "minicpm"]:
            init_weights(self, init_strategy, init_std, model_dim, base_dim)
        else:
            # Keep original initialization
            pass

        # padding for distributed training
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        pad = (-0) % world_size
        if pad == 0:
            pad = world_size  # Ensure we have at least some parameters
        self.scalars = nn.Parameter(torch.ones(pad))

        if self.final_norm is not None:
            for param in self.final_norm.parameters():
                param.lr_mul = 1.0
        for block in self.blocks:
            if hasattr(block, "input_layernorm") and block.input_layernorm is not None:
                for param in block.input_layernorm.parameters():
                    param.lr_mul = 1.0
            if (
                hasattr(block, "post_attention_layernorm")
                and block.post_attention_layernorm is not None
            ):
                for param in block.post_attention_layernorm.parameters():
                    param.lr_mul = 1.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == self.eos_token_id).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = (
                dense_blockmask.argsort(dim=-1, descending=False, stable=True)
                .flip(-1)
                .to(torch.int32)
            )
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=input_seq.device)
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)

        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(
                    partial_kv_num_blocks,
                    torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1),
                ),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        # Create alternating pattern of block masks
        block_masks = []
        for i in range(self.num_layers):
            if i % 4 == 0 or i % 4 == 3:  # positions 0,3,4,7,8,11...
                block_masks.append(long_bm)
            else:  # positions 1,2,5,6,9,10...
                block_masks.append(short_bm)

        x = norm(self.embed(input_seq)[None], self.embedding_norm)

        for i in range(self.num_layers):
            x = self.blocks[i](x, block_masks[i])

        x = norm(x, self.final_norm)

        # If target_seq is None, return logits for inference/stats collection
        if target_seq is None:
            logits = F.linear(x, self.lm_head.weight[: self.valid_vocab_size])
            # Apply tanh softcapping following Gemma 2 paper
            logits = 30.0 * torch.tanh(logits / 30.0)
            return logits

        if self.use_linear_cross_entropy:
            # # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
            # yuvalr: change to linear cross entropy, softcap 30, no shift
            loss = linear_cross_entropy(
                x.flatten(0, -2),
                self.lm_head.weight[: self.valid_vocab_size].bfloat16(),
                target_seq,
                softcap=30,
                reduction="mean",  # "sum" if self.training else "mean",
            )
        else:
            logits = F.linear(x, self.lm_head.weight[: self.valid_vocab_size])
            logits = 30.0 * torch.tanh(logits / 30.0)
            loss = F.cross_entropy(
                logits.flatten(0, -2).float(),
                target_seq,
                reduction="mean",
                ignore_index=-1,
            )

        return loss


# HF Wrapper implementation


class GPTConfig(PretrainedConfig):
    model_type = "custom_gpt"
    attribute_map = {
        "num_hidden_layers": "num_layers",
        "hidden_size": "model_dim",
        "num_attention_heads": "num_heads",
    }

    def __init__(
        self,
        vocab_size=50304,
        valid_vocab_size=None,
        num_layers=12,
        num_heads=12,
        model_dim=768,
        intermediate_dim=3072,
        max_seq_len=2048,
        use_gated_proj=False,
        use_rms_norm=False,
        use_gqa=False,
        num_kv_heads=None,
        use_rope_scaling=False,
        rope_scaling_factor=1.0,
        rope_scaling_type="linear",
        reorder_norms=True,
        init_strategy="default",
        init_std=0.02,
        base_dim=None,
        use_linear_cross_entropy=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.valid_vocab_size = valid_vocab_size if valid_vocab_size is not None else vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.intermediate_dim = intermediate_dim
        self.max_seq_len = max_seq_len
        self.use_gated_proj = use_gated_proj
        self.use_rms_norm = use_rms_norm
        self.use_gqa = use_gqa
        self.num_kv_heads = num_kv_heads
        self.use_rope_scaling = use_rope_scaling
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_scaling_type = rope_scaling_type
        self.reorder_norms = reorder_norms
        self.init_strategy = init_strategy
        self.init_std = init_std
        self.base_dim = base_dim
        self.use_linear_cross_entropy = use_linear_cross_entropy
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class GPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = GPTConfig

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.model_dim)
        self.embedding_norm = (
            RMSNorm(config.model_dim, NORM_EPSILON) if config.use_rms_norm else None
        )
        self.final_norm = RMSNorm(config.model_dim, NORM_EPSILON) if config.use_rms_norm else None

        self.blocks = nn.ModuleList(
            [
                Block(
                    config.model_dim,
                    config.intermediate_dim,
                    config.num_heads,
                    config.max_seq_len,
                    i,
                    config.use_gated_proj,
                    config.use_rms_norm,
                    config.use_gqa,
                    config.num_kv_heads,
                    config.use_rope_scaling,
                    config.rope_scaling_factor,
                    config.rope_scaling_type,
                    config.reorder_norms,
                )
                for i in range(config.num_layers)
            ]
        )

        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else True
        return_dict = return_dict if return_dict is not None else True

        if past_key_values is None:
            past_key_values = DynamicCache()

        x = norm(self.embed(input_ids), self.embedding_norm)

        for block in self.blocks:
            x = block(x, None, past_key_values, use_cache, attention_mask)

        x = norm(x, self.final_norm)
        logits = F.linear(x, self.lm_head.weight[: self.config.valid_vocab_size])
        logits = 30.0 * torch.tanh(logits / 30.0)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (logits,)
            if use_cache:
                output = output + (past_key_values,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            cached_length = past_key_values.get_seq_length()
            input_ids = (
                input_ids[:, cached_length:]
                if input_ids.size(1) > cached_length
                else input_ids[:, -1:]
            )

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        }


def convert_to_hf_model(torch_model: GPT, config_dict: dict = None) -> GPTForCausalLM:
    if config_dict is None:
        config_dict = {
            "vocab_size": torch_model.vocab_size,
            "valid_vocab_size": torch_model.valid_vocab_size,
            "eos_token_id": torch_model.eos_token_id,
            "num_layers": torch_model.num_layers,
            "num_heads": torch_model.blocks[0].attn.num_heads,
            "model_dim": torch_model.model_dim,
            "intermediate_dim": torch_model.intermediate_dim,
            "max_seq_len": torch_model.blocks[0].attn.rotary.cos.size(0),
            "use_gated_proj": torch_model.blocks[0].mlp.use_gated_proj,
            "use_rms_norm": torch_model.use_rms_norm,
            "use_gqa": torch_model.blocks[0].attn.use_gqa,
            "num_kv_heads": torch_model.blocks[0].attn.num_kv_heads
            if torch_model.blocks[0].attn.use_gqa
            else None,
            "use_rope_scaling": torch_model.blocks[0].attn.rotary.use_rope_scaling,
            "rope_scaling_factor": torch_model.blocks[0].attn.rotary.rope_scaling_factor,
            "rope_scaling_type": torch_model.blocks[0].attn.rotary.rope_scaling_type,
            "use_linear_cross_entropy": torch_model.use_linear_cross_entropy,
            "reorder_norms": torch_model.reorder_norms,
        }

    config = GPTConfig(**config_dict)
    hf_model = GPTForCausalLM.__new__(GPTForCausalLM)
    PreTrainedModel.__init__(hf_model, config)

    # Transfer modules directly (no copying)
    hf_model.embed = torch_model.embed
    hf_model.embedding_norm = torch_model.embedding_norm
    hf_model.final_norm = torch_model.final_norm
    hf_model.blocks = torch_model.blocks
    hf_model.lm_head = torch_model.lm_head

    return hf_model
