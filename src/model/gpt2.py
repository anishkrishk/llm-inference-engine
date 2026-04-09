"""
GPT-2 decoder-only transformer with paged KV cache support.

A clean re-implementation of the GPT-2 forward pass — intentionally
in one file, nanoGPT-style, so the whole thing can be read top to
bottom. Weights are loaded from a HuggingFace ``GPT2LMHeadModel``
checkpoint; the architecture here matches HF's implementation exactly
so the two produce identical tokens under greedy decoding.

Why reimplement instead of using HF's forward directly? HF stores its
KV cache as a per-sequence contiguous tensor (``past_key_values``) that
is incompatible with a paged pool. Swapping HF's attention module while
keeping the rest of the stack would mean fighting the HF call graph
forever. It is cheaper and cleaner to own the whole model and use HF
only as a weight source.

Key architectural facts about GPT-2 (the things you have to get right):

  * Pre-LayerNorm: ``ln_1 -> attention -> residual -> ln_2 -> mlp -> residual``
  * Learned absolute position embeddings (not rotary).
  * Weight-tied LM head: the same ``wte`` matrix is used (transposed) as
    the output projection.
  * "New GELU" approximation: ``F.gelu(x, approximate="tanh")``.
  * LayerNorm epsilon = 1e-5.
  * HF stores the attention projections as ``Conv1D``, which shapes its
    weight as ``[in, out]`` — the opposite of ``nn.Linear``. Loading the
    state dict into ``nn.Linear`` therefore requires transposing these
    tensors. This affects ``c_attn``, ``c_proj`` in both attention and
    MLP. LayerNorm and embeddings do NOT need transposing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence as SequenceT

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kv_cache.kv_pool import KVPool


@dataclass(frozen=True)
class GPT2Config:
    num_layers: int
    num_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_epsilon: float = 1e-5

    @property
    def hidden_size(self) -> int:
        return self.num_heads * self.head_dim

    @classmethod
    def gpt2_small(cls) -> "GPT2Config":
        return cls(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            vocab_size=50257,
            max_position_embeddings=1024,
        )


# ----------------------------------------------------------------------
# Attention
# ----------------------------------------------------------------------


class GPT2PagedAttention(nn.Module):
    """
    Multi-head self-attention operating on a paged KV cache.

    Forward signature:

        x           [L, D]           new hidden states to project into Q/K/V
        layer_idx   int              which layer's cache slab to touch
        start_pos   int              logical position of x[0] in the full
                                     sequence; decode has L=1 and a non-zero
                                     start_pos, prefill has L=prompt_len and
                                     start_pos=0
        block_ids   list[int]        physical block ids for this sequence
        kv_pool     KVPool           shared cache tensors

    Returns the attention output of shape [L, D].
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        # HF fuses Q, K, V into one projection called c_attn.
        self.c_attn = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.c_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        start_pos: int,
        block_ids: SequenceT[int],
        kv_pool: KVPool,
    ) -> torch.Tensor:
        L, D = x.shape
        H, d = self.num_heads, self.head_dim

        # Project to Q/K/V. c_attn is shape [D, 3D]; its output is [L, 3D]
        # which we chunk into three [L, D] pieces.
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape [L, D] -> [L, H, d] so each head sees its own d-wide slice.
        q = q.view(L, H, d)
        k = k.view(L, H, d)
        v = v.view(L, H, d)

        # Scatter the newly-computed K,V into the paged cache at the
        # positions [start_pos, start_pos + L).
        kv_pool.write(layer_idx, block_ids, start_pos, k, v)

        # Gather the full history of K,V for this sequence back from the
        # cache. T = total tokens currently cached (including the ones
        # we just wrote above).
        T = start_pos + L
        k_cache, v_cache = kv_pool.read(layer_idx, block_ids, T)  # [T, H, d] each

        # Move the head axis up front so the attention matmul is clean:
        # q:     [H, L, d]
        # k/v:   [H, T, d]
        q_h = q.transpose(0, 1)            # [H, L, d]
        k_h = k_cache.transpose(0, 1)      # [H, T, d]
        v_h = v_cache.transpose(0, 1)      # [H, T, d]

        # scores[h, i, j] = q_h[h, i] . k_h[h, j]
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * self._scale  # [H, L, T]

        # Causal mask: token i (at absolute position start_pos + i) can
        # attend to positions 0..start_pos+i. Vectorized construction.
        q_pos = torch.arange(start_pos, start_pos + L, device=x.device)  # [L]
        k_pos = torch.arange(T, device=x.device)                          # [T]
        # True where the key position is *after* the query position.
        mask = k_pos[None, :] > q_pos[:, None]  # [L, T]
        # Broadcast over heads and apply.
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        # Softmax over the key axis. Upcast to fp32 inside the softmax
        # for numerical stability with fp16 scores.
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)

        # Weighted sum of values.
        out = torch.matmul(attn, v_h)                    # [H, L, d]
        out = out.transpose(0, 1).contiguous().view(L, D)  # [L, D]
        return self.c_proj(out)


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------


class GPT2MLP(nn.Module):
    """
    Two-layer feedforward network with a 4x expansion factor and the
    tanh-approximation GELU activation GPT-2 was trained with.
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        d = config.hidden_size
        self.c_fc = nn.Linear(d, 4 * d, bias=True)
        self.c_proj = nn.Linear(4 * d, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


# ----------------------------------------------------------------------
# Transformer block
# ----------------------------------------------------------------------


class GPT2Block(nn.Module):
    """
    Pre-LayerNorm transformer block.

        x' = x + attn(ln_1(x))
        x'' = x' + mlp(ln_2(x'))
    """

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        d = config.hidden_size
        eps = config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(d, eps=eps)
        self.attn = GPT2PagedAttention(config)
        self.ln_2 = nn.LayerNorm(d, eps=eps)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        start_pos: int,
        block_ids: SequenceT[int],
        kv_pool: KVPool,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),
            layer_idx=layer_idx,
            start_pos=start_pos,
            block_ids=block_ids,
            kv_pool=kv_pool,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


# ----------------------------------------------------------------------
# Whole model
# ----------------------------------------------------------------------


class GPT2PagedModel(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, d)
        self.wpe = nn.Embedding(config.max_position_embeddings, d)
        self.blocks = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.num_layers)]
        )
        self.ln_f = nn.LayerNorm(d, eps=config.layer_norm_epsilon)
        # LM head is weight-tied to wte (see forward()) — no parameter here.

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int,
        block_ids: SequenceT[int],
        kv_pool: KVPool,
    ) -> torch.Tensor:
        """
        Run a forward pass over `input_ids`, treating them as new tokens
        starting at logical position `start_pos`.

        Shapes:
            input_ids: [L]           token ids to process
            returns:   [L, V]        logits, one row per input token
        """
        L = input_ids.shape[0]
        if L == 0:
            raise ValueError("forward called with empty input_ids")
        if start_pos + L > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {start_pos + L} exceeds GPT-2's "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )

        device = input_ids.device
        positions = torch.arange(start_pos, start_pos + L, device=device)

        # Token + learned absolute position embedding.
        x = self.wte(input_ids) + self.wpe(positions)  # [L, D]

        # Run through every transformer block.
        for layer_idx, block in enumerate(self.blocks):
            x = block(
                x,
                layer_idx=layer_idx,
                start_pos=start_pos,
                block_ids=block_ids,
                kv_pool=kv_pool,
            )

        x = self.ln_f(x)
        # Weight-tied LM head: logits = x @ wte.weight.T
        logits = x @ self.wte.weight.t()  # [L, V]
        return logits


# ----------------------------------------------------------------------
# Weight loading from HuggingFace
# ----------------------------------------------------------------------


# HF's GPT-2 attention/MLP projections are stored as `Conv1D`, whose
# weight is laid out as [in, out] — the *transpose* of nn.Linear's
# [out, in] layout. These four parameter name patterns need transposing
# on load; everything else is copied straight through.
_HF_CONV1D_WEIGHT_SUFFIXES = (
    ".attn.c_attn.weight",
    ".attn.c_proj.weight",
    ".mlp.c_fc.weight",
    ".mlp.c_proj.weight",
)


def load_gpt2_from_hf(
    hf_model_name: str = "gpt2",
    dtype: torch.dtype = torch.float16,
) -> GPT2PagedModel:
    """
    Build a GPT2PagedModel and copy weights over from a HuggingFace
    GPT2LMHeadModel checkpoint. Returns the model in eval mode.

    The caller is responsible for moving the model to the target device.
    """
    # Imported here so that users without `transformers` installed can
    # still import this module (e.g. for the tests that only exercise
    # the scheduler plumbing).
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(hf_model_name)
    hf_cfg = hf.config

    cfg = GPT2Config(
        num_layers=hf_cfg.n_layer,
        num_heads=hf_cfg.n_head,
        head_dim=hf_cfg.n_embd // hf_cfg.n_head,
        vocab_size=hf_cfg.vocab_size,
        max_position_embeddings=hf_cfg.n_positions,
        layer_norm_epsilon=hf_cfg.layer_norm_epsilon,
    )
    model = GPT2PagedModel(cfg)

    src_sd = hf.state_dict()
    dst_sd = model.state_dict()

    # Manually build a name mapping from HF's parameters to ours.
    mapping: dict[str, str] = {
        "transformer.wte.weight": "wte.weight",
        "transformer.wpe.weight": "wpe.weight",
        "transformer.ln_f.weight": "ln_f.weight",
        "transformer.ln_f.bias": "ln_f.bias",
    }
    for i in range(cfg.num_layers):
        for suffix in (
            "ln_1.weight", "ln_1.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ):
            mapping[f"transformer.h.{i}.{suffix}"] = f"blocks.{i}.{suffix}"

    with torch.no_grad():
        for hf_key, our_key in mapping.items():
            if hf_key not in src_sd:
                raise KeyError(f"missing HF parameter: {hf_key}")
            if our_key not in dst_sd:
                raise KeyError(f"missing local parameter: {our_key}")
            tensor = src_sd[hf_key]
            # Transpose Conv1D weights so they match nn.Linear layout.
            if any(hf_key.endswith(suffix) for suffix in _HF_CONV1D_WEIGHT_SUFFIXES):
                tensor = tensor.t().contiguous()
            dst_sd[our_key].copy_(tensor.to(dtype))

    # Convert the non-manually-copied parameters (there shouldn't be any
    # aside from what we just copied, but convert the whole model to the
    # target dtype for safety).
    model.to(dtype=dtype)
    model.eval()
    return model
