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
from typing import List, Literal, Sequence as SequenceT

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kv_cache.kv_pool import KVPool

AttentionBackend = Literal["eager", "triton", "paged"]


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

    def __init__(
        self, config: GPT2Config, attention_backend: AttentionBackend = "eager"
    ) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        if attention_backend not in ("eager", "triton", "paged"):
            raise ValueError(
                f"unknown attention_backend {attention_backend!r}; "
                f"expected 'eager', 'triton', or 'paged'"
            )
        self._backend: AttentionBackend = attention_backend
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
        q = q.reshape(L, H, d)
        k = k.reshape(L, H, d)
        v = v.reshape(L, H, d)

        # Scatter the newly-computed K,V into the paged cache at the
        # positions [start_pos, start_pos + L).
        kv_pool.write(layer_idx, block_ids, start_pos, k, v)

        T = start_pos + L

        # The "paged" backend skips the gather entirely on the decode
        # path (L == 1) and reads K,V directly from the pool tensors
        # via a Triton kernel that walks the block table. For prefill
        # (L > 1) it falls back to the same triton FA path the
        # "triton" backend uses.
        if self._backend == "paged" and L == 1:
            out = self._paged_decode(q, layer_idx, block_ids, T, kv_pool)
        else:
            k_cache, v_cache = kv_pool.read(layer_idx, block_ids, T)
            if self._backend in ("triton", "paged"):
                out = self._triton_attention(q, k_cache, v_cache)
            else:
                out = self._eager_attention(q, k_cache, v_cache, L, T, x.device)

        out = out.reshape(L, D)
        return self.c_proj(out)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _eager_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        L: int,
        T: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Pure-PyTorch materialized causal attention. Reference for the kernel."""
        q_h = q.transpose(0, 1)            # [H, L, d]
        k_h = k_cache.transpose(0, 1)      # [H, T, d]
        v_h = v_cache.transpose(0, 1)      # [H, T, d]
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * self._scale  # [H, L, T]
        q_pos = torch.arange(T - L, T, device=device)  # [L]
        k_pos = torch.arange(T, device=device)         # [T]
        mask = k_pos[None, :] > q_pos[:, None]
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(attn, v_h)                    # [H, L, d]
        return out.transpose(0, 1).contiguous()          # [L, H, d]

    def _triton_attention(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Triton FlashAttention path. Causal with queries trailing the cache."""
        # Imported lazily so importing this module on a CUDA-less box is fine.
        from src.kernels.triton.flash_attention import triton_flash_attention

        return triton_flash_attention(q, k_cache, v_cache, causal=True)

    def _paged_decode(
        self,
        q: torch.Tensor,
        layer_idx: int,
        block_ids: SequenceT[int],
        context_len: int,
        kv_pool: KVPool,
    ) -> torch.Tensor:
        """
        Single-query attention via the paged decode kernel. Reads K,V
        directly from the pool tensors using the block table; no
        intermediate gather.
        """
        from src.kernels.triton.paged_decode import paged_decode_attention

        block_table = torch.as_tensor(
            list(block_ids), dtype=torch.int32, device=q.device
        )
        return paged_decode_attention(
            q,
            kv_pool.k_cache,
            kv_pool.v_cache,
            block_table,
            context_len=context_len,
            layer_idx=layer_idx,
        )

    def decode_batch(
        self,
        x_batch: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        block_tables_2d: torch.Tensor,
        context_lens: torch.Tensor,
        kv_pool: KVPool,
    ) -> torch.Tensor:
        """
        Fully batched decode: projects Q/K/V in one matmul, writes all
        K/V in one vectorized scatter, runs paged attention for all B
        sequences in one kernel launch. Returns [B, D].

        block_tables_2d : [B, max_blocks] int32 CUDA tensor (padded)
        context_lens    : [B] int32 CUDA tensor
        """
        from src.kernels.triton.paged_decode import batched_paged_decode_attention

        B, D = x_batch.shape
        H, d = self.num_heads, self.head_dim

        qkv = self.c_attn(x_batch)  # [B, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, H, d)
        k = k.reshape(B, H, d)
        v = v.reshape(B, H, d)

        # Batched scatter write: one op for all B sequences.
        kv_pool.write_decode_batch(layer_idx, block_tables_2d, positions, k, v)

        # Batched paged decode: one kernel launch for all (head, seq) pairs.
        attn_out = batched_paged_decode_attention(
            q, kv_pool.k_cache, kv_pool.v_cache,
            block_tables_2d, context_lens, layer_idx,
        )  # [B, H, d]

        return self.c_proj(attn_out.reshape(B, D))


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

    def __init__(
        self, config: GPT2Config, attention_backend: AttentionBackend = "eager"
    ) -> None:
        super().__init__()
        d = config.hidden_size
        eps = config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(d, eps=eps)
        self.attn = GPT2PagedAttention(config, attention_backend=attention_backend)
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

    def decode_batch(
        self,
        x: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        block_tables_2d: torch.Tensor,
        context_lens: torch.Tensor,
        kv_pool: KVPool,
    ) -> torch.Tensor:
        """Fully batched decode: all linears, writes, and attention batched."""
        x = x + self.attn.decode_batch(
            self.ln_1(x),
            layer_idx=layer_idx,
            positions=positions,
            block_tables_2d=block_tables_2d,
            context_lens=context_lens,
            kv_pool=kv_pool,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


# ----------------------------------------------------------------------
# Whole model
# ----------------------------------------------------------------------


class GPT2PagedModel(nn.Module):
    def __init__(
        self, config: GPT2Config, attention_backend: AttentionBackend = "eager"
    ) -> None:
        super().__init__()
        self.config = config
        self.attention_backend: AttentionBackend = attention_backend
        d = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, d)
        self.wpe = nn.Embedding(config.max_position_embeddings, d)
        self.blocks = nn.ModuleList(
            [
                GPT2Block(config, attention_backend=attention_backend)
                for _ in range(config.num_layers)
            ]
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

    @torch.inference_mode()
    def decode_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables_2d: torch.Tensor,
        context_lens: torch.Tensor,
        kv_pool: KVPool,
    ) -> torch.Tensor:
        """
        Fully batched decode forward for B sequences each generating one
        token. All embeddings, linear projections, LayerNorms, MLP, KV
        writes, and paged attention are batched over the B dimension in a
        single call per layer.

        Args
        ----
        input_ids      : [B]   one token per sequence
        positions      : [B]   absolute position per sequence
        block_tables_2d: [B, max_blocks] int32 CUDA — padded block tables
        context_lens   : [B] int32 CUDA — positions + 1
        kv_pool        : shared KV cache

        Returns
        -------
        logits : [B, V]
        """
        B = input_ids.shape[0]
        x = self.wte(input_ids) + self.wpe(positions)  # [B, D]

        for layer_idx, block in enumerate(self.blocks):
            x = block.decode_batch(
                x,
                layer_idx=layer_idx,
                positions=positions,
                block_tables_2d=block_tables_2d,
                context_lens=context_lens,
                kv_pool=kv_pool,
            )

        x = self.ln_f(x)
        logits = x @ self.wte.weight.t()  # [B, V]
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


def _replace_linears_with_int8(module: nn.Module) -> None:
    """
    Recursively replace every ``nn.Linear`` in the module tree with
    ``bitsandbytes.nn.Linear8bitLt``. The fp16 weight data is copied
    in; bitsandbytes quantizes to int8 + absmax on the first CUDA
    forward pass. LayerNorm and Embedding layers are left untouched
    (quantizing those hurts quality disproportionately).
    """
    import bitsandbytes as bnb

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            int8_lin = bnb.nn.Linear8bitLt(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                has_fp16_weights=False,
            )
            int8_lin.weight = bnb.nn.Int8Params(
                child.weight.data.clone(),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if child.bias is not None:
                int8_lin.bias = nn.Parameter(child.bias.data.clone())
            setattr(module, name, int8_lin)
        else:
            _replace_linears_with_int8(child)


def load_gpt2_from_hf(
    hf_model_name: str = "gpt2",
    dtype: torch.dtype = torch.float16,
    attention_backend: AttentionBackend = "eager",
    quantize_int8: bool = False,
) -> GPT2PagedModel:
    """
    Build a GPT2PagedModel and copy weights over from a HuggingFace
    GPT2LMHeadModel checkpoint. Returns the model in eval mode.

    If ``quantize_int8=True``, all Linear layers are replaced with
    bitsandbytes ``Linear8bitLt`` after weight loading. The model must
    be moved to CUDA before the first forward pass so bitsandbytes can
    run its calibration quantization.
    """
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
    model = GPT2PagedModel(cfg, attention_backend=attention_backend)

    src_sd = hf.state_dict()
    dst_sd = model.state_dict()

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
            if any(hf_key.endswith(suffix) for suffix in _HF_CONV1D_WEIGHT_SUFFIXES):
                tensor = tensor.t().contiguous()
            dst_sd[our_key].copy_(tensor.to(dtype))

    model.to(dtype=dtype)

    if quantize_int8:
        _replace_linears_with_int8(model)

    model.eval()
    return model
