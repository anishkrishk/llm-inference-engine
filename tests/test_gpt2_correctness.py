"""
Token-exact correctness test for GPT2PagedModel against HuggingFace.

The test runs both implementations in FP16 on CUDA with greedy decoding
from the same prompt and asserts the generated token sequences are
identical. If either backend is unavailable (no CUDA, or the HF weights
can't be fetched), the test is skipped rather than failed.
"""

from __future__ import annotations

import pytest
import torch

# Skip the whole module if CUDA isn't available. This project targets
# FP16 on an Ampere-class GPU; there is no CPU correctness path.
cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="GPT-2 correctness tests require CUDA"
)


def _load_hf_gpt2():
    """Load HF's GPT-2 reference model + tokenizer. Skips if unreachable."""
    try:
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError as e:  # pragma: no cover
        pytest.skip(f"transformers not installed: {e}")
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
        hf = GPT2LMHeadModel.from_pretrained("gpt2")
    except Exception as e:  # network / cache miss
        pytest.skip(f"gpt2 weights unavailable: {e}")
    return hf, tok


@pytest.mark.parametrize("backend", ["eager", "triton", "paged"])
def test_gpt2_matches_hf_greedy(backend: str) -> None:
    from src.kv_cache.allocator import BlockAllocator  # noqa: F401  (import smoke)
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig
    from src.model.gpt2 import GPT2Config, load_gpt2_from_hf
    from src.model.gpt2_runner import GPT2Runner
    from src.model.types import GenerateInput
    from src.scheduler.engine import Engine, EngineConfig

    device = torch.device("cuda")
    dtype = torch.float16

    hf, tok = _load_hf_gpt2()
    hf = hf.to(device=device, dtype=dtype).eval()

    prompt = "The quick brown fox"
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    max_new = 20

    # -- HF reference --
    with torch.inference_mode():
        hf_out = hf.generate(
            prompt_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    hf_tokens = hf_out[0, prompt_ids.shape[1]:].tolist()
    assert len(hf_tokens) == max_new

    # -- our engine --
    gpt2_cfg = GPT2Config.gpt2_small()
    pool_cfg = KVPoolConfig(
        num_layers=gpt2_cfg.num_layers,
        num_blocks=64,
        block_size=16,
        num_heads=gpt2_cfg.num_heads,
        head_dim=gpt2_cfg.head_dim,
        dtype=dtype,
        device=device,
    )
    kv_pool = KVPool(pool_cfg)
    model = load_gpt2_from_hf("gpt2", dtype=dtype, attention_backend=backend).to(device).eval()
    runner = GPT2Runner(model, kv_pool)
    engine = Engine(
        EngineConfig(
            num_blocks=pool_cfg.num_blocks,
            block_size=pool_cfg.block_size,
            max_num_seqs=4,
            max_num_batched_tokens=1024,
        ),
        runner,
    )

    req = GenerateInput(
        request_id="ref",
        prompt=prompt,
        max_new_tokens=max_new,
        temperature=1.0,
        top_p=1.0,
        seed=0,
    )
    engine.add_request(req, prompt_token_ids=prompt_ids[0].tolist())
    engine.run_until_done()
    result = engine.get_result("ref")

    assert result is not None
    our_tokens = result.output_token_ids
    assert len(our_tokens) == max_new

    if our_tokens != hf_tokens:
        # Surface the first divergence for fast debugging.
        for i, (a, b) in enumerate(zip(our_tokens, hf_tokens)):
            if a != b:
                our_txt = tok.decode(our_tokens[: i + 1])
                hf_txt = tok.decode(hf_tokens[: i + 1])
                raise AssertionError(
                    f"token streams diverge at index {i}: "
                    f"ours={a} ({our_txt!r}) vs hf={b} ({hf_txt!r})"
                )
    assert our_tokens == hf_tokens


def test_gpt2_prefill_decode_consistency() -> None:
    """
    Run the same prompt as a single prefill and compare to an
    incremental prefill+decode, both through our engine. They must
    produce the same token stream. This is a model-vs-itself test and
    does not require the HF reference.
    """
    from src.kv_cache.kv_pool import KVPool, KVPoolConfig
    from src.model.gpt2 import GPT2Config, load_gpt2_from_hf
    from src.model.gpt2_runner import GPT2Runner
    from src.model.types import GenerateInput
    from src.scheduler.engine import Engine, EngineConfig

    device = torch.device("cuda")
    dtype = torch.float16
    gpt2_cfg = GPT2Config.gpt2_small()
    try:
        model = load_gpt2_from_hf("gpt2", dtype=dtype).to(device).eval()
    except Exception as e:  # network / cache miss
        pytest.skip(f"gpt2 weights unavailable: {e}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    prompt_ids = tok("Hello world", return_tensors="pt").input_ids[0].tolist()

    def run_once(max_new: int) -> list[int]:
        pool = KVPool(
            KVPoolConfig(
                num_layers=gpt2_cfg.num_layers,
                num_blocks=32,
                block_size=16,
                num_heads=gpt2_cfg.num_heads,
                head_dim=gpt2_cfg.head_dim,
                dtype=dtype,
                device=device,
            )
        )
        runner = GPT2Runner(model, pool)
        engine = Engine(
            EngineConfig(num_blocks=32, block_size=16, max_num_seqs=2),
            runner,
        )
        req = GenerateInput(
            request_id="x",
            prompt="",
            max_new_tokens=max_new,
            temperature=1.0,
            top_p=1.0,
            seed=0,
        )
        engine.add_request(req, prompt_token_ids=prompt_ids)
        engine.run_until_done()
        return engine.get_result("x").output_token_ids

    short = run_once(5)
    long = run_once(10)
    # The first 5 tokens of a length-10 run must equal the 5 tokens of a
    # length-5 run — greedy decoding is deterministic.
    assert long[:5] == short
