"""
Drive the scheduler with a batch of synthetic requests and print a
per-step timeline of admissions, decodes, preemptions, and free blocks.

The point is to make the batching dynamics visible: watch the running
set grow as requests are admitted, shrink as they finish, and thrash
when the block pool is tight enough to force preemption.

Run from repo root:
    python scripts/demo_scheduler.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Make the repo root importable when running this file directly with
# `python scripts/demo_scheduler.py` (no install, no PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.dummy import DummyModel, DummyModelConfig  # noqa: E402
from src.model.types import GenerateInput  # noqa: E402
from src.scheduler.engine import Engine, EngineConfig  # noqa: E402


@dataclass
class DemoRequest:
    req_id: str
    prompt_len: int
    max_new: int


def build_requests(n: int, rng: random.Random) -> List[DemoRequest]:
    reqs: List[DemoRequest] = []
    for i in range(n):
        prompt_len = rng.randint(4, 48)
        max_new = rng.randint(8, 40)
        reqs.append(DemoRequest(f"req{i:02d}", prompt_len, max_new))
    return reqs


def main() -> None:
    rng = random.Random(42)
    requests = build_requests(20, rng)

    # Deliberately tight pool: 24 blocks x 16 slots = 384 total slots.
    # With up to 8 concurrent sequences whose max total length approaches
    # ~90 tokens, the worst case (8 * 90 = 720 slots) exceeds the pool,
    # so the scheduler has to preempt under load.
    engine_cfg = EngineConfig(
        num_blocks=24,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=1024,
    )
    model = DummyModel(DummyModelConfig(vocab_size=50257))
    engine = Engine(engine_cfg, model)

    print(
        f"Scheduler demo: {len(requests)} requests, "
        f"pool={engine_cfg.num_blocks} blocks x {engine_cfg.block_size} slots "
        f"({engine_cfg.num_blocks * engine_cfg.block_size} token capacity), "
        f"max_seqs={engine_cfg.max_num_seqs}"
    )
    print()
    print(f"{'req_id':<8} {'prompt_len':>11} {'max_new':>8}")
    print("-" * 30)
    for r in requests:
        print(f"{r.req_id:<8} {r.prompt_len:>11} {r.max_new:>8}")
        engine.add_request(
            GenerateInput(
                request_id=r.req_id,
                prompt="",
                max_new_tokens=r.max_new,
                temperature=1.0,
                top_p=1.0,
                seed=0,
            ),
            prompt_token_ids=list(range(1, r.prompt_len + 1)),
        )
    print()

    header = (
        f"{'step':>4} | {'admit':>5} | {'decode':>6} | {'preempt':>7} "
        f"| {'wait':>4} | {'run':>3} | {'free':>4}"
    )
    print(header)
    print("-" * len(header))

    step = 0
    total_admissions = 0
    total_preemptions = 0
    peak_running = 0
    max_steps = 2000
    printed_rows = 0
    suppressed = 0

    while engine.has_work() and step < max_steps:
        out = engine.step()
        step += 1

        admit = len(out.prefill_seqs)
        decode = len(out.decode_seqs)
        preempt = len(out.preempted_seqs)

        total_admissions += admit
        total_preemptions += preempt
        peak_running = max(peak_running, engine.num_running)

        # Print interesting rows densely; compress runs of pure-decode steps
        # to keep the output readable.
        interesting = admit or preempt or step <= 3
        if interesting:
            if suppressed:
                print(f"  ...  {suppressed} pure-decode steps elided  ...")
                suppressed = 0
            print(
                f"{step:>4} | {admit:>5} | {decode:>6} | {preempt:>7} "
                f"| {engine.num_waiting:>4} | {engine.num_running:>3} "
                f"| {engine.num_free_blocks:>4}"
            )
            printed_rows += 1
        else:
            suppressed += 1

    if suppressed:
        print(f"  ...  {suppressed} pure-decode steps elided  ...")

    print()
    print("summary")
    print("-------")
    print(f"  total steps        : {step}")
    print(f"  total admissions   : {total_admissions}")
    print(f"  total preemptions  : {total_preemptions}")
    print(f"  peak concurrency   : {peak_running}")
    print(f"  blocks free at end : {engine.num_free_blocks}/{engine.num_total_blocks}")

    finished_ok = 0
    total_output_tokens = 0
    for r in requests:
        res = engine.get_result(r.req_id)
        if res is not None and res.output_len == r.max_new:
            finished_ok += 1
            total_output_tokens += res.output_len
    print(f"  finished correctly : {finished_ok}/{len(requests)}")
    print(f"  total output tokens: {total_output_tokens}")


if __name__ == "__main__":
    main()
