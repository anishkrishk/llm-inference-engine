import pytest

from src.kv_cache.allocator import BlockAllocator
from src.kv_cache.block_table import BlockTable
from src.model.dummy import DummyModel, DummyModelConfig
from src.model.types import GenerateInput
from src.scheduler.engine import Engine, EngineConfig
from src.scheduler.scheduler import Scheduler, SchedulerConfig
from src.scheduler.sequence import Sequence, SequenceStatus


def _make_seq(
    allocator: BlockAllocator,
    request_id: str,
    prompt_len: int,
    max_new: int,
    block_size: int = 4,
) -> Sequence:
    bt = BlockTable(allocator, block_size=block_size)
    return Sequence(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        max_new_tokens=max_new,
        block_table=bt,
    )


def _drain(sched: Scheduler, model: DummyModel, guard: int = 500) -> list[Sequence]:
    """Run the scheduler to completion, returning sequences in finish order."""
    finished: list[Sequence] = []
    steps = 0
    while sched.has_work:
        out = sched.step()
        if out.is_empty:
            # The only way step() can return empty while has_work is true is
            # if the pool literally cannot hold the smallest waiting seq.
            raise RuntimeError("scheduler produced an empty step but has work")
        toks = model.step(out)
        finished.extend(sched.commit(out, toks))
        steps += 1
        if steps >= guard:
            raise RuntimeError(f"scheduler did not finish within {guard} steps")
    return finished


def test_single_sequence_end_to_end() -> None:
    a = BlockAllocator(num_blocks=8)
    sched = Scheduler(a, SchedulerConfig(block_size=4, max_num_seqs=4))
    seq = _make_seq(a, "r0", prompt_len=3, max_new=5)
    sched.add_sequence(seq)

    model = DummyModel(DummyModelConfig(vocab_size=1000))
    finished = _drain(sched, model)
    assert len(finished) == 1
    assert finished[0].request_id == "r0"
    assert finished[0].output_len == 5
    assert finished[0].status == SequenceStatus.FINISHED_LENGTH
    assert a.num_free == a.num_total


def test_multiple_sequences_all_finish() -> None:
    a = BlockAllocator(num_blocks=32)
    sched = Scheduler(a, SchedulerConfig(block_size=4, max_num_seqs=8))
    for i in range(4):
        sched.add_sequence(_make_seq(a, f"r{i}", prompt_len=2, max_new=3))

    model = DummyModel(DummyModelConfig(vocab_size=1000))
    finished = _drain(sched, model)
    assert sorted(s.request_id for s in finished) == ["r0", "r1", "r2", "r3"]
    for s in finished:
        assert s.output_len == 3
    assert a.num_free == a.num_total


def test_admission_respects_max_num_seqs() -> None:
    a = BlockAllocator(num_blocks=64)
    sched = Scheduler(a, SchedulerConfig(block_size=4, max_num_seqs=2))
    for i in range(5):
        sched.add_sequence(_make_seq(a, f"r{i}", prompt_len=2, max_new=2))

    model = DummyModel(DummyModelConfig(vocab_size=1000))

    out = sched.step()
    assert len(out.prefill_seqs) == 2
    assert sched.num_running == 2
    assert sched.num_waiting == 3

    # Run to completion and confirm every request finishes.
    sched.commit(out, model.step(out))
    finished = _drain(sched, model)
    all_finished_ids = {f.request_id for f in finished} | {"r0", "r1"}
    assert all_finished_ids == {"r0", "r1", "r2", "r3", "r4"}
    assert a.num_free == a.num_total


def test_preemption_under_tight_budget() -> None:
    """
    Under a block budget smaller than the concurrent demand the scheduler
    must preempt, re-admit, and still finish every sequence correctly.
    """
    a = BlockAllocator(num_blocks=4)
    sched = Scheduler(a, SchedulerConfig(block_size=4, max_num_seqs=8))
    for i in range(4):
        sched.add_sequence(_make_seq(a, f"r{i}", prompt_len=4, max_new=6))

    model = DummyModel(DummyModelConfig(vocab_size=1000))
    finished = _drain(sched, model)

    assert len(finished) == 4
    for s in finished:
        assert s.output_len == 6
        assert s.status == SequenceStatus.FINISHED_LENGTH
    assert a.num_free == a.num_total


def test_recompute_preempt_preserves_token_stream() -> None:
    """
    Running a sequence straight through must produce the same tokens as
    running the same sequence while a decoy competes for blocks and
    forces recompute-based preemption. This is the correctness invariant
    that recompute preemption has to uphold.
    """
    cfg = DummyModelConfig(vocab_size=1000)
    model = DummyModel(cfg)

    # Reference run: plenty of blocks, no preemption possible.
    a_ref = BlockAllocator(num_blocks=16)
    sched_ref = Scheduler(a_ref, SchedulerConfig(block_size=4, max_num_seqs=4))
    ref = _make_seq(a_ref, "rX", prompt_len=4, max_new=8)
    sched_ref.add_sequence(ref)
    _drain(sched_ref, model)
    tokens_ref = list(ref.output_token_ids)

    # Contentious run: tight pool with a decoy to force preemption of rX.
    a_ct = BlockAllocator(num_blocks=3)
    sched_ct = Scheduler(a_ct, SchedulerConfig(block_size=4, max_num_seqs=4))
    seq_x = _make_seq(a_ct, "rX", prompt_len=4, max_new=8)
    seq_y = _make_seq(a_ct, "rY", prompt_len=4, max_new=8)
    sched_ct.add_sequence(seq_x)
    sched_ct.add_sequence(seq_y)
    _drain(sched_ct, model)
    tokens_ct = list(seq_x.output_token_ids)

    assert tokens_ref == tokens_ct
    assert a_ct.num_free == a_ct.num_total


def test_commit_raises_if_model_omits_a_token() -> None:
    a = BlockAllocator(num_blocks=8)
    sched = Scheduler(a, SchedulerConfig(block_size=4, max_num_seqs=4))
    sched.add_sequence(_make_seq(a, "r0", prompt_len=2, max_new=3))
    out = sched.step()
    assert len(out.prefill_seqs) == 1

    # Hand commit() an empty token dict; this must raise.
    with pytest.raises(KeyError):
        sched.commit(out, {})


def test_finished_seq_frees_its_blocks() -> None:
    a = BlockAllocator(num_blocks=8)
    sched = Scheduler(a, SchedulerConfig(block_size=4, max_num_seqs=4))
    sched.add_sequence(_make_seq(a, "r0", prompt_len=2, max_new=1))

    model = DummyModel(DummyModelConfig(vocab_size=1000))
    # One step is enough for a max_new=1 sequence: prefill + emit one token.
    out = sched.step()
    finished = sched.commit(out, model.step(out))
    assert len(finished) == 1
    assert a.num_free == a.num_total
    assert sched.num_running == 0


def test_engine_wrapper_smoke() -> None:
    cfg = EngineConfig(
        num_blocks=16, block_size=4, max_num_seqs=4, max_num_batched_tokens=256
    )
    engine = Engine(cfg, DummyModel(DummyModelConfig(vocab_size=1000)))
    req = GenerateInput(
        request_id="rE",
        prompt="hello world",
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        seed=0,
    )
    engine.add_request(req, prompt_token_ids=[1, 2, 3, 4, 5])
    steps = engine.run_until_done()
    assert steps >= 4
    result = engine.get_result("rE")
    assert result is not None
    assert result.output_len == 4
    assert result.status == SequenceStatus.FINISHED_LENGTH
    assert engine.num_free_blocks == cfg.num_blocks
