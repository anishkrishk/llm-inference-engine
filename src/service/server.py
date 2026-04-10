"""
Async gRPC server driving the inference engine.

Replaces the earlier sync ThreadPoolExecutor-based server with a fully
async implementation built on ``grpc.aio``. The key structural change:
instead of the server blocking on a backend's ``generate_stream()``
iterator per RPC, the server enqueues incoming requests into the
engine's scheduler, runs a shared async engine loop that calls
``step()`` at a fixed cadence, and dispatches produced tokens to
per-request asyncio.Queues that the streaming RPCs drain.

The engine loop is a single asyncio.Task that owns all GPU interaction.
RPC handlers are purely async I/O — they submit work and await tokens
on their queues, never touching the model or CUDA directly. This
avoids the need for cross-thread GPU synchronization entirely.

Backpressure: each request's token queue has a bounded size. If the
client is slow to consume tokens, the queue fills and the engine
loop's ``put_nowait`` will raise; the server logs the overrun and
drops the sequence. In practice, gRPC's HTTP/2 flow control handles
most of the backpressure before we hit the queue limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional

import grpc

from src.generated import inference_pb2, inference_pb2_grpc
from src.model.types import GenerateInput

logger = logging.getLogger(__name__)


class _PendingRequest:
    """Per-request state held by the server while the engine processes it."""

    def __init__(self, req: GenerateInput, prompt_token_ids: list[int]) -> None:
        self.req = req
        self.prompt_token_ids = prompt_token_ids
        self.token_queue: asyncio.Queue[Optional[inference_pb2.GenerateStreamChunk]] = (
            asyncio.Queue(maxsize=256)
        )
        self.start_time = time.perf_counter()
        self.first_token_time: Optional[float] = None


class AsyncInferenceService(inference_pb2_grpc.InferenceServiceServicer):
    """
    Async gRPC servicer. All GPU work is delegated to the engine loop;
    the RPC handlers only enqueue requests and await token events.
    """

    def __init__(self, engine, tokenizer) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self._pending: Dict[str, _PendingRequest] = {}
        self._engine_task: Optional[asyncio.Task] = None

    async def start_engine_loop(self) -> None:
        self._engine_task = asyncio.get_running_loop().create_task(
            self._engine_loop()
        )

    async def _engine_loop(self) -> None:
        """
        Run the scheduler step loop forever. Each iteration:
        1. step() the engine (produces tokens for running sequences)
        2. Dispatch produced tokens to their request queues
        3. If no work, sleep briefly to avoid busy-spinning
        """
        while True:
            if not self._engine.has_work():
                await asyncio.sleep(0.001)
                continue

            output = self._engine.step()

            # Check for newly finished sequences.
            for rid, pending in list(self._pending.items()):
                result = self._engine.get_result(rid)
                if result is None:
                    # Still running — send a token chunk if available.
                    # The engine doesn't directly expose per-step tokens;
                    # we track them via the Sequence's output_token_ids
                    # growing. For now, we defer token streaming to the
                    # final result.
                    continue

                # Sequence finished. Build the final chunk and signal done.
                end_time = time.perf_counter()
                ttft = 0.0
                if pending.first_token_time is not None:
                    ttft = (pending.first_token_time - pending.start_time) * 1000.0
                total_ms = (end_time - pending.start_time) * 1000.0

                output_text = self._tokenizer.decode(result.output_token_ids)

                final_chunk = inference_pb2.GenerateStreamChunk(
                    request_id=rid,
                    final=inference_pb2.FinalChunk(
                        full_text=output_text,
                        token_ids=result.output_token_ids,
                        usage=inference_pb2.UsageStats(
                            prompt_tokens=result.prompt_len,
                            generated_tokens=result.output_len,
                            ttft_ms=ttft,
                            total_latency_ms=total_ms,
                        ),
                        finish_reason=result.status.value,
                    ),
                )
                try:
                    pending.token_queue.put_nowait(final_chunk)
                    pending.token_queue.put_nowait(None)  # sentinel: stream done
                except asyncio.QueueFull:
                    logger.warning("token queue full for %s; dropping", rid)
                    pending.token_queue.put_nowait(None)

                del self._pending[rid]

            # Yield control so RPC handlers can run.
            await asyncio.sleep(0)

    def _enqueue_request(self, request: inference_pb2.GenerateRequest) -> _PendingRequest:
        rid = request.request_id or str(uuid.uuid4())
        prompt_ids = self._tokenizer(request.prompt, return_tensors="pt").input_ids[0].tolist()

        req = GenerateInput(
            request_id=rid,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens or 64,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            seed=request.seed,
            stop_sequences=list(request.stop_sequences),
            model_id=request.model_id,
        )
        pending = _PendingRequest(req, prompt_ids)
        self._pending[rid] = pending
        self._engine.add_request(req, prompt_token_ids=prompt_ids)
        return pending

    async def Generate(
        self,
        request: inference_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> inference_pb2.GenerateResponse:
        pending = self._enqueue_request(request)
        # Wait for the final result.
        while True:
            chunk = await pending.token_queue.get()
            if chunk is None:
                break
            if chunk.HasField("final"):
                f = chunk.final
                return inference_pb2.GenerateResponse(
                    request_id=pending.req.request_id,
                    text=f.full_text,
                    token_ids=list(f.token_ids),
                    usage=f.usage,
                )
        # Should not reach here; defensive.
        return inference_pb2.GenerateResponse(request_id=pending.req.request_id)

    async def GenerateStream(
        self,
        request: inference_pb2.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ):
        pending = self._enqueue_request(request)
        while True:
            chunk = await pending.token_queue.get()
            if chunk is None:
                break
            yield chunk


async def serve_async(engine, tokenizer, host: str = "0.0.0.0", port: int = 50051) -> None:
    """
    Start the async gRPC server with the given engine and tokenizer.
    Blocks until the server is terminated.
    """
    server = grpc.aio.server()
    service = AsyncInferenceService(engine, tokenizer)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    await server.start()
    await service.start_engine_loop()
    logger.info("InferenceService listening on %s:%d", host, port)
    print(f"InferenceService listening on {host}:{port}")
    await server.wait_for_termination()
