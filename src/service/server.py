from __future__ import annotations

import time
from concurrent import futures

import grpc

from src.backends.base import GenerateInput
from src.backends.pytorch_eager import PytorchEagerBackend
from src.generated import inference_pb2, inference_pb2_grpc


class InferenceService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self) -> None:
        self.backend = PytorchEagerBackend()

    def _to_generate_input(self, request: inference_pb2.GenerateRequest) -> GenerateInput:
        return GenerateInput(
            request_id=request.request_id,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            stop_sequences=list(request.stop_sequences),
            model_id=request.model_id,
        )

    def Generate(self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext) -> inference_pb2.GenerateResponse:
        start = time.perf_counter()
        req = self._to_generate_input(request)

        first_token_at = None
        stream_events = []
        for ev in self.backend.generate_stream(req):
            stream_events.append(ev)
            if first_token_at is None:
                first_token_at = time.perf_counter()

        result = self.backend.finalize_stream(req, stream_events)
        end = time.perf_counter()

        ttft_ms = 0.0 if first_token_at is None else (first_token_at - start) * 1000.0
        total_ms = (end - start) * 1000.0

        return inference_pb2.GenerateResponse(
            request_id=req.request_id,
            text=result.text,
            token_ids=result.token_ids,
            usage=inference_pb2.UsageStats(
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
                ttft_ms=ttft_ms,
                total_latency_ms=total_ms,
            ),
        )

    def GenerateStream(self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext):
        start = time.perf_counter()
        req = self._to_generate_input(request)

        first_token_at = None
        events = []

        for ev in self.backend.generate_stream(req):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            events.append(ev)

            yield inference_pb2.GenerateStreamChunk(
                request_id=req.request_id,
                token=inference_pb2.TokenChunk(
                    token_id=ev.token_id,
                    token_text=ev.token_text,
                    token_index=ev.token_index,
                ),
            )

        result = self.backend.finalize_stream(req, events)
        end = time.perf_counter()

        ttft_ms = 0.0 if first_token_at is None else (first_token_at - start) * 1000.0
        total_ms = (end - start) * 1000.0

        yield inference_pb2.GenerateStreamChunk(
            request_id=req.request_id,
            final=inference_pb2.FinalChunk(
                full_text=result.text,
                token_ids=result.token_ids,
                usage=inference_pb2.UsageStats(
                    prompt_tokens=result.prompt_tokens,
                    generated_tokens=result.generated_tokens,
                    ttft_ms=ttft_ms,
                    total_latency_ms=total_ms,
                ),
                finish_reason=result.finish_reason,
            ),
        )


def serve(host: str = "0.0.0.0", port: int = 50051) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"InferenceService listening on {host}:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()