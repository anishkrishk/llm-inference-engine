from __future__ import annotations

import time

import grpc
from src.generated import inference_pb2, inference_pb2_grpc


def make_request(request_id: str) -> inference_pb2.GenerateRequest:
    return inference_pb2.GenerateRequest(
        request_id=request_id,
        prompt="Explain KV cache in one sentence.",
        max_new_tokens=12,
        temperature=0.7,
        top_p=0.9,
        seed=42,
        stop_sequences=[],
        model_id="qwen2.5-1.5b-instruct",
    )


def run_unary(stub: inference_pb2_grpc.InferenceServiceStub) -> None:
    req = make_request("req-unary-1")
    resp = stub.Generate(req, timeout=10)
    print("\n[Unary]")
    print("text:", resp.text)
    print("tokens:", list(resp.token_ids))
    print("usage:", resp.usage)


def run_stream(stub: inference_pb2_grpc.InferenceServiceStub) -> None:
    req = make_request("req-stream-1")
    print("\n[Stream]")
    t0 = time.perf_counter()
    for chunk in stub.GenerateStream(req, timeout=10):
        which = chunk.WhichOneof("payload")
        if which == "token":
            print(f"token[{chunk.token.token_index}] = {chunk.token.token_text!r}")
        elif which == "final":
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print("final_text:", chunk.final.full_text)
            print("finish_reason:", chunk.final.finish_reason)
            print("server_usage:", chunk.final.usage)
            print(f"client_observed_total_ms={dt_ms:.2f}")


def main() -> None:
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)
        run_unary(stub)
        run_stream(stub)


if __name__ == "__main__":
    main()