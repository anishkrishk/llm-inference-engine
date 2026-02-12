from __future__ import annotations

import random
from typing import Iterator, List

from src.backends.base import Backend, GenerateInput, GenerateResult, TokenEvent


class PytorchEagerBackend(Backend):
    """
    This is a temporary mock backend.
    I'll keep class name aligned with the future real backend that'll be more eager.
    """

    def generate(self, req: GenerateInput) -> GenerateResult:
        events = list(self.generate_stream(req))
        return self.finalize_stream(req, events)

    def generate_stream(self, req: GenerateInput) -> Iterator[TokenEvent]:
        random.seed(req.seed)

        base_tokens = ["This", "is", "a", "mock", "response", "for", "now", "."]
        n = min(max(req.max_new_tokens, 1), 64)

        for i in range(n):
            token_text = base_tokens[i % len(base_tokens)]
            # fake token id for now
            token_id = (hash(token_text) + i) % 32000
            yield TokenEvent(token_id=token_id, token_text=token_text, token_index=i)

            # simple stop handling
            if any(stop and stop in token_text for stop in req.stop_sequences):
                break

    def finalize_stream(self, req: GenerateInput, events: List[TokenEvent]) -> GenerateResult:
        text = " ".join(e.token_text for e in events).strip()
        token_ids = [e.token_id for e in events]
        prompt_tokens = max(1, len(req.prompt.split()))
        generated_tokens = len(events)

        finish_reason = "stop" if generated_tokens < req.max_new_tokens else "length"
        return GenerateResult(
            text=text,
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
        )