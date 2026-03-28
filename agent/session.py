import time
from typing import Iterable

from observation.logging import (
    estimate_tokens,
    generate_request_id,
    log_event,
    reset_request_id,
    set_request_id,
)


class AgentSession:
    """Maintains conversation reset policy for agent turns"""

    def __init__(self) -> None:
        self._is_first_turn = True

    def run(self, agent, prompt: str, *, reset: bool | None = None) -> str:
        request_id = generate_request_id()
        request_ctx = set_request_id(request_id)
        effective_reset = self._is_first_turn if reset is None else reset
        start = time.perf_counter()
        log_event(
            "request.start",
            request_id=request_id,
            reset=effective_reset,
            prompt_tokens_est=estimate_tokens(prompt),
        )
        try:
            response = agent.run(prompt, reset=effective_reset)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log_event(
                "request.end",
                request_id=request_id,
                latency_ms=latency_ms,
                response_tokens_est=estimate_tokens(str(response)),
            )
            self._is_first_turn = False
            return response
        finally:
            reset_request_id(request_ctx)

    def run_many(
        self,
        agent,
        prompts: Iterable[str],
        *,
        reset_first: bool = True,
    ) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for i, prompt in enumerate(prompts):
            response = self.run(agent, prompt, reset=(reset_first if i == 0 else False))
            results.append((prompt, response))
        return results

    def reset(self) -> None:
        self._is_first_turn = True
