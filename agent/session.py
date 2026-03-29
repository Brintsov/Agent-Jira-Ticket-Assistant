import time
from typing import Iterable

from observation.logger import (
    estimate_tokens,
    generate_request_id,
    generate_session_id,
    get_logged_tools_for_request,
    get_used_tools,
    log_event,
    reset_session_id,
    reset_turn_index,
    reset_used_tools,
    reset_request_id,
    set_session_id,
    set_turn_index,
    set_request_id,
)


class AgentSession:
    """Wrapper class to help efficiently track/manager session state"""

    def __init__(self) -> None:
        self._is_first_turn = True
        self._session_id = generate_session_id()
        self._turn_index = 0
        log_event("session.start", session_id=self._session_id, turn_index=self._turn_index)

    def run(self, agent, prompt: str, *, reset: bool | None = None) -> str:
        request_id = generate_request_id()
        self._turn_index += 1
        request_ctx = set_request_id(request_id)
        session_ctx = set_session_id(self._session_id)
        turn_ctx = set_turn_index(self._turn_index)
        reset_used_tools()
        effective_reset = self._is_first_turn if reset is None else reset
        start = time.perf_counter()
        log_event(
            "request.start",
            request_id=request_id,
            reset=effective_reset,
            prompt=prompt,
            prompt_tokens_est=estimate_tokens(prompt),
        )
        try:
            response = agent.run(prompt, reset=effective_reset)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            local_tools = get_used_tools()
            logged_tools = get_logged_tools_for_request(request_id)
            tools_used = local_tools or logged_tools
            log_event(
                "request.end",
                request_id=request_id,
                latency_ms=latency_ms,
                response_tokens_est=estimate_tokens(str(response)),
                tools_used=tools_used,
            )
            self._is_first_turn = False
            return response
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            log_event(
                "request.error",
                request_id=request_id,
                latency_ms=latency_ms,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        finally:
            reset_request_id(request_ctx)
            reset_session_id(session_ctx)
            reset_turn_index(turn_ctx)

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
        self._turn_index = 0
        self._session_id = generate_session_id()
        log_event("session.reset", session_id=self._session_id, turn_index=self._turn_index)
