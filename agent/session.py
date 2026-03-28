from typing import Iterable


class AgentSession:
    """Maintains conversation reset policy for agent turns"""

    def __init__(self) -> None:
        self._is_first_turn = True

    def run(self, agent, prompt: str, *, reset: bool | None = None) -> str:
        effective_reset = self._is_first_turn if reset is None else reset
        response = agent.run(prompt, reset=effective_reset)
        self._is_first_turn = False
        return response

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
