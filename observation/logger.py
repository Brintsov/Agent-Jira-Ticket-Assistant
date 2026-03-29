import os
import time
import json
import logging
from uuid import uuid4
from pathlib import Path
from typing import Any, Dict
from contextvars import ContextVar


class ObservabilityLogger:
    """Wrapper aroung logging/context handling."""

    _request_id_env_var = "JIRA_AGENT_REQUEST_ID"

    def __init__(self, name: str = "jira_agent_observability", default_path: str = "logs/agent_observability.jsonl") -> None:
        self._logger = logging.getLogger(name)
        self._default_log_path = Path(default_path)
        self._current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)
        self._current_used_tools: ContextVar[list[str]] = ContextVar("current_used_tools", default=[])
        self._current_session_id: ContextVar[str | None] = ContextVar("current_session_id", default=None)
        self._current_turn_index: ContextVar[int] = ContextVar("current_turn_index", default=0)
        self._configure_logger_once()

    def _configure_logger_once(self) -> None:
        if self._logger.handlers:
            return

        log_path = self.get_log_file_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(file_handler)

        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def get_log_file_path(self) -> Path:
        configured = os.getenv("JIRA_AGENT_OBSERVABILITY_LOG")
        if configured:
            return Path(configured)
        return self._default_log_path

    @staticmethod
    def generate_request_id() -> str:
        return str(uuid4())

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid4())

    def set_request_id(self, request_id: str):
        os.environ[self._request_id_env_var] = request_id
        return self._current_request_id.set(request_id)

    def reset_request_id(self, token) -> None:
        self._current_request_id.reset(token)
        os.environ.pop(self._request_id_env_var, None)

    def get_request_id(self) -> str | None:
        return self._current_request_id.get() or os.getenv(self._request_id_env_var)

    def set_session_id(self, session_id: str):
        return self._current_session_id.set(session_id)

    def reset_session_id(self, token) -> None:
        self._current_session_id.reset(token)

    def get_session_id(self) -> str | None:
        return self._current_session_id.get()

    def set_turn_index(self, turn_index: int):
        return self._current_turn_index.set(turn_index)

    def reset_turn_index(self, token) -> None:
        self._current_turn_index.reset(token)

    def get_turn_index(self) -> int:
        return self._current_turn_index.get()

    def reset_used_tools(self) -> None:
        self._current_used_tools.set([])

    def append_used_tool(self, tool_name: str) -> None:
        tools = list(self._current_used_tools.get())
        tools.append(tool_name)
        self._current_used_tools.set(tools)

    def get_used_tools(self) -> list[str]:
        return list(self._current_used_tools.get())

    def get_logged_tools_for_request(self, request_id: str) -> list[str]:
        """Read persisted JSONL events and return tools observed for a request."""
        path = self.get_log_file_path()
        if not path.exists():
            return []

        tools: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("request_id") != request_id:
                    continue
                if event.get("event") == "tool.start":
                    tool_name = event.get("tool")
                    if isinstance(tool_name, str) and tool_name:
                        tools.append(tool_name)
        return tools

    @staticmethod
    def estimate_tokens(text: str | None) -> int:
        if not text:
            return 0
        return max(1, int(len(text.split()) * 1.3))

    def log_event(self, event: str, **fields: Any) -> None:
        prompt_text = fields.get("prompt")
        if isinstance(prompt_text, str):
            fields.setdefault("prompt_chars", len(prompt_text))

        payload: Dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            "request_id": fields.pop("request_id", self.get_request_id()),
            "session_id": fields.pop("session_id", self.get_session_id()),
            "turn_index": fields.pop("turn_index", self.get_turn_index()),
            "pid": os.getpid(),
            **fields,
        }
        self._logger.info(json.dumps(payload, default=str, ensure_ascii=False))


_observability = ObservabilityLogger()


def get_log_file_path() -> Path:
    return _observability.get_log_file_path()


def generate_request_id() -> str:
    return _observability.generate_request_id()


def generate_session_id() -> str:
    return _observability.generate_session_id()


def set_request_id(request_id: str):
    return _observability.set_request_id(request_id)


def reset_request_id(token) -> None:
    _observability.reset_request_id(token)


def get_request_id() -> str | None:
    return _observability.get_request_id()


def set_session_id(session_id: str):
    return _observability.set_session_id(session_id)


def reset_session_id(token) -> None:
    _observability.reset_session_id(token)


def get_session_id() -> str | None:
    return _observability.get_session_id()


def set_turn_index(turn_index: int):
    return _observability.set_turn_index(turn_index)


def reset_turn_index(token) -> None:
    _observability.reset_turn_index(token)


def get_turn_index() -> int:
    return _observability.get_turn_index()


def reset_used_tools() -> None:
    _observability.reset_used_tools()


def append_used_tool(tool_name: str) -> None:
    _observability.append_used_tool(tool_name)


def get_used_tools() -> list[str]:
    return _observability.get_used_tools()


def get_logged_tools_for_request(request_id: str) -> list[str]:
    return _observability.get_logged_tools_for_request(request_id)


def estimate_tokens(text: str | None) -> int:
    return _observability.estimate_tokens(text)


def log_event(event: str, **fields: Any) -> None:
    _observability.log_event(event, **fields)
