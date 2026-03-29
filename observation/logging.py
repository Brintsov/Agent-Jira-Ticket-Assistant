import json
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4


_current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)
_current_used_tools: ContextVar[list[str]] = ContextVar("current_used_tools", default=[])
_logger = logging.getLogger("jira_agent_observability")
_default_log_path = Path("logs/agent_observability.jsonl")
_request_id_env_var = "JIRA_AGENT_REQUEST_ID"


def get_log_file_path() -> Path:
    configured = os.getenv("JIRA_AGENT_OBSERVABILITY_LOG")
    if configured:
        return Path(configured)
    return _default_log_path


if not _logger.handlers:
    log_path = get_log_file_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(file_handler)

    _logger.setLevel(logging.INFO)
    _logger.propagate = False


def generate_request_id() -> str:
    return str(uuid4())


def set_request_id(request_id: str):
    os.environ[_request_id_env_var] = request_id
    return _current_request_id.set(request_id)


def reset_request_id(token) -> None:
    _current_request_id.reset(token)
    os.environ.pop(_request_id_env_var, None)


def get_request_id() -> str | None:
    return _current_request_id.get() or os.getenv(_request_id_env_var)


def reset_used_tools() -> None:
    _current_used_tools.set([])


def append_used_tool(tool_name: str) -> None:
    tools = list(_current_used_tools.get())
    tools.append(tool_name)
    _current_used_tools.set(tools)


def get_used_tools() -> list[str]:
    return list(_current_used_tools.get())


def get_logged_tools_for_request(request_id: str) -> list[str]:
    """Read persisted JSONL events and return tools observed for a request."""
    path = get_log_file_path()
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


def estimate_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


def log_event(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "event": event,
        "request_id": fields.pop("request_id", get_request_id()),
        **fields,
    }
    _logger.info(json.dumps(payload, default=str, ensure_ascii=False))
