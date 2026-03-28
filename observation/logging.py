import json
import logging
import os
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4


_current_request_id: ContextVar[str | None] = ContextVar("current_request_id", default=None)
_logger = logging.getLogger("jira_agent_observability")
_default_log_path = Path("logs/agent_observability.jsonl")


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
    return _current_request_id.set(request_id)


def reset_request_id(token) -> None:
    _current_request_id.reset(token)


def get_request_id() -> str | None:
    return _current_request_id.get()


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
