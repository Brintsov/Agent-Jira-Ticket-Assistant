import os
import unittest

import json
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


def load_cases(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        content = fh.read()
    if yaml is not None:
        return yaml.safe_load(content)
    return json.loads(content)


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def events_by_request(new_events: list[Dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_request: dict[str, dict[str, Any]] = {}
    for event in new_events:
        request_id = event.get("request_id")
        if not request_id:
            continue

        info = by_request.setdefault(
            request_id,
            {"prompt": None, "request_end": None, "tool_starts": []},
        )

        event_name = event.get("event")
        if event_name == "request.start":
            info["prompt"] = event.get("prompt")
        elif event_name == "request.end":
            info["request_end"] = event
        elif event_name == "tool.start":
            info["tool_starts"].append(event)

    return by_request


def find_case_execution(prompt: str, request_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for req in request_map.values():
        if req.get("prompt") == prompt:
            return req
    return None


def is_param_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


def validate_expected_call(case: Dict[str, Any], tool_starts: list[Dict[str, Any]]) -> tuple[bool, str]:
    expected_tool = case["expected_tool"]
    expected_call = case.get("expected_call") or {}

    for event in tool_starts:
        if event.get("tool") != expected_tool:
            continue

        call = event.get("call") or {}
        kwargs = call.get("kwargs") or {}
        missing = []
        mismatched = []

        for param_name, expected_value in expected_call.items():
            if param_name not in kwargs:
                missing.append(param_name)
                continue
            actual_value = kwargs[param_name]
            if not is_param_match(expected_value, actual_value):
                mismatched.append((param_name, expected_value, actual_value))

        if not missing and not mismatched:
            return True, (
                f"{case['id']}: PASS (expected_tool={expected_tool}, "
                f"expected_call={expected_call})"
            )

        return False, (
            f"{case['id']}: FAIL tool call mismatch "
            f"(missing={missing}, mismatched={mismatched}, kwargs={kwargs})"
        )

    return False, f"{case['id']}: FAIL expected tool '{expected_tool}' was never called"


# @unittest.skipUnless(os.getenv("RUN_LIVE_ROUTING_EVAL") == "1", "Set RUN_LIVE_ROUTING_EVAL=1 to run live routing evaluation")
class LiveRoutingTests(unittest.TestCase):
    def test_live_routing_cases(self):
        from jira_knowledge_expert import JiraAssistantPipeline, PipelineConfig
        from observation.logger import get_log_file_path

        routing_file = Path(__file__).resolve().parents[1] / "evals" / "routing_cases.yaml"
        routing = load_cases(routing_file)
        cases = routing.get("cases", [])
        pipeline = JiraAssistantPipeline(PipelineConfig())
        log_path = get_log_file_path()
        before = read_jsonl(log_path)
        before_count = len(before)

        for case in cases:
            pipeline.run(case["query"], reset=True)

        after = read_jsonl(log_path)
        new_events = after[before_count:]
        request_map = events_by_request(new_events)

        failures = []
        for case in cases:
            executed = find_case_execution(case["query"], request_map)
            if not executed:
                failures.append(f"{case['id']}: missing request trace")
                continue

            request_end = executed.get("request_end") or {}
            used_tools = request_end.get("tools_used", [])
            expected_tool = case["expected_tool"]
            if expected_tool not in used_tools:
                failures.append(
                    f"{case['id']}: expected_tool={expected_tool}, used_tools={used_tools}"
                )
                continue

            ok, message = validate_expected_call(case, executed.get("tool_starts") or [])
            if not ok:
                failures.append(message)

        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
