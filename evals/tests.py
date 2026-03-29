import argparse
import re
import json
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # noqa: BLE001
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTING_FILE = REPO_ROOT / "evals" / "routing_cases.yaml"
GROUNDING_FILE = REPO_ROOT / "evals" / "grounding_cases.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TICKET_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        content = fh.read()

    if yaml is not None:
        return yaml.safe_load(content)

    # Files are intentionally JSON-compatible YAML so this fallback works without PyYAML.
    return json.loads(content)


def _extract_ticket_keys(text: str) -> set[str]:
    return set(TICKET_KEY_PATTERN.findall(text or ""))


def check_grounding_case(case: Dict[str, Any]) -> tuple[bool, str]:
    payload_keys = set(case.get("payload_keys", []))
    answer_keys = _extract_ticket_keys(case.get("answer", ""))
    is_grounded = answer_keys.issubset(payload_keys)
    expected = bool(case.get("must_be_grounded", True))

    if is_grounded == expected:
        return True, f"{case.get('id')}: PASS"

    return (
        False,
        (
            f"{case.get('id')}: FAIL (payload_keys={sorted(payload_keys)}, "
            f"answer_keys={sorted(answer_keys)}, expected_grounded={expected})"
        ),
    )


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
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


def run_live_routing_cases(routing_cases: list[Dict[str, Any]]) -> int:
    """Run routing cases through the real pipeline and validate selected tools from logs."""
    from jira_knowledge_expert import JiraAssistantPipeline, PipelineConfig
    from observation.logging import get_log_file_path

    pipeline = JiraAssistantPipeline(PipelineConfig())
    log_path = get_log_file_path()
    before = _read_jsonl(log_path)
    before_count = len(before)

    for case in routing_cases:
        pipeline.run(case["query"], reset=True)

    after = _read_jsonl(log_path)
    new_events = after[before_count:]

    request_start_by_id: Dict[str, Dict[str, Any]] = {}
    request_end_by_id: Dict[str, Dict[str, Any]] = {}
    for event in new_events:
        event_name = event.get("event")
        request_id = event.get("request_id")
        if not request_id:
            continue
        if event_name == "request.start":
            request_start_by_id[request_id] = event
        elif event_name == "request.end":
            request_end_by_id[request_id] = event

    prompt_to_tools: Dict[str, list[str]] = {}
    for request_id, start_event in request_start_by_id.items():
        end_event = request_end_by_id.get(request_id)
        if not end_event:
            continue
        prompt = start_event.get("prompt")
        if isinstance(prompt, str):
            prompt_to_tools[prompt] = end_event.get("tools_used", [])

    failed = False
    for case in routing_cases:
        used_tools = prompt_to_tools.get(case["query"], [])
        expected = case["expected_tool"]
        if expected in used_tools:
            print(f"{case['id']}: PASS (expected={expected}, used_tools={used_tools})")
        else:
            print(f"{case['id']}: FAIL (expected={expected}, used_tools={used_tools})")
            failed = True
    return 1 if failed else 0


def run(live_routing: bool = False) -> int:
    routing = _load_yaml(ROUTING_FILE)
    grounding = _load_yaml(GROUNDING_FILE)

    print(f"Loaded {len(routing.get('cases', []))} routing case(s) from {ROUTING_FILE}")
    print(f"Loaded {len(grounding.get('cases', []))} grounding case(s) from {GROUNDING_FILE}")

    # Routing pack is declarative by design; execution is done by whichever router harness is used.
    # Here we at least enforce schema presence.
    required_routing = {"id", "query", "expected_tool"}
    for case in routing.get("cases", []):
        missing = required_routing - set(case.keys())
        if missing:
            print(f"{case.get('id', '<missing-id>')}: FAIL missing fields {sorted(missing)}")
            return 1

    failed = False
    for case in grounding.get("cases", []):
        ok, message = check_grounding_case(case)
        print(message)
        if not ok:
            failed = True

    if live_routing:
        try:
            live_status = run_live_routing_cases(routing.get("cases", []))
            if live_status != 0:
                failed = True
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"live_routing: FAIL ({exc})")
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run routing/grounding eval checks.")
    parser.add_argument(
        "--live-routing",
        action="store_true",
        help="Execute routing cases through JiraAssistantPipeline and validate chosen tools from observability logs.",
    )
    args = parser.parse_args()
    raise SystemExit(run(live_routing=args.live_routing))
