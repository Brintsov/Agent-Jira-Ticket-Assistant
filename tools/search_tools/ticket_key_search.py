from smolagents import Tool
from typing import Dict, Any, List


def _parse_keys(ticket_key: str) -> List[str]:
    if not ticket_key:
        return []

    return [
        key.strip()
        for key in ticket_key.split(",")
        if key and key.strip()
    ]


class TicketKeySearchTool(Tool):
    name = "ticket_key_search"
    description = (
        "Retrieve Jira tickets by explicit Jira issue IDs only.\n\n"

        "Use this tool ONLY when the user directly provides one or more full ticket IDs "
        "in the Jira issue key format, such as BEAM-123 or CORE-456.\n\n"

        "Examples of valid use:\n"
        '- "Show BEAM-123"\n'
        '- "Find BEAM-123, BEAM-456"\n'
        '- "Get details for CORE-88"\n\n'

        "DO NOT use this tool when the user provides only a project name or project key, "
        "such as BEAM, CORE, or IGNITE, without a numeric suffix.\n"
        "DO NOT use this tool for project-level searches.\n"
        "DO NOT use this tool for status, priority, type, resolution, keyword, or semantic queries.\n\n"

        "This tool matches only complete issue IDs of the form PROJECT-NUMBER."
    )

    inputs = {
        "ticket_key": {
            "type": "string",
            "description": (
                "A single Jira ticket key or multiple ticket keys separated by commas. "
                "Examples: 'ABC-123' or 'ABC-123,ABC-456'."
            ),
        }
    }

    output_type = "object"

    def __init__(self, ticket_repository):
        super().__init__()
        self.repo = ticket_repository

    def forward(self, ticket_key: str) -> Dict[str, Any]:
        keys = _parse_keys(ticket_key)

        if not keys:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "ticket_key",
                "error": "ticket_key must contain at least one non-empty key",
            }

        try:
            rows = self.repo.get_by_keys(keys)

            return {
                "count": len(rows),
                "tickets": rows,
                "search_type": "ticket_key",
                "requested_keys": keys,
            }

        except Exception as e:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "ticket_key",
                "requested_keys": keys,
                "error": f"ticket key search failed: {e}",
            }
