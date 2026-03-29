from smolagents import Tool
from typing import Dict, Any, Optional


class BroadTicketSearchTool(Tool):
    name = "broad_ticket_search"
    description = (
        "Retrieve a broad, unconstrained set of Jira tickets for overview-style analysis.\n\n"

        "This tool is intended for vague or high-level user requests where no specific filters "
        "are provided, such as:\n"
        '- "Show all tickets"\n'
        '- "Analyze everything"\n'
        '- "Give me an overview of current issues"\n'
        '- "What is going on in Jira lately?"\n'
        '- "What is the status in our ticket database?"\n\n'

        "Best use cases:\n"
        "- broad exploratory analysis\n"
        "- generating status distributions and pattern summaries\n"
        "- fallback behavior when the request is too vague for exact or semantic routing\n\n"

        "Notes:\n"
        "- This tool returns a capped result set, not an unbounded full export\n"
        "- Prefer deterministic ordering such as most recent tickets first, if supported"
    )

    inputs = {
        "limit": {
            "type": "integer",
            "description": "Maximum number of tickets to retrieve.",
            "default": 50,
            "nullable": True,
        }
    }

    output_type = "object"

    def __init__(self, ticket_repository, default_limit: int = 50):
        super().__init__()
        self.repo = ticket_repository
        self.default_limit = default_limit

    def forward(self, limit: Optional[int] = None) -> Dict[str, Any]:

        limit = limit or self.default_limit

        if limit <= 0:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "broad",
                "error": "limit must be greater than 0",
            }

        try:
            rows = self.repo.filter_tickets(limit=limit)

            return {
                "count": len(rows),
                "tickets": rows,
                "search_type": "broad",
            }

        except Exception as e:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "broad",
                "error": f"broad search failed: {e}",
            }
