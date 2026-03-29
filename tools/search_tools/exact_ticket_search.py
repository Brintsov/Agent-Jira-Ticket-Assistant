from smolagents import Tool
from typing import Dict, Any, Optional


class ExactTicketSearchTool(Tool):
    name = "exact_ticket_search"
    description = (
        "Search Jira tickets using structured filters such as project, status, resolution, issue type, "
        "priority, and optional plain text contains filters.\n\n"

        "This is the default tool for standard ticket retrieval.\n\n"

        "Use this tool for requests like:\n"
        '- "Find 50 tickets for project BEAM"\n'
        '- "Show open bugs in IGNITE"\n'
        '- "Find high priority issues"\n'
        '- "Find tickets with login in the summary"\n\n'

        "Use this tool whenever the user specifies normal filters and does not ask for semantic similarity.\n\n"

        "Do not use issue ID lookup unless the user explicitly provides full Jira issue IDs such as BEAM-123."
    )

    inputs = {
        "project_key": {"type": "string", "description": "Exact Jira project key.", "nullable": True},
        "status_name": {"type": "string", "description": "Exact status name.", "nullable": True},
        "resolution_name": {"type": "string", "description": "Exact resolution name.", "nullable": True},
        "issue_type_name": {"type": "string", "description": "Exact issue type name.", "nullable": True},
        "priority_name": {"type": "string", "description": "Exact priority name.", "nullable": True},
        "summary_contains": {"type": "string", "description": "Substring filter for summary.", "nullable": True},
        "description_contains": {"type": "string", "description": "Substring filter for description.", "nullable": True},
        "limit": {"type": "integer", "description": "Maximum number of results.", "default": 20, "nullable": True},
    }

    output_type = "object"

    def __init__(self, ticket_repository, default_limit: int = 20):
        super().__init__()
        self.repo = ticket_repository
        self.default_limit = default_limit

    def forward(
        self,
        project_key: Optional[str] = None,
        status_name: Optional[str] = None,
        resolution_name: Optional[str] = None,
        issue_type_name: Optional[str] = None,
        priority_name: Optional[str] = None,
        summary_contains: Optional[str] = None,
        description_contains: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:

        limit = limit or self.default_limit

        if limit <= 0:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "exact",
                "error": "limit must be greater than 0",
            }

        rows = self.repo.filter_tickets(
            project_key=project_key,
            status_name=status_name,
            resolution_name=resolution_name,
            issue_type_name=issue_type_name,
            priority_name=priority_name,
            summary_contains=summary_contains,
            description_contains=description_contains,
            limit=limit,
        )

        return {
            "count": len(rows),
            "tickets": rows,
            "search_type": "exact",
        }
