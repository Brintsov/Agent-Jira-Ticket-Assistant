from smolagents import Tool
from typing import Dict, Any, Optional


class SemanticTicketSearchTool(Tool):
    name = "semantic_ticket_search"
    description = (
        "Search Jira tickets by semantic similarity using natural-language meaning rather than exact field matching.\n\n"

        "Use this tool when the user describes a topic, issue pattern, symptom, or concept in natural language,\n"
        "but does not provide strong structured constraints such as exact project, status, resolution, or issue type.\n\n"

        "This tool is designed for fuzzy meaning-based retrieval. It finds tickets that are conceptually related\n"
        "to the query even when the same exact words do not appear in the ticket text.\n\n"

        "Best use cases:\n"
        '- "Find tickets about broken authentication flow"\n'
        '- "Show issues similar to SQL deadlocks"\n'
        '- "Find problems related to deployment instability"\n'
        '- "Look for tickets about timeout errors in background jobs"\n'
        '- "Find issues related to log in redirect failures"\n\n'

        "Prefer this tool when:\n"
        "- the user expresses a problem in natural language\n"
        "- the request is thematic or conceptual\n"
        "- exact metadata filters are missing or unknown\n"
        "- simple keyword matching would likely miss relevant tickets\n\n"

        "Do not use this tool when:\n"
        "- the request is purely structured and exact; use exact search instead\n"
        "- the request combines structured filters with semantic meaning; use hybrid search instead\n\n"

        "Notes for agent use:\n"
        "- Use this tool for fuzzy or meaning-based retrieval\n"
        '- Use this tool when the user asks for tickets "about" something rather than specifying exact metadata filters\n'
        "- Returned results are semantic retrieval hits, not necessarily full hydrated SQL rows\n"
        "- If the user also gives exact filters like project or status, prefer hybrid search"
    )

    inputs = {
        "semantic_query": {
            "type": "string",
            "description": (
                "Natural-language semantic query describing the issue, topic, concept, or failure pattern "
                "to search for, for example 'broken login flow', 'SQL deadlocks', or "
                "'deployment instability after release'."
            ),
        },
        "k": {
            "type": "integer",
            "description": "Maximum number of semantically relevant results to return.",
            "default": 5,
            "nullable": False,
        },
    }

    output_type = "object"

    def __init__(self, ticket_repository, default_k: int = 5):
        super().__init__()
        self.repo = ticket_repository
        self.default_k = default_k

    def forward(self, semantic_query: str, k: Optional[int] = None) -> Dict[str, Any]:
        if k is None:
            k = self.default_k

        if not semantic_query or not semantic_query.strip():
            return {
                "count": 0,
                "results": [],
                "search_type": "semantic",
                "error": "query must be a non-empty string",
            }

        if k <= 0:
            return {
                "count": 0,
                "results": [],
                "search_type": "semantic",
                "error": "k must be greater than 0",
            }

        hits = self.repo.semantic_search(query=semantic_query, k=k)

        return {
            "count": len(hits),
            "results": hits,
            "search_type": "semantic",
            "query": semantic_query,
        }
