from smolagents import Tool
from typing import Dict, Any, Optional, List


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


class HybridTicketSearchTool(Tool):
    name = "hybrid_ticket_search"
    description = (
        "Search Jira tickets using a combination of exact structured filtering and semantic similarity.\n\n"

        "Use this tool when the user asks for tickets that match both:\n"
        "1. explicit structured constraints such as project, status, resolution, issue type, or priority\n"
        "2. a topic, concept, problem pattern, or meaning-based description\n\n"

        "This tool performs a hybrid retrieval flow:\n"
        "- first narrows the candidate set using structured filters\n"
        "- then applies semantic search within that candidate set\n"
        "- then returns full ticket rows for the best-matching tickets\n\n"

        "Best use cases:\n"
        '- "Find resolved IGNITE bugs about SQL deadlocks"\n'
        '- "Show high priority WW tickets related to authentication failures"\n'
        '- "Find Done improvements in project CORE related to deployment instability"\n'
        '- "Find open bugs in PAYMENTS similar to timeout issues"\n\n'

        "Prefer this tool when the user combines exact filters with conceptual language.\n\n"

        "Do not use this tool when:\n"
        "- the request is purely exact and does not need semantic matching; use exact search instead\n"
        "- the request is purely fuzzy and has no structured constraints; use semantic search instead\n\n"

        "Notes for agent use:\n"
        "- Use this tool when the user mixes exact attributes and fuzzy topic language\n"
        '- This is usually the best tool for requests like "find bugs in project X about Y"\n'
        "- The returned tickets are full structured rows and are suitable for summarization\n"
        "- This tool is often a better first choice than manually chaining exact and semantic search"
    )

    inputs = {
        "semantic_query": {
            "type": "string",
            "description": (
                "Natural-language description of the issue theme, concept, or problem pattern "
                "to search for semantically, for example 'SQL deadlocks', 'broken login flow', "
                "or 'deployment instability'."
            ),
        },
        "project_key": {
            "type": "string",
            "description": "Exact Jira project key used to narrow candidates before semantic ranking.",
            "nullable": True,
        },
        "status_name": {
            "type": "string",
            "description": "Exact status name used to narrow candidates before semantic ranking.",
            "nullable": True,
        },
        "resolution_name": {
            "type": "string",
            "description": "Exact resolution name used to narrow candidates before semantic ranking.",
            "nullable": True,
        },
        "issue_type_name": {
            "type": "string",
            "description": "Exact issue type name used to narrow candidates before semantic ranking.",
            "nullable": True,
        },
        "priority_name": {
            "type": "string",
            "description": "Exact priority name used to narrow candidates before semantic ranking.",
            "nullable": True,
        },
        "summary_contains": {
            "type": "string",
            "description": (
                "Optional case-insensitive substring that must appear in the summary "
                "before semantic ranking."
            ),
            "nullable": True,
        },
        "description_contains": {
            "type": "string",
            "description": (
                "Optional case-insensitive substring that must appear in the description "
                "before semantic ranking."
            ),
            "nullable": True,
        },
        "limit": {
            "type": "integer",
            "description": (
                "Maximum number of structured candidate tickets to consider before semantic reranking."
            ),
            "default": 20,
            "nullable": True,
        },
        "k": {
            "type": "integer",
            "description": (
                "Maximum number of semantically best-matching tickets to return after reranking."
            ),
            "default": 5,
            "nullable": True,
        },
    }

    output_type = "object"

    def __init__(
        self,
        ticket_repository,
        default_limit: int = 20,
        default_k: int = 5,
    ):
        super().__init__()
        self.repo = ticket_repository
        self.default_limit = default_limit
        self.default_k = default_k

    def forward(
        self,
        semantic_query: str,
        project_key: Optional[str] = None,
        status_name: Optional[str] = None,
        resolution_name: Optional[str] = None,
        issue_type_name: Optional[str] = None,
        priority_name: Optional[str] = None,
        summary_contains: Optional[str] = None,
        description_contains: Optional[str] = None,
        limit: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:

        limit = limit or self.default_limit
        k = k or self.default_k

        if not semantic_query or not semantic_query.strip():
            return {
                "count": 0,
                "tickets": [],
                "search_type": "hybrid",
                "error": "semantic_query must be a non-empty string",
            }

        if limit <= 0:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "hybrid",
                "error": "limit must be greater than 0",
            }

        if k <= 0:
            return {
                "count": 0,
                "tickets": [],
                "search_type": "hybrid",
                "error": "k must be greater than 0",
            }

        rows = self.repo.hybrid_search(
            semantic_query=semantic_query,
            sql_filters={
                "project_key": project_key,
                "status_name": status_name,
                "resolution_name": resolution_name,
                "issue_type_name": issue_type_name,
                "priority_name": priority_name,
                "summary_contains": summary_contains,
                "description_contains": description_contains,
                "limit": limit,
            },
            k=k,
        )

        return {
            "count": len(rows),
            "tickets": rows,
            "search_type": "hybrid",
            "semantic_query": semantic_query,
        }


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
