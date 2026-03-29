from smolagents import Tool
from typing import Dict, Any
from tools.utils import build_analysis_context


class AnalyzeTicketDistributionTool(Tool):
    name = "analyze_ticket_distribution"
    description = (
        "Analyze the structural distribution of Jira tickets across key metadata dimensions "
        "and explain the result in natural language.\n\n"

        "This tool is designed for post-retrieval use. It consumes the output of one of the "
        "Jira retrieval tools, builds a deterministic statistical overview of the ticket set, "
        "and then uses the language model to explain that overview clearly.\n\n"

        "Unlike analyze ticket patterns, which focuses on recurring textual or thematic signals, "
        "this tool focuses on structured metadata distribution such as:\n"
        "- status\n"
        "- resolution\n"
        "- priority\n"
        "- issue type\n"
        "- project\n"
        "- labels\n\n"

        "Use this tool when the user asks questions such as:\n"
        '- "What is the status breakdown?"\n'
        '- "How many of these are still open?"\n'
        '- "Show the priority distribution"\n'
        '- "What is the current situation in our database?"\n'
        '- "What types of tickets are in this result set?"\n'
        '- "Give me a structured overview of these tickets"\n\n'

        "Best use cases:\n"
        "- operational overviews\n"
        "- factual ticket set summaries explained in plain language\n"
        "- supporting prioritization discussions with clear counts\n"
        "- understanding whether a result set is concentrated in one area or spread across many\n\n"

        "Do not use this tool as the first retrieval step.\n"
        "First retrieve tickets using a ticket retrieval tool, then analyze the returned payload.\n\n"

        "Behavior:\n"
        "- builds a shared normalized analysis context using build_analysis_context\n"
        "- reuses deterministic counters and metrics prepared there\n"
        "- uses the language model only to explain the statistics clearly\n"
        "- stays grounded in the computed evidence\n\n"

        "Notes for agent use:\n"
        "- This tool is grounded in the search payload and derived metrics\n"
        '- For semantic search results, some metadata may be sparse or missing; such values may be normalized to "Unknown"\n'
        "- This tool should explain the distribution, not invent root causes"
    )

    inputs = {
        "search_payload": {
            "type": "object",
            "description": "Dictionary returned by a Jira retrieval tool.",
        }
    }

    output_type = "string"

    def __init__(self, used_llm_model):
        super().__init__()
        self.llm_model = used_llm_model

    def forward(self, search_payload: Dict[str, Any]) -> str:
        if not isinstance(search_payload, dict):
            return "Invalid search payload: expected a dictionary."

        context = build_analysis_context(search_payload)

        if not context.get("ok", False):
            return (
                "I could not analyze the ticket distribution because the provided payload "
                f"was not usable. Error: {context.get('error', 'Unknown error')}."
            )

        if context["count"] == 0:
            return "No matching tickets were available for distribution analysis."

        evidence_bundle = {
            "count": context["count"],
            "search_type": context["search_type"],
            "distribution": context["distribution"],
            "high_level_metrics": {
                **context["status_signals"],
                **context["high_level_metrics"],
            },
            "top_labels": context["top_labels"],
            "top_projects": context["top_projects"],
        }

        prompt = (
            "You are a careful Jira analysis assistant.\n\n"
            "Your task is to explain the structural distribution of a set of Jira tickets.\n"
            "Use only the evidence provided.\n"
            "Do not invent causes, team ownership, incidents, or technical explanations.\n\n"
            "Requirements:\n"
            "1. Start with the main high-level takeaway in 1-2 sentences.\n"
            "2. Then explain the most important distribution signals, especially status, priority, issue type, and project concentration.\n"
            "3. Mention whether the result set looks concentrated or spread out.\n"
            "4. Mention open-like, resolved-like, and high-priority-like signals when relevant.\n"
            "5. Keep it natural, concise, and grounded in the numbers.\n"
            "6. If the evidence is limited or uneven, say so.\n\n"
            "Structured evidence:\n"
            f"{evidence_bundle}"
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        response = self.llm_model(messages)
        return getattr(response, "content", str(response))
