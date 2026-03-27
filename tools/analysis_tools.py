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


class AnalyzeTicketPatternsTool(Tool):
    name = "analyze_ticket_patterns"
    description = (
        "Analyze Jira tickets to identify recurring textual and thematic patterns.\n\n"

        "This tool is intended for post-retrieval use. It consumes the output of one of the\n"
        "Jira retrieval tools and identifies repeated problem themes, shared symptoms, and\n"
        "common operational signals across the returned ticket set.\n\n"

        "Best use cases:\n"
        '- "What patterns do you see here?"\n'
        '- "What do these tickets have in common?"\n'
        '- "Does this look like a recurring issue?"\n'
        '- "What are the main themes across these tickets?"\n'
        '- "Are these mostly the same kind of problem?"\n\n'

        "Behavior:\n"
        "- first builds a shared analytical context using build_analysis_context\n"
        "- then uses deterministic signals such as keywords, bigrams, labels, projects, and ticket metadata\n"
        "- then asks the model to produce a grounded thematic interpretation\n\n"

        "Use this tool when:\n"
        "- the user wants cross-ticket pattern detection rather than a simple summary\n"
        "- the agent needs to identify repeated symptoms, domains, or operational clusters\n"
        "- the user asks whether several tickets appear related or recurring\n\n"

        "Do not use this tool before retrieval.\n"
        "First retrieve tickets using a ticket search tool, then analyze the returned payload.\n\n"

        "Notes for agent use:\n"
        "- The tool stays grounded in evidence derived from the ticket payload\n"
        "- It should not invent causes, ownership, incidents, or technical details not present in the tickets\n"
        "- This tool is better for recurring-theme analysis than ordinary summarization"
    )

    inputs = {
        "search_payload": {
            "type": "object",
            "description": "Dictionary returned by a Jira retrieval tool.",
        }
    }

    output_type = "object"

    def __init__(self, used_llm_model):
        super().__init__()
        self.llm_model = used_llm_model

    def forward(self, search_payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(search_payload, dict):
            return {
                "count": 0,
                "search_type": "unknown",
                "top_keywords": [],
                "top_bigrams": [],
                "status_signals": {
                    "open_like": 0,
                    "resolved_like": 0,
                    "high_priority_like": 0,
                },
                "patterns": [],
                "main_observation": "",
                "analysis_quality": "weak",
                "error": "search_payload must be a dictionary",
            }

        context = build_analysis_context(search_payload)

        if not context.get("ok", False):
            return {
                "count": 0,
                "search_type": context.get("search_type", "unknown"),
                "top_keywords": [],
                "top_bigrams": [],
                "top_labels": [],
                "top_projects": [],
                "status_signals": {
                    "open_like": 0,
                    "resolved_like": 0,
                    "high_priority_like": 0,
                },
                "distribution": {
                    "status": {},
                    "priority": {},
                    "issue_type": {},
                },
                "patterns": [],
                "main_observation": "",
                "analysis_quality": "weak",
                "error": context.get("error", "Unknown error"),
            }

        if context["count"] == 0:
            return {
                "count": 0,
                "search_type": context.get("search_type", "unknown"),
                "top_keywords": [],
                "top_bigrams": [],
                "top_labels": [],
                "top_projects": [],
                "status_signals": {
                    "open_like": 0,
                    "resolved_like": 0,
                    "high_priority_like": 0,
                },
                "distribution": {
                    "status": {},
                    "priority": {},
                    "issue_type": {},
                },
                "patterns": [],
                "main_observation": "No matching tickets were available for pattern analysis.",
                "analysis_quality": "weak",
                "model_analysis": "No matching tickets were available for pattern analysis.",
            }

        evidence_bundle = {
            "top_keywords": context["top_keywords"],
            "top_bigrams": context["top_bigrams"],
            "top_labels": context["top_labels"],
            "top_projects": context["top_projects"],
            "status_signals": context["status_signals"],
            "distribution": {
                "status": context["distribution"]["status"],
                "priority": context["distribution"]["priority"],
                "issue_type": context["distribution"]["issue_type"],
            },
        }

        prompt = (
            "You are a careful Jira analysis assistant.\n\n"
            "Your task is to identify recurring patterns across a set of Jira tickets.\n"
            "Use only the evidence provided.\n"
            "Do not invent causes, teams, incidents, or architecture details that are not explicitly supported.\n\n"
            "Requirements:\n"
            "1. Produce concise recurring patterns.\n"
            "2. Each pattern must be grounded in repeated evidence from the tickets.\n"
            "3. Prefer factual themes such as repeated symptoms, repeated flows, repeated failure types, or repeated domains.\n"
            "4. If evidence is weak, say so.\n"
            "5. Start with one main observation sentence, then list the strongest patterns.\n\n"
            "Structured evidence:\n"
            f"{evidence_bundle}\n\n"
            "Tickets:\n\n"
            + "\n\n---\n\n".join(context["compact_ticket_blocks"][:20])
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        response = self.llm_model(messages)
        model_analysis = getattr(response, "content", str(response))

        return model_analysis
