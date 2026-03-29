from smolagents import Tool
from typing import Dict, Any
from tools.utils import build_analysis_context


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
