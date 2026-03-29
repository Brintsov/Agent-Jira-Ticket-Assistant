from smolagents import Tool
from typing import Optional, Dict, Any
from tools.utils import extract_search_results, create_ticket_block


class DiscussTicketFindingsTool(Tool):
    name = "discuss_ticket_findings"
    description = (
        "Provide a grounded conversational interpretation of Jira search results.\n\n"

        "This tool is intended for user-facing explanation after ticket retrieval.\n"
        "It can express a reasoned view or practical judgment, but it must stay\n"
        "anchored in the provided ticket data and clearly distinguish:\n"
        "- factual observations from the tickets\n"
        "- interpretation or opinion based on those observations\n\n"

        "Use this tool when:\n"
        "- the user wants a natural explanation rather than a dry summary\n"
        "- the user asks questions like:\n"
        '  "Does this seem serious?"\n'
        '  "What do you think is happening here?"\n'
        '  "Which of these looks most important?"\n'
        '  "Would you consider this a recurring issue?"\n'
        "- the agent needs to provide a grounded recommendation\n\n"

        "Do not use this tool before retrieval.\n"
        "First retrieve tickets using a search tool, then interpret the results.\n\n"

        "Notes for agent use:\n"
        "- This tool goes beyond summarization and includes structured reasoning\n"
        "- It should remain faithful to the data and avoid hallucination\n"
        "- It is best used when the user asks for meaning, importance, or judgment\n"
    )

    inputs = {
        "search_payload": {
            "type": "object",
            "description": "Dictionary returned by a ticket retrieval tool.",
        },
        "user_question": {
            "type": "string",
            "description": "Optional original user question to guide the explanation.",
            "nullable": True,
        },
        "tone": {
            "type": "string",
            "description": "Style of response. Examples: conversational, concise, analyst.",
            "default": "conversational",
            "nullable": True,
        },
    }

    output_type = "string"

    def __init__(self, used_llm_model):
        super().__init__()
        self.llm_model = used_llm_model

    def forward(
        self,
        search_payload: Dict[str, Any],
        user_question: Optional[str] = None,
        tone: Optional[str] = "conversational",
    ) -> str:

        if not isinstance(search_payload, dict):
            return "Invalid search payload: expected a dictionary."

        normalized_tickets = extract_search_results(search_payload)

        if not normalized_tickets:
            return "No matching tickets found to analyze."

        ticket_blocks = create_ticket_block(normalized_tickets)

        prompt = (
            "You are a careful, conversational Jira assistant.\n\n"
            "Answer the user like a helpful colleague.\n\n"

            "Use this structure:\n"
            "1. Direct answer to the user's question\n"
            "2. Evidence from the tickets\n"
            "3. Your interpretation (clearly label if speculative)\n"
            "4. A useful next step or follow-up question\n\n"

            "Rules:\n"
            "- Keep it natural and not robotic\n"
            "- Avoid raw metadata unless it helps\n"
            "- If the evidence is weak, say so\n"
            "- Mention ticket keys when relevant\n\n"

            f"Requested tone: {tone or 'conversational'}\n"
            f"User question: {user_question or 'Not provided'}\n\n"

            "Tickets:\n\n"
            + "\n\n---\n\n".join(ticket_blocks)
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]

        response = self.llm_model(messages)
        return getattr(response, "content", str(response))
