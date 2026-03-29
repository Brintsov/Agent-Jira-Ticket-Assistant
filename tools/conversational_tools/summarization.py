from smolagents import Tool
from typing import Dict, Any
from tools.utils import extract_search_results, create_ticket_block


class SummarizeTicketsTool(Tool):
    name = "summarize_tickets"
    description = (
        "Summarize Jira tickets from a search payload returned by ticket retrieval tools.\n\n"

        "This tool accepts a dictionary produced by one of the ticket retrieval tools and generates\n"
        "a concise natural-language summary of the returned tickets.\n\n"

        "Supported input schemas:\n"
        '1. Payloads containing a "tickets" field, for example from:\n'
        "- exact ticket search\n"
        "- hybrid ticket search\n"
        "- broad ticket search\n\n"
        '2. Payloads containing a "results" field, for example from:\n'
        "- semantic ticket search\n\n"

        "The tool automatically normalizes these schemas into a single internal ticket list.\n\n"

        "Use this tool when:\n"
        "- the agent has already retrieved a set of tickets and needs to explain them\n"
        "- the user asks for a summary, pattern analysis, or overview\n"
        "- the agent wants to identify trends, recurring issues, or resolution states\n\n"

        "Do not use this tool as the first step of retrieval.\n"
        "First retrieve tickets using an appropriate search tool, then summarize the returned payload.\n\n"

        "Notes for agent use:\n"
        "- This tool works with the schema family returned by the ticket search tools\n"
        '- If the payload contains "results", the tool will extract ticket information from semantic hits\n'
        '- If the payload contains "tickets", the tool will summarize full structured ticket rows directly\n'
        "- This tool is best used after retrieval, not instead of retrieval"
    )

    inputs = {
        "search_payload": {
            "type": "object",
            "description": "Dictionary returned by a ticket retrieval tool.",
        }
    }

    output_type = "string"

    def __init__(self, used_llm_model):
        super().__init__()
        self.llm_model = used_llm_model

    def forward(self, search_payload: Dict[str, Any]) -> str:
        if not isinstance(search_payload, dict):
            return "Invalid search payload: expected a dictionary."

        normalized_tickets = extract_search_results(search_payload)

        if not normalized_tickets:
            return "No matching tickets found to summarize."

        ticket_blocks = create_ticket_block(normalized_tickets)

        prompt = (
            "You are a helpful Jira assistant speaking to a teammate.\n\n"
            "Summarize the tickets in a natural, conversational way.\n"
            "Do not sound robotic.\n"
            "Start with the main takeaway in 1-2 sentences.\n"
            "Then briefly mention the most relevant ticket patterns.\n"
            "If there are only a few tickets, mention each one simply.\n"
            "Avoid repeating unnecessary metadata.\n"
            "Stay factual and do not invent causes.\n\n"
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
        return response.content
