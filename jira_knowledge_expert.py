#!/usr/bin/env python3
"""Jira ticket assistant pipeline extracted from the notebook.

This script turns the notebook workflow into a reusable CLI pipeline.
It keeps the same core components:
- BGE embeddings
- TicketRepository
- search / analysis / conversational tools
- smolagents CodeAgent with Qwen2.5-3B-Instruct via MLX

Expected local project structure:
.
├── data/
│   ├── tickets.db
│   └── jira_bge_small/
├── ticket_repository.py
└── tools/
    ├── analysis_tools.py
    ├── conversational_tools.py
    └── search_tools.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from smolagents import CodeAgent
from smolagents.models import MLXModel
from langchain_huggingface import HuggingFaceEmbeddings

from ticket_repository import TicketRepository
from tools.analysis_tools import AnalyzeTicketPatternsTool, AnalyzeTicketDistributionTool
from tools.conversational_tools import DiscussTicketFindingsTool, SummarizeTicketsTool
from tools.search_tools import (
    BroadTicketSearchTool,
    ExactTicketSearchTool,
    HybridTicketSearchTool,
    SemanticTicketSearchTool,
    TicketKeySearchTool
)


DEFAULT_DB_PATH = "./data/tickets.db"
DEFAULT_VECTORSTORE_PATH = "./data/jira_bge_small"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXIT_COMMANDS = {"exit", "quit", "q", "bye"}
SYSTEM_PROMPT = """
    You are a Jira ticket assistant.

    Your job:
    choose the correct tool
    retrieve tickets
    then analyze or summarize if requested
    never invent data
    
    Core rule:
    Use the simplest correct tool.
    
    Critical distinction:
    "BEAM" means project, not a ticket
    "BEAM-123" is a ticket ID
    
    Never treat BEAM as a ticket ID.
    
    Tool routing examples:
    
    User: Find 50 tickets for project BEAM
    Use exact search with project BEAM and limit 50
    
    User: Show open bugs in IGNITE
    Use exact search
    
    User: Find high priority tickets in CORE
    Use exact search
    
    User: Show BEAM-123
    Use ticket_key_search
    
    User: Find BEAM-123 and CORE-456
    Use ticket_key_search
    
    User: Find tickets about login failures
    Use semantic_search
    
    User: Show issues related to SQL deadlocks
    Use semantic_search
    
    User: Find open BEAM bugs about login failures
    Use hybrid_search
    
    User: Show CORE tickets related to deployment instability
    Use hybrid_search
    
    User: What is going on in Jira lately
    Use broad_search
    
    User: Give me overview of tickets
    Use broad_search
    
    Important negative example:
    User: Find tickets for project BEAM
    Do not use ticket_key_search
    
    Decision shortcuts:
    project only means exact
    project with number like BEAM-123 means ticket_key_search
    topic only means semantic_search
    project plus topic means hybrid_search
    vague request means broad_search
    
    Multi-step behavior:
    If user asks to find and then summarize or analyze
    first search
    then summarize or analyze
    
    Follow-ups:
    those tickets, them, these refer to last results
    do not search again unless needed
    
    Output rules:
    be concise
    do not hallucinate
    if no results say so clearly
"""


@dataclass
class PipelineConfig:
    db_path: str = DEFAULT_DB_PATH
    vectorstore_path: str = DEFAULT_VECTORSTORE_PATH
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str = "mps"
    llm_model: str = DEFAULT_LLM_MODEL
    max_tokens: int = 5000
    timeout_seconds: int = 160
    verbosity_level: int = 1


class JiraAssistantPipeline:
    """Builds and runs the ticket-analysis agent pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.embeddings = self._build_embeddings()
        self.repository = self._build_repository()
        self.model = self._build_model()
        self.agent = self._build_agent()
        self._is_first_turn = True

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={"device": self.config.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_repository(self) -> TicketRepository:
        db_path = Path(self.config.db_path)
        vectorstore_path = Path(self.config.vectorstore_path)

        if not db_path.exists():
            raise FileNotFoundError(f"Ticket DB not found: {db_path}")
        if not vectorstore_path.exists():
            raise FileNotFoundError(f"Vector store not found: {vectorstore_path}")

        return TicketRepository(
            str(db_path),
            str(vectorstore_path),
            self.embeddings,
        )

    def _build_model(self) -> MLXModel:
        return MLXModel(
            model_id=self.config.llm_model,
            max_tokens=self.config.max_tokens,
        )

    def _build_agent(self) -> CodeAgent:
        exact_search = ExactTicketSearchTool(ticket_repository=self.repository)
        semantic_search = SemanticTicketSearchTool(ticket_repository=self.repository)
        hybrid_search = HybridTicketSearchTool(ticket_repository=self.repository)
        broad_search = BroadTicketSearchTool(ticket_repository=self.repository)
        key_search = TicketKeySearchTool(ticket_repository=self.repository)

        summarize_tickets = SummarizeTicketsTool(self.model)
        discuss_ticket_findings = DiscussTicketFindingsTool(self.model)
        analyze_ticket_distribution = AnalyzeTicketDistributionTool(self.model)
        analyze_ticket_patterns = AnalyzeTicketPatternsTool(self.model)

        return CodeAgent(
            tools=[
                exact_search,
                semantic_search,
                hybrid_search,
                broad_search,
                key_search,
                summarize_tickets,
                discuss_ticket_findings,
                analyze_ticket_distribution,
                analyze_ticket_patterns,
            ],
            instructions=SYSTEM_PROMPT,
            model=self.model,
            additional_authorized_imports=["json"],
            executor_type="local",
            executor_kwargs={"timeout_seconds": self.config.timeout_seconds},
            verbosity_level=self.config.verbosity_level,
        )

    def run(self, prompt: str, *, reset: bool | None = None) -> str:
        effective_reset = self._is_first_turn if reset is None else reset
        response = self.agent.run(prompt, reset=effective_reset)
        self._is_first_turn = False
        return response

    def run_many(self, prompts: Iterable[str], *, reset_first: bool = True) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        for i, prompt in enumerate(prompts):
            response = self.run(prompt, reset=(reset_first if i == 0 else False))
            results.append((prompt, response))
        return results

    def interactive_chat(self) -> None:
        print("Jira assistant ready. Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in EXIT_COMMANDS:
                    print("Assistant: Goodbye.")
                    break

                response = self.run(user_input)
                print(f"Assistant: {response}\n")
            except KeyboardInterrupt:
                print("\nAssistant: Interrupted. Goodbye.")
                break
            except Exception as exc:
                print(f"Assistant: Error: {exc}\n")


def example_prompts() -> dict[str, list[str]]:
    return {
        "pattern_overview": [
            "What is current situation in our ticket database? Analyze the pattern",
        ],
        "ignite_flow": [
            "Search for exact Ignite project tickets",
            "Taking into account those tickets, analyze distribution",
            "Make a summary of those tickets",
        ],
        "ignite_case": [
            "Search for up to three tickets in exact manner where status is open and project name is Ignite, Let's discuss those findings",
            "Let's stick to IGNITE-20754, what solution to this you can propose?",
            "Have we seen similar issues before in other projects?",
            "Taking into account tickets that you have found, can you summarize to me them?",
        ],
        "http_bug_case": [
            "Search for tickets where project key is WW where the issue type is Bug and the status is Open. Among those tickets, select the ones related to HTTP status issues, and then summarize them using the summarization tool.",
            "What do you think about found ticket?",
            "Do we have any simialar tickets in our tickets database? Summarize them",
        ],
        "sql_case": [
            "Find tickets related to SQL, summarize them",
            "Analyze status distribution of this?",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Jira ticket assistant pipeline.")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to the SQLite tickets database.")
    parser.add_argument(
        "--vectorstore-path",
        default=DEFAULT_VECTORSTORE_PATH,
        help="Path to the Chroma / vector store directory.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Embedding model name.",
    )
    parser.add_argument(
        "--embedding-device",
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device for embeddings.",
    )
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="MLX LLM model identifier.")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Maximum generation tokens.")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=160,
        help="Execution timeout for the local agent executor.",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="smolagents verbosity level.",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Run one or more prompts non-interactively. Use multiple times to keep context across prompts.",
    )
    parser.add_argument(
        "--example",
        choices=sorted(example_prompts().keys()),
        help="Run one of the built-in notebook-inspired prompt sequences.",
    )
    parser.add_argument(
        "--no-reset-first",
        action="store_true",
        help="Do not reset memory on the first non-interactive query.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        db_path=args.db_path,
        vectorstore_path=args.vectorstore_path,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        llm_model=args.llm_model,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        verbosity_level=args.verbosity_level,
    )

    try:
        pipeline = JiraAssistantPipeline(config)
    except Exception as exc:
        print(f"Failed to initialize pipeline: {exc}", file=sys.stderr)
        return 1

    prompts: list[str] = []
    if args.example:
        prompts.extend(example_prompts()[args.example])
    if args.query:
        prompts.extend(args.query)

    if prompts:
        try:
            results = pipeline.run_many(prompts, reset_first=not args.no_reset_first)
            for idx, (prompt, response) in enumerate(results, start=1):
                print(f"\n--- Step {idx} ---")
                print(f"Prompt: {prompt}\n")
                print(response)
        except Exception as exc:
            print(f"Pipeline run failed: {exc}", file=sys.stderr)
            return 1
        return 0

    pipeline.interactive_chat()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
