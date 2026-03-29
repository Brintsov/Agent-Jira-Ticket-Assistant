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
from typing import Iterable

from agent.builder import create_code_agent, create_embeddings, create_model, create_repository
from agent.session import AgentSession


DEFAULT_DB_PATH = "./data/tickets.db"
DEFAULT_VECTORSTORE_PATH = "./data/jira_bge_small"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXIT_COMMANDS = {"exit", "quit", "q", "bye"}


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
        self.embeddings = create_embeddings(config)
        self.repository = create_repository(config, self.embeddings)
        self.model = create_model(config)
        self.agent = create_code_agent(config, self.repository, self.model)
        self.session = AgentSession()

    def run(self, prompt: str, *, reset: bool | None = None) -> str:
        return self.session.run(self.agent, prompt, reset=reset)

    def run_many(self, prompts: Iterable[str], *, reset_first: bool = True) -> list[tuple[str, str]]:
        return self.session.run_many(self.agent, prompts, reset_first=reset_first)

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
