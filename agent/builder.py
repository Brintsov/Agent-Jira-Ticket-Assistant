from typing import Any
from pathlib import Path
from time import perf_counter

from smolagents import CodeAgent
from smolagents.models import MLXModel
from langchain_huggingface import HuggingFaceEmbeddings

from agent.system_prompt import SYSTEM_PROMPT
from observation.logger import log_event, append_used_tool
from ticket_repository import TicketRepository
from tools.analysis_tools import AnalyzeTicketDistributionTool, AnalyzeTicketPatternsTool
from tools.conversational_tools import DiscussTicketFindingsTool, SummarizeTicketsTool
from tools.search_tools import (
    BroadTicketSearchTool,
    ExactTicketSearchTool,
    HybridTicketSearchTool,
    SemanticTicketSearchTool,
    TicketKeySearchTool,
)


def _instrument_tool(tool) -> None:
    original_forward = tool.forward
    tool_name = getattr(tool, "name", tool.__class__.__name__)

    def wrapped_forward(*args, **kwargs):
        start = perf_counter()
        log_event(
            "tool.start",
            tool=tool_name,
            args_count=max(0, len(args) - 1),
            kwargs_keys=sorted(kwargs.keys()),
        )
        try:
            result = original_forward(*args, **kwargs)
            latency_ms = round((perf_counter() - start) * 1000, 2)
            log_event(
                "tool.end",
                tool=tool_name,
                latency_ms=latency_ms,
                result_type=type(result).__name__,
                result_chars=len(str(result)),
            )
            append_used_tool(tool_name)
            return result
        except Exception as exc:
            latency_ms = round((perf_counter() - start) * 1000, 2)
            log_event("tool.error", tool=tool_name, latency_ms=latency_ms, error=str(exc))
            raise

    tool.forward = wrapped_forward


def create_embeddings(config: Any) -> HuggingFaceEmbeddings:
    log_event(
        "pipeline.embeddings.init",
        embedding_model=config.embedding_model,
        embedding_device=config.embedding_device,
    )
    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_repository(config: Any, embeddings: HuggingFaceEmbeddings) -> TicketRepository:
    db_path = Path(config.db_path)
    vectorstore_path = Path(config.vectorstore_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Ticket DB not found: {db_path}")
    if not vectorstore_path.exists():
        raise FileNotFoundError(f"Vector store not found: {vectorstore_path}")

    log_event(
        "pipeline.repository.init",
        db_path=str(db_path),
        vectorstore_path=str(vectorstore_path),
    )

    return TicketRepository(
        str(db_path),
        str(vectorstore_path),
        embeddings,
    )


def create_model(config: Any) -> MLXModel:
    log_event(
        "pipeline.model.init",
        llm_model=config.llm_model,
        max_tokens=config.max_tokens,
    )
    return MLXModel(
        model_id=config.llm_model,
        max_tokens=config.max_tokens,
    )


def create_code_agent(config: Any, repository: TicketRepository, model: MLXModel) -> CodeAgent:
    exact_search = ExactTicketSearchTool(ticket_repository=repository)
    semantic_search = SemanticTicketSearchTool(ticket_repository=repository)
    hybrid_search = HybridTicketSearchTool(ticket_repository=repository)
    broad_search = BroadTicketSearchTool(ticket_repository=repository)
    key_search = TicketKeySearchTool(ticket_repository=repository)

    summarize_tickets = SummarizeTicketsTool(model)
    discuss_ticket_findings = DiscussTicketFindingsTool(model)
    analyze_ticket_distribution = AnalyzeTicketDistributionTool(model)
    analyze_ticket_patterns = AnalyzeTicketPatternsTool(model)
    tools = [
        exact_search,
        semantic_search,
        hybrid_search,
        broad_search,
        key_search,
        summarize_tickets,
        discuss_ticket_findings,
        analyze_ticket_distribution,
        analyze_ticket_patterns,
    ]
    for tool in tools:
        _instrument_tool(tool)

    log_event(
        "pipeline.agent.init",
        tool_names=[getattr(tool, "name", tool.__class__.__name__) for tool in tools],
        timeout_seconds=config.timeout_seconds,
        verbosity_level=config.verbosity_level,
    )

    return CodeAgent(
        tools=tools,
        instructions=SYSTEM_PROMPT,
        model=model,
        additional_authorized_imports=["json"],
        executor_type="local",
        executor_kwargs={"timeout_seconds": config.timeout_seconds},
        verbosity_level=config.verbosity_level,
    )
