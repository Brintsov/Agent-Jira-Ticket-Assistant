#!/usr/bin/env python3
"""Streamlit UI for the Jira assistant pipeline.

This app wraps the `JiraAssistantPipeline` from `jira_knowledge_expert.py`
and provides a simple chat interface plus notebook-inspired examples.

Run:
    streamlit run jira_streamlit_app.py
"""

from __future__ import annotations

import traceback
from dataclasses import asdict
from typing import Any

import streamlit as st

from jira_knowledge_expert import (
    DEFAULT_DB_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_VECTORSTORE_PATH,
    JiraAssistantPipeline,
    PipelineConfig,
    example_prompts,
)

st.set_page_config(
    page_title="Jira Ticket Assistant",
    layout="wide",
)


DEFAULT_CHAT_GREETING = (
    "Hi! I’m your Jira ticket assistant. Ask about ticket patterns, run a project-specific "
    "search, summarize issues, or analyze distributions."
)


@st.cache_resource(show_spinner=False)
def build_pipeline_cached(config_items: tuple[tuple[str, Any], ...]) -> JiraAssistantPipeline:
    """Cache the pipeline so the heavy initialization only happens when config changes."""
    config_dict = dict(config_items)
    config = PipelineConfig(**config_dict)
    return JiraAssistantPipeline(config)


def get_pipeline(config: PipelineConfig) -> JiraAssistantPipeline:
    return build_pipeline_cached(tuple(sorted(asdict(config).items())))


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": DEFAULT_CHAT_GREETING}]
    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False
    if "pipeline_error" not in st.session_state:
        st.session_state.pipeline_error = None
    if "active_config" not in st.session_state:
        st.session_state.active_config = None


def reset_chat() -> None:
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_CHAT_GREETING}]


def render_sidebar() -> PipelineConfig:
    st.sidebar.header("Configuration")

    db_path = st.sidebar.text_input("Database path", value=DEFAULT_DB_PATH)
    vectorstore_path = st.sidebar.text_input("Vector store path", value=DEFAULT_VECTORSTORE_PATH)
    embedding_model = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
    embedding_device = st.sidebar.selectbox("Embedding device", options=["mps", "cpu", "cuda"], index=0)
    llm_model = st.sidebar.text_input("LLM model", value=DEFAULT_LLM_MODEL)
    max_tokens = st.sidebar.number_input("Max tokens", min_value=256, max_value=20000, value=5000, step=256)
    timeout_seconds = st.sidebar.number_input(
        "Timeout seconds", min_value=30, max_value=1800, value=160, step=10
    )
    verbosity_level = st.sidebar.number_input("Verbosity level", min_value=0, max_value=3, value=1, step=1)

    config = PipelineConfig(
        db_path=db_path,
        vectorstore_path=vectorstore_path,
        embedding_model=embedding_model,
        embedding_device=embedding_device,
        llm_model=llm_model,
        max_tokens=int(max_tokens),
        timeout_seconds=int(timeout_seconds),
        verbosity_level=int(verbosity_level),
    )

    st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Initialize / Reload", use_container_width=True):
            st.session_state.active_config = asdict(config)
            st.session_state.pipeline_ready = False
            st.session_state.pipeline_error = None
            build_pipeline_cached.clear()
            st.rerun()
    with col2:
        if st.button("Reset chat", use_container_width=True):
            reset_chat()
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Examples")
    for name, prompts in example_prompts().items():
        with st.sidebar.expander(name, expanded=False):
            st.write("Prompts in this flow:")
            for idx, prompt in enumerate(prompts, start=1):
                st.markdown(f"{idx}. {prompt}")
            if st.button(f"Run {name}", key=f"run_example_{name}", use_container_width=True):
                st.session_state.run_example = name
                st.rerun()

    return config


def ensure_pipeline(config: PipelineConfig) -> JiraAssistantPipeline | None:
    active_config = st.session_state.active_config or asdict(config)
    config_obj = PipelineConfig(**active_config)

    try:
        with st.spinner("Initializing Jira assistant pipeline..."):
            pipeline = get_pipeline(config_obj)
        st.session_state.pipeline_ready = True
        st.session_state.pipeline_error = None
        st.session_state.active_config = active_config
        return pipeline
    except Exception as exc:  # noqa: BLE001
        st.session_state.pipeline_ready = False
        st.session_state.pipeline_error = f"{exc}\n\n{traceback.format_exc()}"
        return None


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def run_prompt(pipeline: JiraAssistantPipeline, prompt: str) -> None:
    append_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing tickets..."):
            try:
                response = pipeline.run(prompt)
            except Exception as exc:  # noqa: BLE001
                response = f"Error while running the pipeline:\n\n```\n{exc}\n```"
        st.markdown(response)
    append_message("assistant", response)


def run_example_flow(pipeline: JiraAssistantPipeline, flow_name: str) -> None:
    prompts = example_prompts()[flow_name]
    st.info(f"Running example flow: {flow_name}")
    for idx, prompt in enumerate(prompts):
        append_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner(f"Running step {idx + 1} of {len(prompts)}..."):
                try:
                    response = pipeline.run(prompt, reset=(idx == 0))
                except Exception as exc:  # noqa: BLE001
                    response = f"Error while running example flow:\n\n```\n{exc}\n```"
            st.markdown(response)
        append_message("assistant", response)


def main() -> None:
    init_session_state()
    config = render_sidebar()

    st.title("Jira Ticket Assistant")
    st.caption("Streamlit frontend for the notebook-derived Jira model pipeline.")

    pipeline = ensure_pipeline(config)
    if st.session_state.pipeline_error:
        st.error("Pipeline initialization failed.")
        st.code(st.session_state.pipeline_error)

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Chat")
        render_messages()
    with right:
        st.subheader("Quick actions")
        st.markdown(
            "Use the chat to ask things like:\n"
            "- `What is current situation in our ticket database? Analyze the pattern`\n"
            "- `Find tickets related to SQL, summarize them`\n"
            "- `Search for exact Ignite project tickets`"
        )
        st.markdown("**Current config**")
        st.json(st.session_state.active_config or asdict(config))

    example_to_run = st.session_state.pop("run_example", None)
    if example_to_run and pipeline is not None:
        run_example_flow(pipeline, example_to_run)

    user_prompt = st.chat_input("Ask about tickets, patterns, summaries, or examples...")
    if user_prompt:
        if pipeline is None:
            st.error("The pipeline is not initialized. Fix the configuration in the sidebar first.")
        else:
            run_prompt(pipeline, user_prompt)


if __name__ == "__main__":
    main()
