# Agent Jira Ticket Assistant

## Overview

Agent Jira Ticket Assistant is a local, agent-driven system for querying and analyzing Jira tickets using a hybrid retrieval pipeline.

The system combines **structured retrieval** (SQLite) and **semantic retrieval** (vector search) within a unified tool-augmented LLM framework. A local language model orchestrates tool usage, dynamically selecting between exact filtering, semantic search, and hybrid strategies depending on the query.

This enables context-aware reasoning over large-scale ticket data and supports complex analytical queries such as:

- "What are the main recurring issues in the selected project?"
- "Find open bugs related to HTTP errors and provide a summary"
- "Analyze the distribution of issue types and priorities in a project"

### Key capabilities

- **Hybrid retrieval**: combines SQL filtering with embedding-based similarity search  
- **Tool-augmented reasoning**: LLM dynamically selects and executes search/analysis tools  
- **Context-aware summarization**: synthesizes insights from retrieved tickets  
- **Local-first architecture**: runs fully locally (LLM + vector store + database)  

## 🎥 [Watch the Streamlit application demo](https://www.loom.com/share/a3d66e888bad4f579830098c8c8c58ee)


## QA Examples
### Example Question 1

Question:
   `Search for tickets where project key is WW where the issue type is Bug and the status is Open. Among those tickets, select the ones related to HTTP status issues. Summarize it`

Executed code:
```python
search_payload = hybrid_ticket_search(semantic_query="HTTP status issues", project_key="WW", issue_type_name="Bug", status_name="Open", k=10)                                                                 
summary = summarize_tickets(search_payload)    
final_answer(summary)  
```

Answer:
```
The main issue is an application returning an HTTP status code of 14 instead of 200 for a specific URL, which is inconsistent across servers. We've identified that the HttpHeaderResult constructor is receiving a 14, and DefaultActionInvocation.getResultCode() is showing "success", which seems correct.
There's only one ticket, so far. It's a bug in the Struts 2 project with a major priority.
```

### Example Question 2
#### Question
Question:
`Search for exact Ignite project tickets, Make a summary of those tickets`

Executed code:
```python
search_payload = exact_ticket_search(project_key="IGNITE", status_name="Open", limit=10)                                                                                                                      
summary = summarize_tickets(search_payload)             
```
Answer: 
``` markdown
Sure, here's a summary of the tickets:

The main takeaway is that we have a few tickets open, mostly related to improvements and bug fixes. IGNITE-20754 is about fixing log prefix truncation in logit storage, which is a major priority. It involves 
differentiating between safe and unsafe truncation and possibly changing the approach to handling gaps in the index files.

IGNITE-11046 is about moving a utility from the ignite-core module to a separate module, which is a minor improvement.

IGNITE-8852 is about standardizing the use of boolean flags and long fields in CacheJoinNodeDiscoveryData.CacheInfo, which is another minor improvement.

IGNITE-17224 is a bug related to the eviction of data in the volatile (in-memory) data region. We need to implement the eviction feature and write tests for it, similar to what was done in 2.0.

IGNITE-17211 is a more complex issue where nested polymorphic configurations need a default type, which can lead to errors if the type is not known. We need to find a way to handle this without always setting
a default type.

IGNITE-17838 is about implementing automatic retries for implicit transaction management methods, which is a major feature.

IGNITE-23455 is about handling fragments sent to worker nodes where no response is received, which can lead to NodeLeft events. We need to have mechanisms to handle different types of transactions (RO, RW) 
appropriately.

IGNITE-20948 is about making column rename a forward-compatible schema change, which is a major improvement.

IGNITE-19550 is about reusing IDs of dropped Catalog objects for new objects, which is a forward-compatible change and a major improvement.

IGNITE-16929 is a new feature request for implementing sessions support for the .NET thin client, which is a major new feature.

These tickets cover a range of issues from minor improvements to major new features and bug fixes.
```
#### Follow-up
Question: `Taking into account those tickets, analyze distribution`

Executed Code:
```python
distribution = analyze_ticket_distribution(search_payload)                                                                                                                                                    
final_answer(distribution)                         
```
Answer:
```markdown
The main high-level takeaway is that all 10 Jira tickets are in the "Open" status, have a "Major" priority, are of type "Improvement," and are all part of the "IGNITE" project. There are no 
resolved or high-priority tickets in this set.

The distribution signals are quite concentrated. All tickets are in the "Open" status, indicating a high number of unresolved issues. The "Major" priority is consistent across all tickets, suggesting a high 
level of importance. The "Improvement" issue type is the only type present, with no other types like "Bug" or "New Feature." The tickets are all from the "IGNITE" project, indicating a single project focus.

Given the limited evidence, the result set looks concentrated. There are no open-like or resolved-like signals, and no high-priority-like signals. The lack of variety in issue types and the single project 
focus further reinforce the concentrated nature of the data.
```

## DataSet

This project utilizes the [Apache JIRA Issues Dataset](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues?select=changelog.csv), published on Kaggle by Ted Lozzo.  
The full dataset contains approximately 1,149,323 issue records. Due to hardware and memory constraints, a subset of **50,000 tickets** was sampled for downstream processing.

The following steps were performed on the sampled dataset:

1. **Relational Storage**  
   The data was ingested into a local **SQLite** database to enable structured querying and efficient filtering.

2. **Vectorization and Indexing**  
   Textual data from the tickets was embedded using the Hugging Face model  
   `BAAI/bge-small-en-v1.5`.  
   The embeddings were then stored using **LangChain** with a **Chroma** vector store, creating a local semantic index for similarity-based retrieval.

3. **Hybrid Retrieval Setup**  
   Both the SQLite database (for structured queries) and the Chroma vector database (for semantic search) were integrated into a hybrid retrieval pipeline.

4. **LLM Integration**  
   The retrieved context is passed to a local language model,  
   `Qwen/Qwen2.5-3B-Instruct`, which serves as the reasoning and response generation component of the agent.


## Technical Description
A local, tool-using Jira analysis assistant that combines:

- structured SQL filtering over a Jira ticket SQLite dataset,
- semantic retrieval over a Chroma vector store,
- an MLX-hosted local LLM (`Qwen/Qwen2.5-3B-Instruct`) acting as the orchestration brain,
- post-retrieval analysis/summarization tools,
- optional Streamlit UI,
- request/tool observability logs,
- routing evaluation harness.

---

### What this project does

The assistant answers queries about Jira tickets by routing to the right tool:

- **Exact search** for structured filters (project, status, type, priority, etc.).
- **Ticket key search** for direct IDs like `IGNITE-20754`.
- **Semantic search** for concept/topic queries (e.g., “SQL deadlocks”).
- **Hybrid search** for structured + semantic constraints.
- **Broad search** for high-level “what is going on?” overviews.

Then it can:

- summarize results,
- discuss findings conversationally,
- analyze distribution (status/priority/type/project),
- analyze recurring patterns.

---

### Architecture (end-to-end)

#### 1 Execution flow

1. User sends a prompt (CLI or Streamlit).
2. `JiraAssistantPipeline` builds:
   - embeddings,
   - ticket repository (SQLite + Chroma),
   - MLX model,
   - `smolagents.CodeAgent` with all tools.
3. Agent follows `SYSTEM_PROMPT` routing policy and calls a search tool first.
4. Optional analysis tool is called on search payload.
5. `AgentSession` logs request lifecycle and returns final response.

#### 2 Directory guide

```text
agent/
  builder.py           # Creates embeddings, repository, model, tools, code agent
  session.py           # Request/session lifecycle, reset behavior, logging integration
  system_prompt.py     # Routing and behavioral policy for tool selection

tools/
  search_tools/
    exact_ticket_search.py
    ticket_key_search.py
    semantic_ticket_search.py
    hybrid_ticket_search.py
    broad_ticket_search.py
  analysis_tools/
    ticket_status_distribution.py
    ticket_patterns.py
  conversational_tools/
    summarization.py
    discuss_ticket_findings.py
  utils.py             # Payload normalization + deterministic analysis helpers

observation/
  logger.py            # JSONL observability logging

evals/
  routing_cases.yaml   # Routing test cases
  routing_tests.py     # Live routing test harness

jira_knowledge_expert.py # CLI pipeline entrypoint
jira_streamlit_app.py    # Streamlit chat app
ticket_repository.py     # SQLite + Chroma data access abstraction
```

---

#### Quick smoke run (CLI)

```bash
python jira_knowledge_expert.py --query "What is current situation in our ticket database? Analyze the pattern"
```

---

### Running the assistant

### Option A — Interactive CLI chat

```bash
python jira_knowledge_expert.py
```

Type prompts; exit with `exit`, `quit`, `q`, or `bye`.

### Option B — Non-interactive one-shot query

```bash
python jira_knowledge_expert.py --query "Find tickets related to SQL, summarize them"
```

### Option C — Multi-step scripted sequence

```bash
python jira_knowledge_expert.py \
  --query "Search for exact Ignite project tickets" \
  --query "Taking into account those tickets, analyze distribution" \
  --query "Make a summary of those tickets"
```

### Option D — Built-in examples

```bash
python jira_knowledge_expert.py --example ignite_case
```

Available example groups:

- `pattern_overview`
- `ignite_flow`
- `ignite_case`
- `http_bug_case`
- `sql_case`

### Option E — Streamlit UI

```bash
streamlit run jira_streamlit_app.py
```

Then configure paths/models in sidebar and click **Initialize / Reload**.

---

### CLI options reference

```bash
python jira_knowledge_expert.py --help
```

Important flags:

- `--db-path` (default `./data/tickets.db`)
- `--vectorstore-path` (default `./data/jira_bge_small`)
- `--embedding-model` (default `BAAI/bge-small-en-v1.5`)
- `--embedding-device` (`mps|cpu|cuda`, default `mps`)
- `--llm-model` (default `Qwen/Qwen2.5-3B-Instruct`)
- `--max-tokens` (default `5000`)
- `--timeout-seconds` (default `160`)
- `--verbosity-level` (default `1`)
- `--query` (repeatable)
- `--example` (predefined flow)
- `--no-reset-first` (disable first-turn memory reset)

---

### Tool routing guide

Use these patterns for best results:

#### Exact search

- “Show open bugs in IGNITE”
- “Find 20 high priority CORE tickets”

#### Ticket key search

- “Show IGNITE-20754”
- “Find IGNITE-20754, CORE-88”

#### Semantic search

- “Find tickets about SQL deadlocks”
- “Find issues related to login redirect loops”

#### Hybrid search

- “Find open WW bugs related to HTTP status issues”
- “Find resolved IGNITE bugs about timeout failures”

#### Broad search

- “What is going on in Jira lately?”
- “Give me an overview of current ticket status”

#### Follow-up analysis

After any search result, ask:

- “Summarize them.”
- “Analyze status distribution.”
- “What patterns do you see?”
- “What do you think is most critical?”

---

### Observability and logs

The assistant writes JSONL logs (default):

```text
logs/agent_observability.jsonl
```

Example:
```markdown
{"ts": 1774791522.8927581, "event": "request.start", "request_id": "2c40356d-1ec5-41b9-8e2b-91d09da31355", "session_id": "500836e0-ff43-48bb-8c64-5a6a5b4683d2", "turn_index": 3, "pid": 47285, "reset": false, "prompt": "Find 100 tickets for project FLINK, where Oracle CDC is mentioned", "prompt_tokens_est": 14, "prompt_chars": 65}
{"ts": 1774791534.028783, "event": "tool.start", "request_id": "2c40356d-1ec5-41b9-8e2b-91d09da31355", "session_id": null, "turn_index": 0, "pid": 47285, "tool": "exact_ticket_search", "args_count": 0, "kwargs_keys": ["limit", "project_key", "summary_contains"], "call": {"kwargs": {"project_key": "FLINK", "summary_contains": "Oracle CDC", "limit": 100}}}
{"ts": 1774791534.131694, "event": "tool.end", "request_id": "2c40356d-1ec5-41b9-8e2b-91d09da31355", "session_id": null, "turn_index": 0, "pid": 47285, "tool": "exact_ticket_search", "latency_ms": 102.9, "result_type": "dict", "result_chars": 3807}
{"ts": 1774791534.136179, "event": "request.end", "request_id": "2c40356d-1ec5-41b9-8e2b-91d09da31355", "session_id": "500836e0-ff43-48bb-8c64-5a6a5b4683d2", "turn_index": 3, "pid": 47285, "latency_ms": 11242.88, "response_tokens_est": 551, "tools_used": ["exact_ticket_search"]}
{"ts": 1774791547.330509, "event": "request.start", "request_id": "e0fa2bc4-c8de-4684-87e9-f1e34cc024ab", "session_id": "500836e0-ff43-48bb-8c64-5a6a5b4683d2", "turn_index": 4, "pid": 47285, "reset": false, "prompt": "Let's discuss those ticket findings", "prompt_tokens_est": 6, "prompt_chars": 35}
{"ts": 1774791559.560622, "event": "tool.start", "request_id": "e0fa2bc4-c8de-4684-87e9-f1e34cc024ab", "session_id": null, "turn_index": 0, "pid": 47285, "tool": "discuss_ticket_findings", "args_count": 3, "kwargs_keys": [], "call": {"kwargs": {}, "args": ["{'count': 1, 'tickets': [{'id': 13572554.0, 'key': 'FLINK-34775', 'summary': \"[Bug] oracle cdc logminer can't catch up the latest records when scn huge increment occured.\", 'resolution.id': None, 'resolution.description': None, 'resolution....", "Let's discuss those ticket findings", "conversational"]}}
{"ts": 1774791571.881355, "event": "tool.end", "request_id": "e0fa2bc4-c8de-4684-87e9-f1e34cc024ab", "session_id": null, "turn_index": 0, "pid": 47285, "tool": "discuss_ticket_findings", "latency_ms": 12320.69, "result_type": "str", "result_chars": 1628}
{"ts": 1774791622.900225, "event": "request.end", "request_id": "e0fa2bc4-c8de-4684-87e9-f1e34cc024ab", "session_id": "500836e0-ff43-48bb-8c64-5a6a5b4683d2", "turn_index": 4, "pid": 47285, "latency_ms": 75568.2, "response_tokens_est": 341, "tools_used": ["discuss_ticket_findings"]}
```

Events include:

- `session.start` / `session.reset`
- `request.start` / `request.end` / `request.error`
- `tool.start` / `tool.end` / `tool.error`

---

### Evaluations (routing behavior)

`evals/routing_tests.py` replays prompts from `evals/routing_cases.yaml`, then validates:

1. expected tool was called,
2. expected call parameters match observed args.

Run:

```bash
python -m unittest evals.routing_tests -v
```

Notes:

- This is a **live** test (needs local data/model ready).

---
