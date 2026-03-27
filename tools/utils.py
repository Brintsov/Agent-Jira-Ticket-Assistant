from collections import Counter


STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "to", "from",
        "of", "on", "in", "at", "by", "with", "without", "into", "onto", "over", "under",
        "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
        "those", "as", "not", "no", "yes", "can", "could", "should", "would", "will",
        "may", "might", "must", "do", "does", "did", "done", "have", "has", "had",
        "user", "users", "ticket", "issue", "issues", "bug", "bugs", "task", "story",
        "error", "errors", "problem", "problems", "request", "requests", "case", "cases",
        "jira", "please", "need", "needs", "needed", "using", "use", "used",
        "summary", "description", "log", "logs", "screen", "page", "data", "system",
        "null", "none", "unknown", "empty"
    }

DOMAIN_KEEPWORDS = {
    "login", "redirect", "timeout", "authentication", "auth", "deploy", "deployment",
    "deadlock", "sql", "database", "api", "payment", "payments", "session", "cache",
    "sync", "queue", "worker", "background", "cron", "permission", "permissions",
    "access", "token", "refresh", "latency", "performance", "crash", "failure",
    "validation", "webhook", "pipeline", "release", "checkout", "invoice", "search",
    "notification", "signup", "registration", "import", "export", "integration"
}


def extract_search_results(search_payload):
    normalized_tickets = []
    if "tickets" in search_payload and isinstance(search_payload["tickets"], list):
        for t in search_payload["tickets"]:
            if not isinstance(t, dict):
                continue

            normalized_tickets.append({
                "key": t.get("key", ""),
                "project_key": t.get("project.key", t.get("project_key", "")),
                "project_name": t.get("project.name", t.get("project_name", "")),
                "issue_type_name": t.get("issuetype.name", t.get("issue_type_name", "")),
                "status_name": t.get("status.name", t.get("status_name", "")),
                "resolution_name": t.get("resolution.name", t.get("resolution_name", "")),
                "priority_name": t.get("priority.name", t.get("priority_name", "")),
                "summary": t.get("summary", ""),
                "description": t.get("description", ""),
                "labels_text": t.get("labels_text", t.get("labels", "")),
            })

    elif "results" in search_payload and isinstance(search_payload["results"], list):
        for r in search_payload["results"]:
            if not isinstance(r, dict):
                continue

            metadata = r.get("metadata", {}) or {}

            normalized_tickets.append({
                "key": r.get("key") or metadata.get("key", ""),
                "project_key": metadata.get("project_key", ""),
                "project_name": metadata.get("project_name", ""),
                "issue_type_name": metadata.get("issue_type_name", ""),
                "status_name": metadata.get("status_name", ""),
                "resolution_name": metadata.get("resolution_name", ""),
                "priority_name": metadata.get("priority_name", ""),
                "summary": "",
                "description": r.get("content", ""),
                "labels_text": metadata.get("labels_text", ""),
            })
    return normalized_tickets


def create_ticket_block(normalized_tickets):
    ticket_blocks = []
    for t in normalized_tickets:
        block = "\n".join([
            f"Key: {t.get('key', '')}",
            f"Project: {t.get('project_key', '')} - {t.get('project_name', '')}".strip(" -"),
            f"Type: {t.get('issue_type_name', '')}",
            f"Status: {t.get('status_name', '')}",
            f"Resolution: {t.get('resolution_name', '')}",
            f"Priority: {t.get('priority_name', '')}",
            f"Labels: {t.get('labels_text', '')}",
            f"Summary: {t.get('summary', '')}",
            f"Description: {t.get('description', '')}",
        ])
        ticket_blocks.append(block)
    return ticket_blocks


def tokenize(text: str) -> list[str]:
    text = text.lower()
    cleaned_chars = []
    for ch in text:
        if ch.isalnum() or ch in {"_", "-", " "}:
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)

    tokens = []
    for tok in text.split():
        tok = tok.strip("-_ ")
        if not tok:
            continue
        if len(tok) < 3 and tok not in {"ui", "db", "qa"}:
            continue
        if tok in STOPWORDS and tok not in DOMAIN_KEEPWORDS:
            continue
        if tok.isdigit():
            continue
        tokens.append(tok)
    return tokens


def clean_text(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split()).strip()


def clean_value(value, fallback: str = "Unknown") -> str:
    value = clean_text(value)
    return value if value else fallback


def split_labels(labels_text) -> list[str]:
    if labels_text is None:
        return []
    if isinstance(labels_text, list):
        raw_labels = labels_text
    else:
        raw_labels = str(labels_text).replace(";", ",").split(",")

    cleaned = []
    for label in raw_labels:
        label = clean_text(label)
        if label:
            cleaned.append(label)
    return cleaned


def build_analysis_context(search_payload: dict) -> dict:
    """
    Build a reusable analytical context from a Jira search payload.

    This is an internal helper intended to normalize Jira retrieval results into a single
    consistent structure that can be reused by multiple analytical tools such as:
    - analyze_ticket_distribution
    - analyze_ticket_patterns
    - analyze_ticket_overview
    - future ranking or triage tools

    Supported input schemas:
    1. Payloads containing a "tickets" field:
       - exact_search
       - hybrid_search
       - broad_search (if you add one later)
    2. Payloads containing a "results" field:
       - semantic_search

    The function performs the following steps:
    - validates payload shape
    - normalizes ticket fields into a common schema
    - extracts structured counters (status, resolution, priority, issue type, project, labels)
    - computes lightweight operational signals (open-like, resolved-like, high-priority-like)
    - extracts simple textual signals (keywords, bigrams)
    - prepares compact ticket blocks for downstream LLM analysis

    Args:
        search_payload: Dictionary returned by a Jira retrieval tool.

    Returns:
        A dictionary with:
        {
            "ok": <bool>,
            "error": <str | None>,
            "count": <int>,
            "search_type": <str>,
            "normalized_tickets": [ ... ],
            "distribution": {
                "status": {...},
                "resolution": {...},
                "priority": {...},
                "issue_type": {...},
                "project": {...},
                "labels": {...}
            },
            "top_keywords": [
                {"keyword": <str>, "count": <int>}, ...
            ],
            "top_bigrams": [
                {"phrase": <str>, "count": <int>}, ...
            ],
            "top_labels": [
                {"label": <str>, "count": <int>}, ...
            ],
            "top_projects": [
                {"project": <str>, "count": <int>}, ...
            ],
            "status_signals": {
                "open_like": <int>,
                "resolved_like": <int>,
                "high_priority_like": <int>
            },
            "high_level_metrics": {
                "unique_projects": <int>,
                "unique_statuses": <int>,
                "unique_issue_types": <int>
            },
            "compact_ticket_blocks": [<str>, ...]
        }

    Notes:
    - This function is deterministic and does not call the model.
    - Missing or sparse metadata is normalized to "Unknown" where appropriate.
    - For semantic search payloads, some fields may be less complete than exact/hybrid payloads.
    """
    try:
        payload = search_payload
    except Exception as e:
        return {
            "ok": False,
            "error": f"Invalid payload: {e}",
            "count": 0,
            "search_type": "unknown",
            "normalized_tickets": [],
            "distribution": {
                "status": {},
                "resolution": {},
                "priority": {},
                "issue_type": {},
                "project": {},
                "labels": {},
            },
            "top_keywords": [],
            "top_bigrams": [],
            "top_labels": [],
            "top_projects": [],
            "status_signals": {
                "open_like": 0,
                "resolved_like": 0,
                "high_priority_like": 0,
            },
            "high_level_metrics": {
                "unique_projects": 0,
                "unique_statuses": 0,
                "unique_issue_types": 0,
            },
            "compact_ticket_blocks": [],
        }

    normalized_tickets = []

    if isinstance(payload, dict) and "tickets" in payload and isinstance(payload["tickets"], list):
        for t in payload["tickets"]:
            if not isinstance(t, dict):
                continue

            normalized_tickets.append({
                "key": clean_text(t.get("key", "")),
                "project_key": clean_text(t.get("project.key", t.get("project_key", ""))),
                "project_name": clean_text(t.get("project.name", t.get("project_name", ""))),
                "issue_type_name": clean_text(t.get("issuetype.name", t.get("issue_type_name", ""))),
                "status_name": clean_text(t.get("status.name", t.get("status_name", ""))),
                "resolution_name": clean_text(t.get("resolution.name", t.get("resolution_name", ""))),
                "priority_name": clean_text(t.get("priority.name", t.get("priority_name", ""))),
                "summary": clean_text(t.get("summary", "")),
                "description": clean_text(t.get("description", "")),
                "labels_text": t.get("labels_text", t.get("labels", "")),
            })

    elif isinstance(payload, dict) and "results" in payload and isinstance(payload["results"], list):
        for r in payload["results"]:
            if not isinstance(r, dict):
                continue

            metadata = r.get("metadata", {}) or {}

            normalized_tickets.append({
                "key": clean_text(r.get("key") or metadata.get("key", "")),
                "project_key": clean_text(metadata.get("project_key", "")),
                "project_name": clean_text(metadata.get("project_name", "")),
                "issue_type_name": clean_text(metadata.get("issue_type_name", "")),
                "status_name": clean_text(metadata.get("status_name", "")),
                "resolution_name": clean_text(metadata.get("resolution_name", "")),
                "priority_name": clean_text(metadata.get("priority_name", "")),
                "summary": "",
                "description": clean_text(r.get("content", "")),
                "labels_text": metadata.get("labels_text", ""),
            })

    else:
        return {
            "ok": False,
            "error": "No supported ticket data found in payload. Expected an object with either 'tickets' or 'results'.",
            "count": 0,
            "search_type": payload.get("search_type", "unknown") if isinstance(payload, dict) else "unknown",
            "normalized_tickets": [],
            "distribution": {
                "status": {},
                "resolution": {},
                "priority": {},
                "issue_type": {},
                "project": {},
                "labels": {},
            },
            "top_keywords": [],
            "top_bigrams": [],
            "top_labels": [],
            "top_projects": [],
            "status_signals": {
                "open_like": 0,
                "resolved_like": 0,
                "high_priority_like": 0,
            },
            "high_level_metrics": {
                "unique_projects": 0,
                "unique_statuses": 0,
                "unique_issue_types": 0,
            },
            "compact_ticket_blocks": [],
        }

    if not normalized_tickets:
        return {
            "ok": True,
            "error": None,
            "count": 0,
            "search_type": payload.get("search_type", "unknown") if isinstance(payload, dict) else "unknown",
            "normalized_tickets": [],
            "distribution": {
                "status": {},
                "resolution": {},
                "priority": {},
                "issue_type": {},
                "project": {},
                "labels": {},
            },
            "top_keywords": [],
            "top_bigrams": [],
            "top_labels": [],
            "top_projects": [],
            "status_signals": {
                "open_like": 0,
                "resolved_like": 0,
                "high_priority_like": 0,
            },
            "high_level_metrics": {
                "unique_projects": 0,
                "unique_statuses": 0,
                "unique_issue_types": 0,
            },
            "compact_ticket_blocks": [],
        }

    open_like_statuses = {
        "open", "to do", "todo", "in progress", "selected for development",
        "reopened", "backlog", "ready", "pending", "blocked"
    }
    resolved_like_statuses = {
        "done", "closed", "resolved", "completed", "complete"
    }
    high_priority_names = {
        "highest", "high", "critical", "blocker", "urgent", "p1", "sev1", "severity 1"
    }

    status_counter = Counter()
    resolution_counter = Counter()
    priority_counter = Counter()
    issue_type_counter = Counter()
    project_counter = Counter()
    labels_counter = Counter()
    keyword_counter = Counter()
    bigram_counter = Counter()

    open_like_count = 0
    resolved_like_count = 0
    high_priority_like_count = 0

    compact_ticket_blocks = []

    for t in normalized_tickets:
        status_name = clean_value(t.get("status_name"))
        resolution_name = clean_value(t.get("resolution_name"))
        priority_name = clean_value(t.get("priority_name"))
        issue_type_name = clean_value(t.get("issue_type_name"))

        project_key = clean_text(t.get("project_key", ""))
        project_name = clean_text(t.get("project_name", ""))
        if project_key:
            project_value = project_key
        elif project_name:
            project_value = project_name
        else:
            project_value = "Unknown"

        status_counter[status_name] += 1
        resolution_counter[resolution_name] += 1
        priority_counter[priority_name] += 1
        issue_type_counter[issue_type_name] += 1
        project_counter[project_value] += 1

        labels = split_labels(t.get("labels_text", ""))
        for label in labels:
            labels_counter[label.lower()] += 1

        summary = clean_text(t.get("summary", ""))
        description = clean_text(t.get("description", ""))
        combined_text = f"{summary} {description}".strip()
        tokens = tokenize(combined_text)
        keyword_counter.update(tokens)

        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a == b:
                continue
            if len(a) >= 3 and len(b) >= 3:
                bigram_counter[f"{a} {b}"] += 1

        if status_name.lower() in open_like_statuses:
            open_like_count += 1

        if status_name.lower() in resolved_like_statuses or resolution_name.lower() not in {"", "unknown", "unresolved", "none"}:
            resolved_like_count += 1

        if priority_name.lower() in high_priority_names:
            high_priority_like_count += 1

        compact_ticket_blocks.append("\n".join([
            f"Key: {clean_text(t.get('key', ''))}",
            f"Project: {project_value}",
            f"Type: {issue_type_name}",
            f"Status: {status_name}",
            f"Resolution: {resolution_name}",
            f"Priority: {priority_name}",
            f"Labels: {', '.join(labels)}",
            f"Summary: {summary}",
            f"Description: {description[:1200]}",
        ]))

    return {
        "ok": True,
        "error": None,
        "count": len(normalized_tickets),
        "search_type": payload.get("search_type", "unknown") if isinstance(payload, dict) else "unknown",
        "normalized_tickets": normalized_tickets,
        "distribution": {
            "status": dict(status_counter),
            "resolution": dict(resolution_counter),
            "priority": dict(priority_counter),
            "issue_type": dict(issue_type_counter),
            "project": dict(project_counter),
            "labels": dict(labels_counter),
        },
        "top_keywords": [
            {"keyword": keyword, "count": count}
            for keyword, count in keyword_counter.most_common(12)
        ],
        "top_bigrams": [
            {"phrase": phrase, "count": count}
            for phrase, count in bigram_counter.most_common(10)
        ],
        "top_labels": [
            {"label": label, "count": count}
            for label, count in labels_counter.most_common(8)
        ],
        "top_projects": [
            {"project": project, "count": count}
            for project, count in project_counter.most_common(5)
        ],
        "status_signals": {
            "open_like": open_like_count,
            "resolved_like": resolved_like_count,
            "high_priority_like": high_priority_like_count,
        },
        "high_level_metrics": {
            "unique_projects": len(project_counter),
            "unique_statuses": len(status_counter),
            "unique_issue_types": len(issue_type_counter),
        },
        "compact_ticket_blocks": compact_ticket_blocks,
    }
