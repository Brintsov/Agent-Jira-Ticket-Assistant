import sqlite3
import pandas as pd

from typing import Optional, Any
from langchain_chroma import Chroma


class TicketRepository:
    def __init__(self, tickets_db_path, vector_db_path, embedding_model):
        sql_conn = sqlite3.connect(tickets_db_path, check_same_thread=False)
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=vector_db_path,
        )
        self.sql_conn = sql_conn
        self.vectorstore = vectorstore

    def get_by_keys(self, keys: list[str]) -> list[dict]:
        if not keys:
            return []

        placeholders = ",".join("?" for _ in keys)
        query = f'''
            SELECT *
            FROM tickets
            WHERE "key" IN ({placeholders})
        '''
        df = pd.read_sql_query(query, self.sql_conn, params=keys)

        # preserve requested order if needed
        order_map = {k: i for i, k in enumerate(keys)}
        if "key" in df.columns:
            df["_order"] = df["key"].map(order_map)
            df = df.sort_values("_order").drop(columns=["_order"])

        return df.to_dict(orient="records")

    def filter_tickets(
        self,
        project_key: Optional[str] = None,
        status_name: Optional[str] = None,
        resolution_name: Optional[str] = None,
        issue_type_name: Optional[str] = None,
        priority_name: Optional[str] = None,
        summary_contains: Optional[str] = None,
        description_contains: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict]:
        conditions = []
        params: list[Any] = []

        exact_filters = {
            "project.key": project_key,
            "status.name": status_name,
            "resolution.name": resolution_name,
            "issuetype.name": issue_type_name,
            "priority.name": priority_name,
        }

        for col, value in exact_filters.items():
            if value:
                conditions.append(f'LOWER(COALESCE("{col}", "")) = LOWER(?)')
                params.append(value)

        if summary_contains:
            conditions.append('LOWER(COALESCE("summary", "")) LIKE LOWER(?)')
            params.append(f"%{summary_contains}%")

        if description_contains:
            conditions.append('LOWER(COALESCE("description", "")) LIKE LOWER(?)')
            params.append(f"%{description_contains}%")

        query = 'SELECT * FROM tickets'
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += ' ORDER BY COALESCE("updated", "created") DESC'
        if limit:
            query += ' LIMIT ?'
            params.append(limit)

        df = pd.read_sql_query(query, self.sql_conn, params=params)
        return df.to_dict(orient="records")

    def semantic_search(
        self,
        query: str,
        keys: Optional[list[str]] = None,
        k: int = 5,
        search_type: str = "similarity",
    ) -> list[dict]:
        if not query.strip():
            return []

        if keys:
            docs = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
                filter={"key": {"$in": keys}},
            )
        else:
            docs = self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=k,
            )

        results = []
        for rank, (doc, score) in enumerate(docs, start=1):
            if score < 0.5:
                continue
            results.append({
                "rank": rank,
                "key": doc.metadata.get("key"),
                "content": doc.page_content,
                "metadata": doc.metadata,
            })
        return results

    def hybrid_search(
        self,
        semantic_query: str,
        sql_filters: Optional[dict] = None,
        k: int = 5,
    ) -> list[dict]:
        sql_filters = sql_filters or {}

        filtered_rows = self.filter_tickets(**sql_filters)
        keys = [row["key"] for row in filtered_rows if row.get("key")]

        if not keys:
            return []

        semantic_hits = self.semantic_search(
            query=semantic_query,
            keys=keys,
            k=k,
            search_type="similarity",
        )

        best_keys = []
        seen = set()
        for hit in semantic_hits:
            key = hit.get("key")
            if key and key not in seen:
                best_keys.append(key)
                seen.add(key)

        return self.get_by_keys(best_keys)
