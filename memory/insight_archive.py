"""Insight archive backed by ChromaDB."""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional

import chromadb

from config import Config

from .models import InsightEntry

logger = logging.getLogger("Memory.InsightArchive")

EmbeddingFn = Callable[[List[str]], List[List[float]]]


def _default_embedding_fn() -> Optional[EmbeddingFn]:
    """Reuse the same local Qwen embedding family as the main RAG path."""
    try:
        from rag.retriever_v2_mix_Reranking import QwenEmbeddingFunction

        project_root = Path(__file__).resolve().parent.parent
        model_candidates = [
            project_root / "models" / "qwen3-embedding",
            project_root / "models" / "Qwen3-Embedding-0.6B",
        ]
        model_path = next((path for path in model_candidates if path.exists()), None)
        if model_path is None:
            logger.warning("Local Qwen embedding model not found for insight archive")
            return None

        embedder = QwenEmbeddingFunction(model_path)

        def _encode(texts: List[str]) -> List[List[float]]:
            return embedder.embed_documents(texts)

        return _encode
    except Exception as exc:
        logger.warning("Insight archive embedding init failed: %s", exc)
        return None


class InsightArchive:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
    ):
        self.db_path = Path(db_path or Config.MEMORY_INSIGHT_DB)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or Config.MEMORY_INSIGHT_COLLECTION

        self.embedding_fn = embedding_fn or _default_embedding_fn()
        self.fallback_mode = self.embedding_fn is None

        self._lock = threading.Lock()
        self._client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "LLM extracted user insights"},
        )

        logger.info(
            "Insight archive ready path=%s collection=%s fallback=%s count=%d",
            self.db_path,
            self.collection_name,
            self.fallback_mode,
            self._collection.count(),
        )

    def _ensure_embedding(self, entry: InsightEntry) -> Optional[List[float]]:
        if entry.embedding:
            return entry.embedding
        if self.fallback_mode or self.embedding_fn is None:
            return None
        try:
            vector = self.embedding_fn([entry.content])[0]
            entry.embedding = vector
            return vector
        except Exception as exc:
            logger.error("Insight embedding failed: %s", exc)
            return None

    def _commit_sync(self, entry: InsightEntry) -> bool:
        with self._lock:
            embedding = self._ensure_embedding(entry)
            metadata = entry.to_chroma_metadata()
            try:
                add_kwargs: Dict = {
                    "ids": [entry.insight_id],
                    "documents": [entry.content],
                    "metadatas": [metadata],
                }
                if embedding is not None:
                    add_kwargs["embeddings"] = [embedding]
                self._collection.upsert(**add_kwargs)
                logger.info(
                    "[INSIGHT] committed id=%s topic=%s user=%s content=%s entities=%s",
                    entry.insight_id,
                    entry.topic or "-",
                    entry.user_id,
                    entry.content,
                    entry.key_entities,
                )
                return True
            except Exception as exc:
                logger.error("Insight commit failed: %s", exc)
                return False

    async def commit_insight(self, entry: InsightEntry) -> bool:
        return await asyncio.to_thread(self._commit_sync, entry)

    def commit_insight_sync(self, entry: InsightEntry) -> bool:
        return self._commit_sync(entry)

    def _build_where(self, user_id: Optional[str], session_id: Optional[str]) -> Optional[Dict]:
        clauses: List[Dict] = []
        if user_id:
            clauses.append({"user_id": user_id})
        if session_id:
            clauses.append({"session_id": session_id})
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def search_insights(
        self,
        query_vector: List[float],
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[InsightEntry]:
        where = self._build_where(user_id, session_id)
        try:
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("Insight vector search failed: %s", exc)
            return []

        entries: List[InsightEntry] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(documents, metadatas):
            try:
                entries.append(InsightEntry.from_chroma(doc, meta or {}))
            except Exception as exc:
                logger.warning("Skip invalid insight payload: %s", exc)
        return entries

    def search_by_text(
        self,
        query: str,
        top_k: int = 3,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[InsightEntry]:
        if self.fallback_mode or self.embedding_fn is None:
            return self._keyword_fallback(query, top_k, user_id, session_id)
        try:
            vector = self.embedding_fn([query])[0]
        except Exception as exc:
            logger.error("Insight query encode failed, fallback to keyword: %s", exc)
            return self._keyword_fallback(query, top_k, user_id, session_id)
        return self.search_insights(vector, top_k, user_id, session_id)

    def _keyword_fallback(
        self,
        query: str,
        top_k: int,
        user_id: Optional[str],
        session_id: Optional[str],
    ) -> List[InsightEntry]:
        query_lc = (query or "").lower()
        if not query_lc:
            return []
        where = self._build_where(user_id, session_id)
        try:
            payload = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.error("Insight keyword fallback failed: %s", exc)
            return []

        scored: List = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            text = (doc or "").lower()
            score = 0
            for token in query_lc.split():
                if token and token in text:
                    score += 1
            if score > 0:
                scored.append((score, doc, meta or {}))
        scored.sort(key=lambda item: item[0], reverse=True)

        entries: List[InsightEntry] = []
        for _, doc, meta in scored[:top_k]:
            try:
                entries.append(InsightEntry.from_chroma(doc, meta))
            except Exception:
                continue
        return entries

    def delete_insight(self, insight_id: str) -> bool:
        with self._lock:
            try:
                self._collection.delete(ids=[insight_id])
                logger.info("Insight deleted %s", insight_id)
                return True
            except Exception as exc:
                logger.error("Insight delete failed: %s", exc)
                return False

    def list_user_insights(self, user_id: str, limit: int = 100) -> List[InsightEntry]:
        try:
            payload = self._collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"],
                limit=limit,
            )
        except Exception as exc:
            logger.error("List user insights failed: %s", exc)
            return []

        entries: List[InsightEntry] = []
        for doc, meta in zip(payload.get("documents", []), payload.get("metadatas", [])):
            try:
                entries.append(InsightEntry.from_chroma(doc, meta or {}))
            except Exception:
                continue
        return entries

    def get_stats(self) -> Dict:
        try:
            return {
                "total_insights": self._collection.count(),
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
            }
        except Exception:
            return {
                "total_insights": 0,
                "db_path": str(self.db_path),
                "collection": self.collection_name,
                "fallback_mode": self.fallback_mode,
                "status": "error",
            }
