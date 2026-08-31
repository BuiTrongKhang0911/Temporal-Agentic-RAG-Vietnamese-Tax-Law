"""Shared retrieval infrastructure for the production Structured Retriever."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from pyvi import ViTokenizer
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ..config import (
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MODEL,
)
from ..model_registry import get_embedding_model
from .temporal import (
    metadata_overlaps_scope,
    normalize_temporal_scope,
    select_points_for_scope,
    temporal_scope_key,
)


class RetrieverCore:
    """Own models, Qdrant access, temporal filtering, Dense, BM25, and RRF."""

    def __init__(
        self,
        use_hybrid: bool = True,
        target_date: Optional[str] = None,
    ) -> None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = get_embedding_model(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
            local_files_only=True,
        )
        print(f"Loading reranker model: {RERANKER_MODEL} on {RERANKER_DEVICE}")
        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device=RERANKER_DEVICE,
            local_files_only=True,
        )
        self.reranker_batch_size = RERANKER_BATCH_SIZE

        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        self.use_hybrid = use_hybrid
        initial_scope = (
            {"type": "exact_date", "date": target_date}
            if target_date
            else {"type": "current"}
        )
        self.temporal_scope = normalize_temporal_scope(initial_scope)
        self.target_date = self.temporal_scope.get("date")
        self._temporal_scope_key = temporal_scope_key(self.temporal_scope)
        self._all_points_cache: Optional[List[object]] = None
        self.retrieval_point_ids: Set[object] = set()
        self.bm25 = None
        self.bm25_corpus: List[str] = []
        self.bm25_ids: List[object] = []

        if self.use_hybrid:
            self._build_bm25_index()
        else:
            self._build_temporal_point_ids()

        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    def set_temporal_scope(self, scope: Dict) -> None:
        """Apply one Python-validated temporal scope to all retrieval paths."""

        normalized = normalize_temporal_scope(scope)
        scope_key = temporal_scope_key(normalized)
        if scope_key == self._temporal_scope_key:
            return
        self.temporal_scope = normalized
        self.target_date = normalized.get("date")
        self._temporal_scope_key = scope_key
        if self.use_hybrid:
            self._build_bm25_index()
        elif hasattr(self, "_build_structural_indexes"):
            self._build_structural_indexes()
        else:
            self._build_temporal_point_ids()

    def _is_temporally_applicable(self, metadata: Dict) -> bool:
        return metadata_overlaps_scope(metadata, self.temporal_scope)

    def _select_applicable_points(self, points: List[object]) -> List[object]:
        return select_points_for_scope(points, self.temporal_scope)

    @staticmethod
    def _is_article_level_location(location_label: object) -> bool:
        if not isinstance(location_label, str):
            return False
        segments = [segment.strip() for segment in location_label.split(">")]
        has_article = any(
            re.fullmatch(r"Điều\s+\d+[a-zđ]*", segment, re.IGNORECASE)
            for segment in segments
        )
        has_clause = any(
            re.fullmatch(r"Khoản\s+\d+[a-zđ]*", segment, re.IGNORECASE)
            for segment in segments
        )
        return has_article and not has_clause

    @classmethod
    def _has_substantive_article_content(cls, payload: Dict) -> bool:
        metadata = payload.get("metadata") or {}
        if not cls._is_article_level_location(metadata.get("location_label")):
            return True

        raw_content = str(
            payload.get("provision_content")
            or payload.get("original_content")
            or ""
        ).strip()
        if not raw_content:
            return False

        headingless = re.sub(r"^\s*#{1,6}\s*", "", raw_content).strip()
        normalized_content = re.sub(r"\s+", " ", headingless).strip()
        article_heading = re.sub(
            r"\s+", " ", str(metadata.get("Dieu") or "").strip()
        )
        if article_heading:
            content_folded = normalized_content.casefold()
            heading_folded = article_heading.casefold()
            if content_folded == heading_folded:
                return False
            if content_folded.startswith(heading_folded):
                remainder = normalized_content[len(article_heading):].strip(
                    " \t\r\n:;.-"
                )
                return bool(re.search(r"\w", remainder, re.UNICODE))

        lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
        if lines and re.match(
            r"^#{1,6}\s*Điều\s+\d+[a-zđ]*\b",
            lines[0],
            re.IGNORECASE,
        ):
            return len(lines) > 1
        return True

    @classmethod
    def _is_retrievable_point(cls, point: object) -> bool:
        payload = getattr(point, "payload", None) or {}
        return cls._has_substantive_article_content(payload)

    def _select_retrievable_points(self, points: List[object]) -> List[object]:
        return [
            point
            for point in self._select_applicable_points(points)
            if self._is_retrievable_point(point)
        ]

    def _scroll_all_points(self) -> List[object]:
        if self._all_points_cache is not None:
            return list(self._all_points_cache)
        offset = None
        all_points = []
        while True:
            points, offset = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if offset is None:
                break
        self._all_points_cache = all_points
        return list(all_points)

    def _active_point_filter(self) -> Optional[models.Filter]:
        if not self.retrieval_point_ids:
            return None
        return models.Filter(
            must=[models.HasIdCondition(has_id=list(self.retrieval_point_ids))]
        )

    def _build_temporal_point_ids(self) -> None:
        selected = self._select_retrievable_points(self._scroll_all_points())
        self.retrieval_point_ids = {point.id for point in selected}

    def _build_bm25_index(self) -> None:
        points = self._select_retrievable_points(self._scroll_all_points())
        self.retrieval_point_ids = {point.id for point in points}
        self.bm25_corpus = [
            str((point.payload or {}).get("content", "")) for point in points
        ]
        self.bm25_ids = [point.id for point in points]
        tokenized = [
            ViTokenizer.tokenize(content.lower()).split()
            for content in self.bm25_corpus
        ]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25_search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Tuple[object, float]]:
        if self.bm25 is None:
            return []
        tokens = ViTokenizer.tokenize(query.lower()).split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]
        return [
            (self.bm25_ids[index], float(scores[index]))
            for index in top_indices
            if scores[index] > 0
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: List[Dict],
        bm25_results: List[Tuple[object, float]],
        k: int = 60,
    ) -> List[object]:
        scores = defaultdict(float)
        for rank, chunk in enumerate(vector_results, 1):
            scores[chunk["id"]] += 1 / (k + rank)
        for rank, (chunk_id, _) in enumerate(bm25_results, 1):
            scores[chunk_id] += 1 / (k + rank)
        return sorted(scores, key=scores.get, reverse=True)

    def retrieve_chunks(self, question: str, limit: int = 5) -> List[Dict]:
        if self.use_hybrid:
            return self._hybrid_retrieve(question, limit)
        return self._vector_retrieve(question, limit)

    def _vector_retrieve(self, question: str, limit: int = 5) -> List[Dict]:
        if not self.retrieval_point_ids:
            return []
        embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True,
        ).tolist()
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            query_filter=self._active_point_filter(),
            limit=limit,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload["content"],
                "original_content": hit.payload["original_content"],
                "provision_content": (
                    hit.payload.get("provision_content")
                    or hit.payload["original_content"]
                ),
                "sac_tier1": hit.payload.get("sac_tier1", ""),
                "metadata": hit.payload["metadata"],
            }
            for hit in results.points
        ]

    def _hybrid_retrieve(self, question: str, limit: int) -> List[Dict]:
        raise NotImplementedError
