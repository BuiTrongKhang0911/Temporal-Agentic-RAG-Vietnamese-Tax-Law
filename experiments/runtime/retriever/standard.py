"""Subchunk-only Retriever used by the standard experimental baselines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

from .base import RAGQuestionAnswering


class StandardBaselineRetriever(RAGQuestionAnswering):
    """Provide Dense, BM25, RRF, and raw-subchunk reranking primitives.

    The class deliberately contains no Article expansion, provision
    restoration, required-fact reranking, location deduplication, or sibling
    expansion. Temporal filtering can be disabled for blind end-to-end
    baselines and retained for the locked Retriever experiment.
    """

    def __init__(
        self,
        *,
        disable_temporal_filtering: bool = False,
        semantic_candidate_limit: int = 20,
    ) -> None:
        self.disable_temporal_filtering = disable_temporal_filtering
        self.semantic_candidate_limit = semantic_candidate_limit
        self.point_cache: Dict[object, Dict[str, Any]] = {}
        super().__init__(use_hybrid=True)

    def _select_applicable_points(self, points: List[object]) -> List[object]:
        if self.disable_temporal_filtering:
            return list(points)
        return super()._select_applicable_points(points)

    @staticmethod
    def _point_to_chunk(point: object) -> Dict[str, Any]:
        payload = getattr(point, "payload", None) or {}
        return {
            "id": getattr(point, "id"),
            "score": 0.0,
            "content": payload.get("content", ""),
            "original_content": payload.get("original_content", ""),
            "provision_content": (
                payload.get("provision_content")
                or payload.get("original_content", "")
            ),
            "sac_tier1": payload.get("sac_tier1", ""),
            "metadata": payload.get("metadata") or {},
        }

    def _build_bm25_index(self) -> None:
        points = self._select_retrievable_points(self._scroll_all_points())
        chunks = [self._point_to_chunk(point) for point in points]
        self.point_cache = {chunk["id"]: chunk for chunk in chunks}
        self.retrieval_point_ids = set(self.point_cache)
        self.bm25_corpus = [str(chunk["content"]) for chunk in chunks]
        self.bm25_ids = [chunk["id"] for chunk in chunks]
        tokenized = [
            ViTokenizer.tokenize(content.lower()).split()
            for content in self.bm25_corpus
        ]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  Standard BM25 indexed {len(tokenized)} subchunks")

    def _get_chunk(self, chunk_id: object) -> Optional[Dict[str, Any]]:
        chunk = self.point_cache.get(chunk_id)
        if chunk is None:
            return None
        return {
            **chunk,
            "metadata": dict(chunk.get("metadata") or {}),
        }

    def fetch_fused_candidates(
        self,
        vector_candidates: List[Dict[str, Any]],
        fused_ids: List[object],
    ) -> List[Dict[str, Any]]:
        vector_by_id = {chunk["id"]: chunk for chunk in vector_candidates}
        candidates = []
        for chunk_id in fused_ids[: self.semantic_candidate_limit]:
            chunk = vector_by_id.get(chunk_id) or self._get_chunk(chunk_id)
            if chunk is not None:
                candidates.append(chunk)
        return candidates
