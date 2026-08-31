"""Hybrid retriever with article-cluster expansion for Vietnamese legal text.

This module keeps sub-chunks as search units, but reconstructs one complete
provision per ``version_id`` before returning evidence. Promising provisions
may add a bounded number of sibling clauses from the same article.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from pyvi import ViTokenizer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from ..config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT
from .core import RetrieverCore


ArticleKey = Tuple[str, int]
LocationKey = Tuple[str, str]


class StructuredRetriever(RetrieverCore):
    """RAG retriever that expands seed results to their complete articles."""

    def __init__(
        self,
        use_hybrid: bool = True,
        seed_location_limit: int = 5,
        article_expansion_seed_limit: int = 3,
        final_location_limit: int = 5,
        semantic_candidate_limit: int = 20,
        sibling_expansion_enabled: bool = True,
        max_sibling_source_locations: int = 2,
        sibling_location_limit_per_article: int = 2,
        max_total_sibling_locations: int = 4,
        target_date: Optional[str] = None,
    ):
        self.seed_location_limit = seed_location_limit
        self.article_expansion_seed_limit = article_expansion_seed_limit
        self.final_location_limit = final_location_limit
        self.semantic_candidate_limit = semantic_candidate_limit
        self.sibling_expansion_enabled = sibling_expansion_enabled
        self.max_sibling_source_locations = max_sibling_source_locations
        self.sibling_location_limit_per_article = sibling_location_limit_per_article
        self.max_total_sibling_locations = max_total_sibling_locations
        self.required_facts: List[str] = []
        self.fact_requirements: List[Dict[str, str]] = []
        self.last_retrieval_trace: Dict[str, object] = {}

        # These indexes are populated by the overridden _build_bm25_index().
        # They must exist before super().__init__ invokes that method.
        self.point_cache: Dict[object, Dict] = {}
        self.article_index: Dict[ArticleKey, List[object]] = defaultdict(list)
        self.location_index: Dict[LocationKey, List[object]] = defaultdict(list)
        self.article_context_index: Dict[ArticleKey, List[object]] = defaultdict(list)
        self._tokenized_content_cache: Dict[object, List[str]] = {}
        super().__init__(use_hybrid=use_hybrid, target_date=target_date)

        # Vector-only mode does not call _build_bm25_index in the base class.
        if not use_hybrid:
            self._build_structural_indexes()

    @staticmethod
    def _extract_article_number(location_label: str) -> Optional[int]:
        """Extract an exact relative ``Điều X`` segment from a location label.

        Examples:
            ``Chương I > Điều 3 > Khoản 2`` -> 3
            ``Điều 3 > Khoản 2`` -> 3
            ``Điều 30`` -> 30 (never confused with Điều 3)
        """
        if not isinstance(location_label, str):
            return None

        for segment in location_label.split(">"):
            match = re.fullmatch(r"\s*Điều\s+(\d+)\s*", segment, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_clause_number(location_label: str) -> Optional[int]:
        """Extract an exact relative ``Khoản X`` segment from a location label."""
        if not isinstance(location_label, str):
            return None
        for segment in location_label.split(">"):
            match = re.fullmatch(r"\s*Khoản\s+(\d+)\s*", segment, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _is_article_level(location_label: str) -> bool:
        """Return True only for a location ending at article level."""
        if not isinstance(location_label, str):
            return False
        segments = [segment.strip() for segment in location_label.split(">")]
        has_article = any(
            re.fullmatch(r"Điều\s+\d+", segment, re.IGNORECASE)
            for segment in segments
        )
        has_clause = any(
            re.fullmatch(r"Khoản\s+\d+", segment, re.IGNORECASE)
            for segment in segments
        )
        return has_article and not has_clause

    @classmethod
    def _article_key(cls, chunk: Dict) -> Optional[ArticleKey]:
        metadata = chunk.get("metadata") or {}
        document_id = str(metadata.get("document_id") or "").strip()
        location = str(metadata.get("location_label") or "").strip()
        article_number = cls._extract_article_number(location)
        if not document_id or article_number is None:
            return None
        return document_id, article_number

    @staticmethod
    def _location_key(chunk: Dict) -> Optional[LocationKey]:
        metadata = chunk.get("metadata") or {}
        document_id = str(metadata.get("document_id") or "").strip()
        location = str(metadata.get("location_label") or "").strip()
        if not document_id or not location:
            return None
        return document_id, location

    @staticmethod
    def _point_to_chunk(point) -> Dict:
        payload = point.payload or {}
        return {
            "id": point.id,
            "score": 0.0,
            "content": payload.get("content", ""),
            "original_content": payload.get("original_content", ""),
            "provision_content": (
                payload.get("provision_content")
                or payload.get("original_content", "")
            ),
            "sac_tier1": payload.get("sac_tier1", ""),
            "metadata": payload.get("metadata") or {},
            "retrieval_sources": [],
            "expanded_from": [],
        }

    def _scroll_all_chunks(self) -> List[Dict]:
        all_points = self._scroll_all_points()
        selected_points = self._select_retrievable_points(all_points)
        return [self._point_to_chunk(point) for point in selected_points]

    def _index_chunks(self, chunks: List[Dict]) -> None:
        self.point_cache.clear()
        self.article_index.clear()
        self.location_index.clear()
        self.article_context_index.clear()

        for chunk in chunks:
            chunk_id = chunk["id"]
            self.point_cache[chunk_id] = chunk

            location_key = self._location_key(chunk)
            if location_key is not None:
                self.location_index[location_key].append(chunk_id)

            article_key = self._article_key(chunk)
            if article_key is None:
                continue
            self.article_index[article_key].append(chunk_id)

            location = str((chunk.get("metadata") or {}).get("location_label") or "")
            if self._is_article_level(location):
                self.article_context_index[article_key].append(chunk_id)

        self.retrieval_point_ids = set(self.point_cache)

    def _build_structural_indexes(self) -> None:
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        chunks = self._scroll_all_chunks()
        self._index_chunks(chunks)
        print(
            "  Article indexes built: "
            f"{len(self.article_index)} articles, "
            f"{len(self.location_index)} legal locations"
        )

    def _build_bm25_index(self):
        """Build BM25 and structural indexes in a single Qdrant scroll."""
        print("  Building BM25 and article-cluster indexes from Qdrant...")
        all_chunks = self._scroll_all_chunks()
        self._index_chunks(all_chunks)

        self.bm25_corpus = [chunk["content"] for chunk in all_chunks]
        self.bm25_ids = [chunk["id"] for chunk in all_chunks]
        tokenized_corpus = []
        for chunk in all_chunks:
            chunk_id = chunk["id"]
            tokens = self._tokenized_content_cache.get(chunk_id)
            if tokens is None:
                tokens = ViTokenizer.tokenize(chunk["content"].lower()).split()
                self._tokenized_content_cache[chunk_id] = tokens
            tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(f"  BM25 indexed {len(tokenized_corpus)} chunks")
        print(
            "  Structural index: "
            f"{len(self.article_index)} articles, "
            f"{len(self.location_index)} legal locations"
        )

    def _get_chunk(self, chunk_id: object) -> Optional[Dict]:
        chunk = self.point_cache.get(chunk_id)
        if chunk is None:
            return None
        return {
            **chunk,
            "metadata": dict(chunk.get("metadata") or {}),
            "retrieval_sources": list(chunk.get("retrieval_sources") or []),
            "expanded_from": list(chunk.get("expanded_from") or []),
        }

    def set_required_facts(self, required_facts: Optional[List[str]]) -> None:
        """Set the factual coverage targets for the next retrieval task."""
        self.required_facts = list(
            dict.fromkeys(
                str(fact).strip()
                for fact in (required_facts or [])
                if str(fact).strip()
            )
        )
        self.fact_requirements = [
            {"type": "unspecified", "description": fact}
            for fact in self.required_facts
        ]

    def set_fact_requirements(
        self,
        fact_requirements: Optional[List[Dict]],
        required_facts: Optional[List[str]] = None,
    ) -> None:
        """Preserve typed Agent 2 requirements for retrieval reranking."""

        normalized: List[Dict[str, str]] = []
        seen = set()
        for requirement in fact_requirements or []:
            if not isinstance(requirement, dict):
                continue
            fact_type = str(requirement.get("type") or "unspecified").strip()
            description = str(requirement.get("description") or "").strip()
            if not description:
                continue
            key = (fact_type.casefold(), description.casefold())
            if key in seen:
                continue
            seen.add(key)
            normalized.append(
                {"type": fact_type, "description": description}
            )

        if not normalized:
            normalized = [
                {"type": "unspecified", "description": str(fact).strip()}
                for fact in (required_facts or [])
                if str(fact).strip()
            ]

        self.fact_requirements = normalized
        self.required_facts = [
            requirement["description"] for requirement in normalized
        ]

    @staticmethod
    def _version_key(chunk: Dict) -> str:
        metadata = chunk.get("metadata") or {}
        version_id = str(metadata.get("version_id") or "").strip()
        if version_id:
            return version_id
        provision_key = str(metadata.get("provision_key") or "").strip()
        version = metadata.get("version")
        if provision_key:
            return f"{provision_key}|V{version or 1}"
        return f"POINT|{chunk.get('id')}"

    def _collapse_to_provisions(self, chunks: List[Dict]) -> List[Dict]:
        """Collapse search sub-chunks into complete immutable provisions."""
        grouped: Dict[str, List[Dict]] = defaultdict(list)
        for chunk in chunks:
            grouped[self._version_key(chunk)].append(chunk)

        provisions: List[Dict] = []
        for version_id, members in grouped.items():
            representative = min(
                members,
                key=lambda item: (
                    (
                        -1
                        if (item.get("metadata") or {}).get("sub_chunk_index")
                        is None
                        else int(
                            (item.get("metadata") or {}).get("sub_chunk_index")
                        )
                    ),
                    str(item.get("id")),
                ),
            )
            item = {
                **representative,
                "metadata": dict(representative.get("metadata") or {}),
                "retrieval_sources": [],
                "expanded_from": [],
            }
            full_text = str(
                representative.get("provision_content")
                or representative.get("original_content")
                or ""
            )
            item["provision_content"] = full_text
            item["original_content"] = full_text
            item["source_chunk_ids"] = [str(member["id"]) for member in members]
            item["version_id"] = version_id
            item["score"] = max(float(member.get("score") or 0.0) for member in members)
            if any("rerank_score" in member for member in members):
                item["rerank_score"] = max(
                    float(member.get("rerank_score") or 0.0)
                    for member in members
                )
            for member in members:
                for field in ("retrieval_sources", "expanded_from"):
                    for value in member.get(field) or []:
                        if value not in item[field]:
                            item[field].append(value)
            item["metadata"]["sub_chunk_index"] = None
            provisions.append(item)
        return provisions

    @staticmethod
    def _add_provenance(
        chunk: Dict,
        source: str,
        expanded_from: Optional[object] = None,
    ) -> None:
        sources = chunk.setdefault("retrieval_sources", [])
        if source not in sources:
            sources.append(source)
        if expanded_from is not None:
            parents = chunk.setdefault("expanded_from", [])
            parent_id = str(expanded_from)
            if parent_id not in parents:
                parents.append(parent_id)

    @staticmethod
    def _merge_candidate(target: Dict[object, Dict], incoming: Dict) -> None:
        chunk_id = incoming["id"]
        existing = target.get(chunk_id)
        if existing is None:
            target[chunk_id] = incoming
            return

        existing["score"] = max(
            float(existing.get("score") or 0.0),
            float(incoming.get("score") or 0.0),
        )
        for field in ("retrieval_sources", "expanded_from"):
            values = existing.setdefault(field, [])
            for value in incoming.get(field) or []:
                if value not in values:
                    values.append(value)

    @staticmethod
    def _rerank_document(chunk: Dict) -> str:
        metadata = chunk.get("metadata") or {}
        return (
            f"Văn bản: {metadata.get('document_id', '')}\n"
            f"Loại văn bản: {metadata.get('document_type', '')}\n"
            f"Cấp hiệu lực: {metadata.get('legal_rank', '')}\n"
            f"Vị trí: {metadata.get('location_label', '')}\n"
            f"Tình trạng: {metadata.get('status', '')}\n"
            f"Phiên bản: {metadata.get('version', '')}\n"
            f"Có hiệu lực từ: {metadata.get('valid_from', '')}\n"
            f"Có hiệu lực đến: {metadata.get('valid_to', '')}\n"
            f"Nội dung gốc: {chunk.get('provision_content') or chunk.get('original_content', '')}\n"
            f"Được sửa bởi văn bản: {metadata.get('amended_by_document_id', '')}\n"
            f"Vị trí sửa đổi: {metadata.get('amended_by_source_location', '')}\n"
            f"Loại thay đổi: {metadata.get('change_type', '')}"
        )

    def _score_rerank_pairs(self, pairs: List[List[str]]) -> List[float]:
        """Score every query-document pair in GPU/CPU batches."""

        if not pairs:
            return []
        return [
            float(score)
            for score in self.reranker.predict(
                pairs,
                batch_size=self.reranker_batch_size,
            )
        ]

    def _rerank(self, question: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []
        pairs = [
            [question, self._rerank_document(chunk)]
            for chunk in chunks
        ]
        scores = self._score_rerank_pairs(pairs)
        ranked = []
        for chunk, score in zip(chunks, scores):
            item = {**chunk, "rerank_score": float(score)}
            ranked.append(item)
        return sorted(ranked, key=lambda item: item["rerank_score"], reverse=True)

    def _rerank_with_required_facts(
        self,
        question: str,
        chunks: List[Dict],
    ) -> List[Dict]:
        """Rerank provisions by query relevance and outstanding fact coverage."""
        if not chunks:
            return []
        if not self.fact_requirements:
            return self._rerank(question, chunks)

        fact_type_labels = {
            "applicability": "đối tượng, phạm vi hoặc điều kiện áp dụng",
            "tax_base": "căn cứ tính thuế hoặc thu nhập chịu thuế",
            "tax_rate": "thuế suất hoặc tỷ lệ áp dụng",
            "threshold_or_exemption": "ngưỡng, miễn thuế hoặc không phải nộp",
            "calculation_method": "công thức hoặc phương pháp tính",
            "obligation": "nghĩa vụ pháp lý",
            "procedure": "hồ sơ, biểu mẫu hoặc trình tự thủ tục",
            "deadline": "thời hạn thực hiện",
            "authority": "cơ quan có thẩm quyền hoặc nơi nộp",
        }
        ranking_queries = [question]
        for requirement in self.fact_requirements:
            fact_type = requirement["type"]
            description = requirement["description"]
            type_hint = fact_type_labels.get(fact_type, fact_type)
            ranking_queries.append(
                f"{question}\n"
                f"Loại dữ kiện pháp lý cần tìm: {type_hint}\n"
                f"Mô tả dữ kiện bắt buộc: {description}"
            )

        documents = [self._rerank_document(chunk) for chunk in chunks]
        pairs = [
            [ranking_query, document]
            for ranking_query in ranking_queries
            for document in documents
        ]
        scores = self._score_rerank_pairs(pairs)
        chunk_count = len(chunks)
        score_rows = [
            scores[offset : offset + chunk_count]
            for offset in range(0, len(scores), chunk_count)
        ]
        best_scores = [
            max(float(row[index]) for row in score_rows)
            for index in range(chunk_count)
        ]

        reranked = []
        for chunk, score in zip(chunks, best_scores):
            item = {**chunk}
            item["coverage_rerank_score"] = score
            item["rerank_score"] = item["coverage_rerank_score"]
            reranked.append(item)
        return sorted(
            reranked,
            key=lambda item: item["coverage_rerank_score"],
            reverse=True,
        )

    def _fetch_fused_candidates(
        self,
        vector_candidates: List[Dict],
        fused_ids: List[object],
    ) -> List[Dict]:
        vector_by_id = {chunk["id"]: chunk for chunk in vector_candidates}
        candidates = []
        for chunk_id in fused_ids[: self.semantic_candidate_limit]:
            chunk = vector_by_id.get(chunk_id)
            if chunk is None:
                chunk = self._get_chunk(chunk_id)
            else:
                chunk = {
                    **chunk,
                    "metadata": dict(chunk.get("metadata") or {}),
                    "retrieval_sources": [],
                    "expanded_from": [],
                }
            if chunk is None:
                continue
            self._add_provenance(chunk, "semantic")
            candidates.append(chunk)
        return candidates

    def _select_seed_chunks(self, ranked: List[Dict]) -> List[Dict]:
        """Select at most one seed per legal location."""
        selected = []
        seen_locations: Set[LocationKey] = set()
        for chunk in ranked:
            location_key = self._location_key(chunk)
            if location_key is not None and location_key in seen_locations:
                continue
            selected.append(chunk)
            if location_key is not None:
                seen_locations.add(location_key)
            if len(selected) >= self.seed_location_limit:
                break
        return selected

    def _expand_article_clusters(
        self,
        semantic_candidates: List[Dict],
        seeds: List[Dict],
    ) -> List[Dict]:
        merged: Dict[object, Dict] = {}
        for chunk in semantic_candidates:
            self._merge_candidate(merged, chunk)

        expanded_articles: Set[ArticleKey] = set()
        for seed in seeds[: self.article_expansion_seed_limit]:
            article_key = self._article_key(seed)
            if article_key is None or article_key in expanded_articles:
                continue
            expanded_articles.add(article_key)

            for chunk_id in self.article_index.get(article_key, []):
                chunk = self._get_chunk(chunk_id)
                if chunk is None:
                    continue
                self._add_provenance(
                    chunk,
                    "same_article_expansion",
                    expanded_from=seed["id"],
                )
                self._merge_candidate(merged, chunk)

        return list(merged.values())

    def _select_top_locations(
        self,
        ranked_chunks: List[Dict],
        location_limit: int,
    ) -> Tuple[List[LocationKey], Dict[LocationKey, float]]:
        selected = []
        location_scores: Dict[LocationKey, float] = {}

        # Article-level chunks are attached later as supplemental context. They
        # should not consume one of the substantive location slots intended for
        # clauses (Khoản) or other directly retrieved locations.
        for chunk in ranked_chunks:
            location_key = self._location_key(chunk)
            if location_key is None:
                continue
            if self._is_article_level(location_key[1]):
                continue
            location_scores[location_key] = max(
                location_scores.get(location_key, float("-inf")),
                float(chunk.get("rerank_score") or 0.0),
            )
            if location_key not in selected:
                selected.append(location_key)
            if len(selected) >= location_limit:
                break

        # Documents without clause headings may only have article-level chunks.
        # Use them as a fallback so retrieval still returns a result.
        if len(selected) < location_limit:
            for chunk in ranked_chunks:
                location_key = self._location_key(chunk)
                if location_key is None or location_key in selected:
                    continue
                location_scores[location_key] = max(
                    location_scores.get(location_key, float("-inf")),
                    float(chunk.get("rerank_score") or 0.0),
                )
                selected.append(location_key)
                if len(selected) >= location_limit:
                    break
        return selected, location_scores

    @classmethod
    def _order_sibling_locations(
        cls,
        locations: List[LocationKey],
        location_scores: Dict[LocationKey, float],
        selected_clause_numbers: Set[int],
    ) -> List[LocationKey]:
        """Rank every sibling by fact score, using adjacency only as a tie-break."""

        adjacent = {
            location
            for location in locations
            if any(
                abs(
                    (cls._extract_clause_number(location[1]) or -1000)
                    - clause_number
                )
                == 1
                for clause_number in selected_clause_numbers
            )
        }
        return sorted(
            locations,
            key=lambda location: (
                -location_scores.get(location, float("-inf")),
                0 if location in adjacent else 1,
                cls._extract_clause_number(location[1]) or 0,
            ),
        )

    def _expand_sibling_locations(
        self,
        question: str,
        selected_locations: List[LocationKey],
        ranked_chunks: List[Dict],
        location_scores: Dict[LocationKey, float],
    ) -> Tuple[
        List[LocationKey],
        Dict[LocationKey, float],
        Set[LocationKey],
        Dict[LocationKey, object],
    ]:
        """Add bounded sibling clauses from the same document and article."""
        if not self.sibling_expansion_enabled:
            return selected_locations, location_scores, set(), {}

        expanded_locations = list(selected_locations)
        expanded_scores = dict(location_scores)
        selected_set = set(selected_locations)
        sibling_locations: Set[LocationKey] = set()
        sibling_origins: Dict[LocationKey, object] = {}

        ranked_location_chunks: Dict[LocationKey, List[Dict]] = defaultdict(list)
        for chunk in ranked_chunks:
            location_key = self._location_key(chunk)
            if location_key is not None:
                ranked_location_chunks[location_key].append(chunk)

        selected_articles: List[ArticleKey] = []
        article_origin: Dict[ArticleKey, object] = {}
        selected_clauses: Dict[ArticleKey, Set[int]] = defaultdict(set)
        # Only the strongest final locations may introduce new sibling clauses.
        # All final locations themselves are still preserved in the result.
        sibling_sources = selected_locations[: self.max_sibling_source_locations]
        for location_key in sibling_sources:
            document_id, location = location_key
            article_number = self._extract_article_number(location)
            clause_number = self._extract_clause_number(location)
            if article_number is None or clause_number is None:
                continue
            article_key = (document_id, article_number)
            if article_key not in selected_articles:
                selected_articles.append(article_key)
            selected_clauses[article_key].add(clause_number)
            chunks = ranked_location_chunks.get(location_key) or []
            if chunks:
                article_origin.setdefault(article_key, chunks[0]["id"])

        for article_key in selected_articles:
            if len(sibling_locations) >= self.max_total_sibling_locations:
                break

            article_location_chunks: Dict[LocationKey, List[Dict]] = defaultdict(list)
            for chunk_id in self.article_index.get(article_key, []):
                chunk = self._get_chunk(chunk_id)
                if chunk is None:
                    continue
                location_key = self._location_key(chunk)
                if (
                    location_key is None
                    or self._extract_clause_number(location_key[1]) is None
                    or location_key in selected_set
                ):
                    continue
                article_location_chunks[location_key].append(chunk)

            if not article_location_chunks:
                continue

            sibling_chunks = self._collapse_to_provisions([
                chunk
                for chunks in article_location_chunks.values()
                for chunk in chunks
            ])
            sibling_ranked = self._rerank_with_required_facts(
                question,
                sibling_chunks,
            )
            sibling_score: Dict[LocationKey, float] = {}
            sibling_best_id: Dict[LocationKey, object] = {}
            for chunk in sibling_ranked:
                location_key = self._location_key(chunk)
                if location_key is None:
                    continue
                score = float(chunk.get("rerank_score") or 0.0)
                if location_key not in sibling_score or score > sibling_score[location_key]:
                    sibling_score[location_key] = score
                    sibling_best_id[location_key] = chunk["id"]

            all_locations = list(article_location_chunks)
            # Every sibling has already been reranked against the question and
            # each required fact. Keep that semantic score as the primary
            # signal; adjacency only breaks equal-score ties.
            ordered = self._order_sibling_locations(
                all_locations,
                sibling_score,
                selected_clauses[article_key],
            )

            per_article_added = 0
            for location_key in ordered:
                if per_article_added >= self.sibling_location_limit_per_article:
                    break
                if len(sibling_locations) >= self.max_total_sibling_locations:
                    break
                if location_key in selected_set:
                    continue
                expanded_locations.append(location_key)
                selected_set.add(location_key)
                sibling_locations.add(location_key)
                expanded_scores[location_key] = sibling_score.get(location_key, 0.0)
                sibling_origins[location_key] = article_origin.get(
                    article_key,
                    sibling_best_id.get(location_key),
                )
                per_article_added += 1

        return (
            expanded_locations,
            expanded_scores,
            sibling_locations,
            sibling_origins,
        )

    @classmethod
    def _article_key_from_location(
        cls,
        location_key: LocationKey,
    ) -> Optional[ArticleKey]:
        document_id, location = location_key
        article_number = cls._extract_article_number(location)
        if article_number is None:
            return None
        return document_id, article_number

    def _complete_selected_locations(
        self,
        selected_locations: List[LocationKey],
        ranked_by_id: Dict[object, Dict],
        location_scores: Dict[LocationKey, float],
        sibling_locations: Optional[Set[LocationKey]] = None,
        sibling_origins: Optional[Dict[LocationKey, object]] = None,
    ) -> List[Dict]:
        """Return one complete provision per selected immutable version."""
        final_by_id: Dict[object, Dict] = {}
        sibling_locations = sibling_locations or set()
        sibling_origins = sibling_origins or {}
        selected_articles: Set[ArticleKey] = set()

        for location_key in selected_locations:
            document_id, location = location_key
            article_number = self._extract_article_number(location)
            if article_number is not None:
                selected_articles.add((document_id, article_number))

            for chunk_id in self.location_index.get(location_key, []):
                chunk = ranked_by_id.get(chunk_id) or self._get_chunk(chunk_id)
                if chunk is None:
                    continue
                chunk = {**chunk}
                chunk["location_score"] = location_scores.get(location_key, 0.0)
                if location_key in sibling_locations:
                    self._add_provenance(
                        chunk,
                        "sibling_clause_expansion",
                        expanded_from=sibling_origins.get(location_key),
                    )
                self._add_provenance(chunk, "provision_reconstruction")
                self._merge_candidate(final_by_id, chunk)

        # Article-level chunks are supplemental context and do not consume one
        # of the five selected substantive locations.
        for article_key in selected_articles:
            for chunk_id in self.article_context_index.get(article_key, []):
                chunk = ranked_by_id.get(chunk_id) or self._get_chunk(chunk_id)
                if chunk is None:
                    continue
                self._add_provenance(chunk, "article_context")
                self._merge_candidate(final_by_id, chunk)

        collapsed = self._collapse_to_provisions(list(final_by_id.values()))

        def sort_key(chunk: Dict):
            metadata = chunk.get("metadata") or {}
            location = str(metadata.get("location_label") or "")
            location_key = self._location_key(chunk)
            selected_order = (
                selected_locations.index(location_key)
                if location_key in selected_locations
                else len(selected_locations)
            )
            return (
                selected_order,
                location,
                int(metadata.get("version") or 1),
            )

        return sorted(collapsed, key=sort_key)

    def _hybrid_retrieve(self, question: str, limit: int = 5) -> List[Dict]:
        """Hybrid retrieval followed by article-cluster expansion."""
        retrieval_started = time.perf_counter()
        vector_candidates = self._vector_retrieve(
            question,
            limit=self.semantic_candidate_limit,
        )
        bm25_results = self._bm25_search(
            question,
            limit=self.semantic_candidate_limit,
        )
        fused_ids = self._reciprocal_rank_fusion(
            vector_candidates,
            bm25_results,
        )
        semantic_candidates = self._fetch_fused_candidates(
            vector_candidates,
            fused_ids,
        )
        semantic_seconds = time.perf_counter() - retrieval_started

        preliminary_started = time.perf_counter()
        preliminary_ranked = self._rerank(question, semantic_candidates)
        seeds = self._select_seed_chunks(preliminary_ranked)
        preliminary_seconds = time.perf_counter() - preliminary_started

        expansion_started = time.perf_counter()
        expanded_candidates = self._expand_article_clusters(
            semantic_candidates,
            seeds,
        )
        expanded_provisions = self._collapse_to_provisions(expanded_candidates)
        expansion_seconds = time.perf_counter() - expansion_started

        fact_rerank_started = time.perf_counter()
        final_ranked = self._rerank_with_required_facts(
            question,
            expanded_provisions,
        )
        fact_rerank_seconds = time.perf_counter() - fact_rerank_started

        context_started = time.perf_counter()
        location_limit = limit or self.final_location_limit
        selected_locations, location_scores = self._select_top_locations(
            final_ranked,
            location_limit=location_limit,
        )
        ranked_by_id = {chunk["id"]: chunk for chunk in final_ranked}
        main_locations = list(selected_locations)
        main_scores = dict(location_scores)
        main_context = self._complete_selected_locations(
            main_locations,
            ranked_by_id,
            main_scores,
        )
        (
            selected_locations,
            location_scores,
            sibling_locations,
            sibling_origins,
        ) = self._expand_sibling_locations(
            question,
            selected_locations,
            final_ranked,
            location_scores,
        )
        final_context = self._complete_selected_locations(
            selected_locations,
            ranked_by_id,
            location_scores,
            sibling_locations=sibling_locations,
            sibling_origins=sibling_origins,
        )
        context_seconds = time.perf_counter() - context_started
        # Diagnostic-only ranking views. Supplemental article context remains
        # in main_context/final_context and therefore in the production output,
        # but it must not consume a ranked location slot in ablation metrics.
        main_location_set = set(main_locations)
        final_location_set = set(selected_locations)
        stage_b_ranked_locations = [
            chunk
            for chunk in main_context
            if self._location_key(chunk) in main_location_set
        ]
        stage_c_ranked_locations = [
            chunk
            for chunk in final_context
            if self._location_key(chunk) in final_location_set
        ]
        self.last_retrieval_trace = {
            "query": question,
            # Keep the 20-item list for diagnostics, but score the actual
            # Hybrid + Rerank output: at most five seed legal locations.
            "stage_a_hybrid_candidate_pool": list(preliminary_ranked),
            "stage_a_hybrid_reranked_seed_locations": list(seeds),
            "stage_b_ranked_locations": stage_b_ranked_locations,
            "stage_b_top_locations_full_provisions": list(main_context),
            "stage_c_ranked_locations_with_siblings": stage_c_ranked_locations,
            "stage_c_full_context_with_siblings": list(final_context),
            "seed_chunk_ids": [str(chunk.get("id")) for chunk in seeds],
            "main_location_count": len(main_locations),
            "sibling_location_count": len(sibling_locations),
            "timing_seconds": {
                "semantic_retrieval": round(semantic_seconds, 3),
                "preliminary_rerank": round(preliminary_seconds, 3),
                "article_expansion": round(expansion_seconds, 3),
                "required_fact_rerank": round(fact_rerank_seconds, 3),
                "context_and_sibling_expansion": round(context_seconds, 3),
                "total": round(time.perf_counter() - retrieval_started, 3),
            },
        }
        return final_context
