"""
Bước 4: RAG Question Answering với Hybrid Search
- Nhận câu hỏi từ người dùng
- Hybrid Search: Vector (bge-m3) + Sparse (BM25 với pyvi) với RRF fusion
- BGE-Reranker: Rerank kết quả sau RRF fusion
- Generate answer bằng Gemini với context từ chunks
"""
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from pyvi import ViTokenizer
from typing import List, Dict, Tuple, Optional, Set
from datetime import date
import json
import re
from collections import defaultdict
from ..config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RERANKER_MODEL,
)
from .temporal import (
    describe_temporal_scope,
    metadata_overlaps_scope,
    normalize_temporal_scope,
    select_points_for_scope,
    temporal_scope_key,
)
from ..model_registry import get_embedding_model


class RAGQuestionAnswering:
    def __init__(
        self,
        use_hybrid: bool = True,
        target_date: Optional[str] = None,
    ):
        # Configure Gemini for generation
        genai.configure(api_key=GOOGLE_API_KEY)
        self.llm = genai.GenerativeModel(GEMINI_MODEL)
        
        # Load embedding model for retrieval
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = get_embedding_model(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
        )
        
        # Load reranker model (BGE family for consistency)
        print(f"Loading reranker model: {RERANKER_MODEL} on {RERANKER_DEVICE}")
        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device=RERANKER_DEVICE,
        )
        self.reranker_batch_size = RERANKER_BATCH_SIZE
        
        # Qdrant client
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection_name = QDRANT_COLLECTION
        
        # Hybrid search settings
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
        self.bm25_corpus = []
        self.bm25_ids = []
        
        if self.use_hybrid:
            print(f"Initializing hybrid search (Vector + BM25)...")
            self._build_bm25_index()
        else:
            self._build_temporal_point_ids()
        
        print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"LLM Model: {GEMINI_MODEL}")
        print(f"Search Mode: {'Hybrid (Vector + BM25 + Reranker)' if self.use_hybrid else 'Vector only'}")
        print(f"Legal temporal filter: {describe_temporal_scope(self.temporal_scope)}")
        print(f"Ready for questions!\n")

    def set_temporal_scope(self, scope: Dict) -> None:
        """Apply a Python-validated scope to every vector/BM25/tool query."""

        normalized = normalize_temporal_scope(scope)
        scope_key = temporal_scope_key(normalized)
        if scope_key == self._temporal_scope_key:
            return

        self.temporal_scope = normalized
        self.target_date = normalized.get("date")
        self._temporal_scope_key = scope_key
        print(f"Applying temporal scope: {describe_temporal_scope(normalized)}")
        if self.use_hybrid:
            self._build_bm25_index()
        elif hasattr(self, "_build_structural_indexes"):
            self._build_structural_indexes()
        else:
            self._build_temporal_point_ids()

    @staticmethod
    def _date_is_on_or_before(value: object, target: str) -> bool:
        if value in (None, ""):
            return True
        return str(value)[:10] <= target

    @staticmethod
    def _date_is_after(value: object, target: str) -> bool:
        if value in (None, ""):
            return True
        return str(value)[:10] > target

    def _is_temporally_applicable(self, metadata: Dict) -> bool:
        """Evaluate the half-open legal interval against the locked scope."""
        return metadata_overlaps_scope(metadata, self.temporal_scope)

    def _select_applicable_points(self, points: List[object]) -> List[object]:
        """Keep points matching the locked current/date/period scope."""
        return select_points_for_scope(points, self.temporal_scope)

    @staticmethod
    def _is_article_level_location(location_label: object) -> bool:
        """Return True when a location stops at Điều and has no Khoản."""
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
        """Keep an Điều point only when it contains more than its heading."""
        metadata = payload.get("metadata") or {}
        location = metadata.get("location_label")
        if not cls._is_article_level_location(location):
            return True

        raw_content = str(
            payload.get("provision_content")
            or payload.get("original_content")
            or ""
        ).strip()
        if not raw_content:
            return False

        # Strip Markdown heading syntax before comparing with metadata["Dieu"].
        headingless = re.sub(r"^\s*#{1,6}\s*", "", raw_content).strip()
        normalized_content = re.sub(r"\s+", " ", headingless).strip()
        article_heading = re.sub(
            r"\s+",
            " ",
            str(metadata.get("Dieu") or "").strip(),
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

        # Fallback for older payloads without metadata["Dieu"]: a heading-only
        # point has no non-empty line after the first Markdown heading.
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
        """Apply temporal rules and remove heading-only Điều points."""
        applicable = self._select_applicable_points(points)
        return [
            point
            for point in applicable
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
        """Prepare the deterministic allow-list used by vector-only retrieval."""

        selected = self._select_retrievable_points(self._scroll_all_points())
        self.retrieval_point_ids = {point.id for point in selected}
    
    def _build_bm25_index(self):
        """
        Build BM25 index từ tất cả chunks trong Qdrant
        QUAN TRỌNG: 
        - Index trên 'content' (có SAC Tier 1 + Tier 2)
        - Dùng pyvi để tách từ tiếng Việt (word segmentation)
        """
        print(f"  Building BM25 index from Qdrant...")
        
        # Cache the immutable corpus once; rebuild only the scoped in-memory index.
        all_chunks = self._scroll_all_points()
        all_chunks = self._select_retrievable_points(all_chunks)
        self.retrieval_point_ids = {point.id for point in all_chunks}
        all_chunks = [
            {
                "id": point.id,
                "content": (point.payload or {}).get("content", ""),
            }
            for point in all_chunks
        ]

        # Tokenize corpus với pyvi
        self.bm25_corpus = [chunk["content"] for chunk in all_chunks]
        self.bm25_ids = [chunk["id"] for chunk in all_chunks]
        
        print(f"  Tokenizing {len(self.bm25_corpus)} documents with pyvi...")
        # Tách từ tiếng Việt: "chuyển nhượng tài sản" -> "chuyển_nhượng tài_sản"
        tokenized_corpus = [
            ViTokenizer.tokenize(text.lower()).split()
            for text in self.bm25_corpus
        ]
        
        # Build BM25
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"  ✓ BM25 index built with {len(tokenized_corpus)} documents")
        print(f"  ✓ Indexed on augmented content with Vietnamese word segmentation")
    
    def _bm25_search(self, query: str, limit: int = 20) -> List[Tuple[int, float]]:
        """
        BM25 search với Vietnamese word segmentation
        
        Returns: List of (chunk_id, score) sorted by score
        """
        if self.bm25 is None:
            return []
        
        # Tokenize query với pyvi: "chuyển nhượng tài sản" -> "chuyển_nhượng tài_sản"
        tokenized_query = ViTokenizer.tokenize(query.lower()).split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:limit]
        
        results = [(self.bm25_ids[i], scores[i]) for i in top_indices if scores[i] > 0]
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[int]:
        """
        Reciprocal Rank Fusion (RRF) để merge vector và BM25 results
        
        Formula: RRF(d) = Σ 1/(k + rank(d))
        
        Returns: List of chunk IDs sorted by fused score
        """
        rrf_scores = defaultdict(float)
        
        # Add vector results
        for rank, chunk in enumerate(vector_results, 1):
            chunk_id = chunk["id"]
            rrf_scores[chunk_id] += 1 / (k + rank)
        
        # Add BM25 results
        for rank, (chunk_id, score) in enumerate(bm25_results, 1):
            rrf_scores[chunk_id] += 1 / (k + rank)
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        return sorted_ids
    
    def retrieve_chunks(self, question: str, limit: int = 5) -> List[Dict]:
        """
        Hybrid Search: Vector + BM25 với RRF fusion
        
        Returns: List of chunks với score và metadata
        """
        if self.use_hybrid:
            return self._hybrid_retrieve(question, limit)
        else:
            return self._vector_retrieve(question, limit)
    
    def _vector_retrieve(self, question: str, limit: int = 5) -> List[Dict]:
        """
        Pure vector search (bge-m3)
        """
        # Never turn an empty temporal result into an unfiltered vector query.
        if not self.retrieval_point_ids:
            return []

        # Generate embedding
        query_embedding = self.embedding_model.encode(
            question,
            convert_to_numpy=True
        ).tolist()
        
        # Search in Qdrant
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=self._active_point_filter(),
            limit=limit
        )
        
        # Format results
        chunks = []
        for hit in results.points:
            chunks.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload["content"],
                "original_content": hit.payload["original_content"],
                "provision_content": (
                    hit.payload.get("provision_content")
                    or hit.payload["original_content"]
                ),
                "sac_tier1": hit.payload["sac_tier1"],
                "metadata": hit.payload["metadata"]
            })
        
        return chunks
    
    def _hybrid_retrieve(self, question: str, limit: int = 5) -> List[Dict]:
        """
        Hybrid search: Vector (bge-m3) + BM25 với RRF fusion + BGE-Reranker
        """
        # Step 1: Vector search (get more candidates for fusion)
        vector_candidates = self._vector_retrieve(question, limit=20)
        
        # Step 2: BM25 search
        bm25_results = self._bm25_search(question, limit=20)
        
        # Step 3: RRF fusion
        fused_ids = self._reciprocal_rank_fusion(vector_candidates, bm25_results)
        
        # Step 4: Fetch candidate chunks for reranking (get more than limit for reranking)
        rerank_limit = min(20, len(fused_ids))  # Get top 20 for reranking
        candidate_chunks = []
        
        for chunk_id in fused_ids[:rerank_limit]:
            # Find chunk in vector_candidates
            for chunk in vector_candidates:
                if chunk["id"] == chunk_id:
                    candidate_chunks.append(chunk)
                    break
            else:
                # If not in vector results, fetch from Qdrant
                try:
                    point = self.qdrant_client.retrieve(
                        collection_name=self.collection_name,
                        ids=[chunk_id]
                    )[0]
                    
                    candidate_chunks.append({
                        "id": point.id,
                        "score": 0.0,  # RRF score, not vector score
                        "content": point.payload["content"],
                        "original_content": point.payload["original_content"],
                        "provision_content": (
                            point.payload.get("provision_content")
                            or point.payload["original_content"]
                        ),
                        "sac_tier1": point.payload["sac_tier1"],
                        "metadata": point.payload["metadata"]
                    })
                except:
                    continue
        
        # Step 5: Rerank using BGE-Reranker
        if len(candidate_chunks) > 0:
            # Prepare pairs for reranking: (query, document)
            pairs = [[question, chunk["content"]] for chunk in candidate_chunks]
            
            # Get reranker scores
            rerank_scores = self.reranker.predict(
                pairs,
                batch_size=self.reranker_batch_size,
            )
            
            # Sort chunks by reranker scores (descending)
            ranked_chunks = sorted(
                zip(candidate_chunks, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Return top-k after reranking
            final_chunks = [chunk for chunk, score in ranked_chunks[:limit]]
        else:
            final_chunks = candidate_chunks[:limit]
        
        return final_chunks
    
    def build_prompt(self, question: str, chunks: List[Dict]) -> str:
        """
        Build prompt cho LLM với context từ retrieved chunks
        Dùng content (có SAC Tier 1 + Tier 2) thay vì original_content
        """
        # Build context từ chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            
            # Extract document info
            doc_id = metadata.get('document_id', 'N/A')
            doc_title = metadata.get('title', 'N/A')
            
            # Extract location (Chương, Điều, Khoản)
            location_parts = []
            if metadata.get('Chuong'):
                match = re.search(r'Chương\s+([IVXLCDM]+)', metadata['Chuong'])
                if match:
                    location_parts.append(f"Chương {match.group(1)}")
            
            if metadata.get('Dieu'):
                match = re.search(r'Điều\s+(\d+)', metadata['Dieu'])
                if match:
                    location_parts.append(f"Điều {match.group(1)}")
            
            if metadata.get('Khoan'):
                match = re.search(r'^(\d+|[a-z])\)', metadata['Khoan'])
                if match:
                    location_parts.append(f"Khoản {match.group(1)}")
            
            location_str = " > ".join(location_parts) if location_parts else "Không xác định"
            full_citation = f"{doc_title} ({doc_id}) - {location_str}"
            amendment_text = ""
            if metadata.get("amended_by_document_id"):
                amendment_text = (
                    "\nNguồn tạo version hiện hành:\n"
                    f"- Văn bản sửa: {metadata.get('amended_by_document_id')}\n"
                    f"- Vị trí sửa: {metadata.get('amended_by_source_location') or 'Không rõ'}\n"
                    f"- Loại thay đổi: {metadata.get('change_type') or 'Không rõ'}"
                )
            
            # Dùng content (đã có SAC Tier 1 + Tier 2 + original)
            context_parts.append(
                f"[Đoạn {i}] {full_citation}\n{chunk['content']}"
                f"{amendment_text}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Build prompt - tự nhiên, gần gũi, BẮT BUỘC trích dẫn chính xác
        prompt = f"""Bạn là trợ lý AI thân thiện, giúp người dùng hiểu về luật pháp Việt Nam.

Hãy trả lời câu hỏi của người dùng một cách TỰ NHIÊN, Dễ HIỂU như đang trò chuyện.

QUY TẮC BẮT BUỘC:
1. Trả lời ngắn gọn, đi thẳng vào vấn đề (không dài dòng)
2. Dùng ngôn ngữ thân thiện, gần gũi (như nói chuyện với bạn bè)
3. **QUAN TRỌNG**: Khi trích dẫn, phải nói CHÍNH XÁC:
   - Văn bản nào (tên + mã số)
   - Chương mấy, Điều mấy, Khoản mấy (nếu có)
   - Ví dụ: "Theo Điều 3 Khoản 10 của Luật Thuế thu nhập cá nhân (112-VBHN-VPQH)..."
4. Nếu có số liệu cụ thể, nêu rõ ràng
5. Nếu không có thông tin, nói thẳng "Mình không tìm thấy thông tin này trong tài liệu"
6. Nếu đoạn có "Quan hệ sửa đổi đang áp dụng", phải dùng nội dung mới cho đúng Điều/Khoản và nêu văn bản sửa; không dùng lại phần cũ đã bị sửa, thay thế hoặc bãi bỏ

KHÔNG ĐƯỢC:
- Viết dạng bullet points (*, -)
- Dùng markdown (**, ##, *)
- Viết dài dòng, văn vẻ
- Nói chung chung mà không trích dẫn cụ thể

VÍ DỤ TỐT: 
"Có nhé, theo Khoản 10 Điều 3 của Luật Thuế thu nhập cá nhân (112-VBHN-VPQH) thì thu nhập từ chuyển nhượng tài sản số thuộc các khoản thu nhập khác phải chịu thuế."

VÍ DỤ XẤU: 
"Có, thu nhập từ tài sản số phải chịu thuế theo quy định."

CÁC ĐOẠN VĂN BẢN LIÊN QUAN:
{context}

CÂU HỎI: {question}

TRẢ LỜI (tự nhiên, gần gũi, PHẢI trích dẫn chính xác văn bản/điều/khoản):"""
        
        return prompt
    
    def generate_answer(self, question: str, top_k: int = 5) -> Dict:
        """
        RAG pipeline: Hybrid Retrieve + Generate
        
        Returns: Dict với answer, chunks, metadata, search_method
        """
        print(f"📝 Câu hỏi: {question}\n")
        
        # Step 1: Hybrid Retrieve
        search_method = "Hybrid (Vector + BM25)" if self.use_hybrid else "Vector only"
        print(f"🔍 Đang tìm kiếm với {search_method}...")
        chunks = self.retrieve_chunks(question, limit=top_k)
        
        if not chunks:
            return {
                "question": question,
                "answer": "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
                "chunks": [],
                "sources": [],
                "search_method": search_method
            }
        
        if chunks[0].get('score', 0) > 0:
            print(f"✓ Tìm thấy {len(chunks)} chunks (score: {chunks[0]['score']:.4f} - {chunks[-1].get('score', 0):.4f})\n")
        else:
            print(f"✓ Tìm thấy {len(chunks)} chunks (ranked by RRF)\n")
        
        # Step 2: Build prompt
        prompt = self.build_prompt(question, chunks)
        
        # Step 3: Generate answer
        print(f"🤖 Đang tạo câu trả lời với {GEMINI_MODEL}...")
        try:
            response = self.llm.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            print(f"Lỗi khi tạo câu trả lời: {e}")
            answer = "Xin lỗi, đã xảy ra lỗi khi tạo câu trả lời."
        
        # Collect sources với location đầy đủ
        sources = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            
            # Extract full location (Chương, Điều, Khoản)
            location_parts = []
            if metadata.get('Chuong'):
                match = re.search(r'Chương\s+([IVXLCDM]+)', metadata['Chuong'])
                if match:
                    location_parts.append(f"Chương {match.group(1)}")
            
            if metadata.get('Dieu'):
                match = re.search(r'Điều\s+(\d+)', metadata['Dieu'])
                if match:
                    location_parts.append(f"Điều {match.group(1)}")
            
            if metadata.get('Khoan'):
                match = re.search(r'^(\d+|[a-z])\)', metadata['Khoan'])
                if match:
                    location_parts.append(f"Khoản {match.group(1)}")
            
            location_str = " > ".join(location_parts) if location_parts else "Không xác định"
            
            source = {
                "document": metadata.get("title", "N/A"),
                "document_id": metadata.get("document_id", "N/A"),
                "location": location_str,
                "score": chunk.get("score", 0.0)
            }
            sources.append(source)
        
        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
            "sources": sources,
            "search_method": search_method
        }
    
    def print_result(self, result: Dict):
        """In kết quả đẹp với trích dẫn đầy đủ"""
        print(f"\n{'='*80}")
        print(f"❓ CÂU HỎI: {result['question']}")
        print(f"🔍 PHƯƠNG PHÁP: {result.get('search_method', 'N/A')}")
        print(f"{'='*80}")
        print(f"\n💡 TRẢ LỜI:\n{result['answer']}")
        print(f"\n{'='*80}")
        print(f"📚 NGUỒN THAM KHẢO:")
        print(f"{'='*80}")
        
        for i, source in enumerate(result['sources'], 1):
            citation = f"{source['document']} ({source['document_id']})"
            print(f"\n[{i}] {citation}")
            print(f"    Vị trí: {source['location']}")
            if source['score'] > 0:
                print(f"    Score: {source['score']:.4f}")
            else:
                print(f"    Rank: {i} (RRF)")
        
        print(f"\n{'='*80}\n")
    
    def interactive_mode(self):
        """Chế độ hỏi đáp tương tác"""
        print(f"\n{'='*80}")
        print(f"🤖 RAG Question Answering - Interactive Mode")
        print(f"{'='*80}")
        print(f"Nhập câu hỏi hoặc 'exit' để thoát\n")
        
        while True:
            try:
                question = input("❓ Câu hỏi: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['exit', 'quit', 'thoat']:
                    print("\n👋 Tạm biệt!")
                    break
                
                # Generate answer
                result = self.generate_answer(question, top_k=5)
                self.print_result(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Lỗi: {e}\n")


if __name__ == "__main__":
    # Initialize RAG system
    rag = RAGQuestionAnswering()
    
    # Interactive mode
    rag.interactive_mode()

