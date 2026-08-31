"""Step 3 versioned: provision-first chunking and temporal legal updates.

It reuses the internal embedding and SAC Tier 2 utilities while storing legal
changes as immutable provision versions.
"""

import argparse
from copy import deepcopy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

from qdrant_client.models import PointStruct

from config import OUTPUT_DIR
from embedding_core import EmbeddingQdrantProcessor
from step1_markdown_hierarchy import MarkdownHierarchyConverter


class VersionedEmbeddingQdrantProcessor(EmbeddingQdrantProcessor):
    """Store each Điều/Khoản as an immutable temporal version."""

    def __init__(self):
        super().__init__()
        self.markdown_converter = MarkdownHierarchyConverter(
            enable_llm=False,
            manifest_path=None,
        )
        self.tier2_request_interval_seconds = 10.0
        self._last_tier2_request_at: Optional[float] = None
        self.qdrant_upsert_batch_size = 64
        self.qdrant_upsert_max_retries = 3

    def _wait_for_tier2_request_slot(self) -> None:
        """Keep SAC Tier 2 requests spaced across the entire Step 3 run."""
        if self._last_tier2_request_at is not None:
            elapsed = time.monotonic() - self._last_tier2_request_at
            remaining = self.tier2_request_interval_seconds - elapsed
            if remaining > 0:
                print(f"    Nghỉ {remaining:.1f}s để tránh rate limit...")
                time.sleep(remaining)
        self._last_tier2_request_at = time.monotonic()

    def _upsert_points_batched(self, points: List[PointStruct]) -> None:
        """Upsert deterministic points in retryable batches."""
        if not points:
            return

        total_batches = (
            len(points) + self.qdrant_upsert_batch_size - 1
        ) // self.qdrant_upsert_batch_size
        for batch_index, start in enumerate(
            range(0, len(points), self.qdrant_upsert_batch_size),
            start=1,
        ):
            batch = points[start:start + self.qdrant_upsert_batch_size]
            for attempt in range(1, self.qdrant_upsert_max_retries + 1):
                try:
                    print(
                        f"    Qdrant upsert batch {batch_index}/{total_batches} "
                        f"({len(batch)} points)..."
                    )
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch,
                        wait=True,
                    )
                    break
                except Exception:
                    if attempt >= self.qdrant_upsert_max_retries:
                        raise
                    delay = 2 ** attempt
                    print(
                        f"    Qdrant timeout/lỗi, thử lại batch "
                        f"{batch_index} sau {delay}s..."
                    )
                    time.sleep(delay)

    # ------------------------------------------------------------------
    # Canonical legal identities
    # ------------------------------------------------------------------
    def extract_location_label(self, hierarchical_metadata: dict) -> str:
        """Preserve alphanumeric Điều/Khoản identifiers such as Khoản 14a."""
        parts = []
        chuong = str(hierarchical_metadata.get("Chuong") or "")
        appendix = re.search(
            r"Phụ\s*lục\s+([IVXLCDM\d]+)",
            chuong,
            re.IGNORECASE,
        )
        chapter = re.search(r"Chương\s+([IVXLCDM\d]+)", chuong, re.IGNORECASE)
        if appendix:
            parts.append(f"Phụ lục {appendix.group(1).upper()}")
        elif chapter:
            parts.append(f"Chương {chapter.group(1).upper()}")

        dieu = str(hierarchical_metadata.get("Dieu") or "")
        section = re.search(r"Mục\s+([IVXLCDM\d]+)", dieu, re.IGNORECASE)
        article = re.search(r"Điều\s+(\d+[a-zđ]?)", dieu, re.IGNORECASE)
        if section:
            parts.append(f"Mục {section.group(1).upper()}")
        elif article:
            parts.append(f"Điều {article.group(1).lower()}")

        khoan = str(hierarchical_metadata.get("Khoan") or "")
        clause = re.search(r"Khoản\s+(\d+[a-zđ]?)", khoan, re.IGNORECASE)
        if clause:
            parts.append(f"Khoản {clause.group(1).lower()}")
        return " > ".join(parts) if parts else "Không xác định"

    def _location_tokens(self, location: str) -> Dict[str, str]:
        text = re.sub(r"\s+", " ", str(location or "")).strip()
        patterns = {
            "chuong": r"Chương\s+([IVXLCDM\d]+)",
            "dieu": r"Điều\s+(\d+[a-zđ]?)",
            "khoan": r"Khoản\s+(\d+[a-zđ]?)",
            "diem": r"Điểm\s+([a-zđ](?:\d+|(?:\.\d+)+)?)\b",
            "phu_luc": r"Phụ\s*lục\s+([IVXLCDM\d]+)",
            "muc": r"Mục\s+([IVXLCDM\d]+)",
        }
        tokens = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tokens[name] = match.group(1).upper()
        return tokens

    def canonical_location(self, location: str, include_point: bool = True) -> str:
        tokens = self._location_tokens(location)
        parts = []
        if "phu_luc" in tokens:
            parts.append(f"PHU_LUC_{tokens['phu_luc']}")
        if "muc" in tokens:
            parts.append(f"MUC_{tokens['muc']}")
        if "dieu" in tokens:
            parts.append(f"DIEU_{tokens['dieu']}")
        elif "chuong" in tokens and "phu_luc" not in tokens:
            # Chương is independently stored, while Điều/Khoản identities stay
            # stable even if their containing Chương changes later.
            parts.append(f"CHUONG_{tokens['chuong']}")
        if "khoan" in tokens:
            parts.append(f"KHOAN_{tokens['khoan']}")
        if include_point and "diem" in tokens:
            parts.append(f"DIEM_{tokens['diem']}")
        return "|".join(parts)

    def provision_key(self, document_id: str, location: str) -> str:
        canonical = self.canonical_location(location, include_point=False)
        if not canonical:
            raise ValueError(f"Không tạo được provision_key từ {location!r}")
        return f"{document_id}|{canonical}"

    def _version_id(self, provision_key: str, version: int) -> str:
        return f"{provision_key}|V{version}"

    def _point_id(self, version_id: str, sub_chunk_index: Optional[int]) -> int:
        index = 0 if sub_chunk_index is None else sub_chunk_index
        raw = f"{version_id}|SUB_{index}"
        return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)

    def _same_article(self, location: str, target_location: str) -> bool:
        left = self._location_tokens(location)
        right = self._location_tokens(target_location)
        return bool(left.get("dieu") and left.get("dieu") == right.get("dieu"))

    def _same_provision(self, location: str, target_location: str) -> bool:
        return self.canonical_location(location, False) == self.canonical_location(
            target_location,
            False,
        )

    # ------------------------------------------------------------------
    # Provision-first chunking
    # ------------------------------------------------------------------
    def build_provisions(self, markdown_text: str) -> List[Dict]:
        """Return every Chương/Điều/Khoản with only its direct Markdown body."""
        provisions = []
        lines = markdown_text.splitlines()
        headings = []
        hierarchy: Dict[str, str] = {}

        for index, line in enumerate(lines):
            match = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
            if not match:
                continue

            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                hierarchy = {"Chuong": title}
            elif level == 2:
                hierarchy = {
                    **({"Chuong": hierarchy["Chuong"]} if hierarchy.get("Chuong") else {}),
                    "Dieu": title,
                }
            else:
                hierarchy = {
                    **({"Chuong": hierarchy["Chuong"]} if hierarchy.get("Chuong") else {}),
                    **({"Dieu": hierarchy["Dieu"]} if hierarchy.get("Dieu") else {}),
                    "Khoan": title,
                }

            headings.append({
                "start": index,
                "hierarchical_metadata": deepcopy(hierarchy),
            })

        for position, heading in enumerate(headings):
            end = (
                headings[position + 1]["start"]
                if position + 1 < len(headings)
                else len(lines)
            )
            content = "\n".join(lines[heading["start"]:end]).strip()
            hierarchy_metadata = heading["hierarchical_metadata"]
            location = self.extract_location_label(hierarchy_metadata)
            if location == "Không xác định":
                continue
            if not self.canonical_location(location, include_point=False):
                continue
            provisions.append({
                "location_label": location,
                "hierarchical_metadata": hierarchy_metadata,
                "provision_content": content,
            })
        return provisions

    def split_provision(self, provision: Dict) -> List[Dict]:
        content = provision["provision_content"]
        if len(content) <= self.chunk_size:
            pieces = [content]
        else:
            pieces = self.recursive_splitter.split_text(content)
        return [
            {
                "original_content": piece,
                "provision_content": content,
                "hierarchical_metadata": deepcopy(
                    provision.get("hierarchical_metadata") or {}
                ),
                "location_label": provision["location_label"],
                "sub_chunk_index": index if len(pieces) > 1 else None,
                "chunk_size": len(piece),
            }
            for index, piece in enumerate(pieces)
        ]

    def _format_tier2_text(
        self,
        document_id: str,
        location: str,
        tier2: Dict,
    ) -> str:
        return (
            f"[{document_id} > {location}]\n"
            f"ĐỐI TƯỢNG: {tier2.get('doi_tuong', '')}\n"
            f"NỘI DUNG: {tier2.get('noi_dung', '')}\n"
            f"SỐ LIỆU/QUY ĐỊNH: {tier2.get('so_lieu_quy_dinh', '')}"
        )

    def _base_version_metadata(
        self,
        document_metadata: Dict,
        provision: Dict,
        version: int = 1,
        valid_from: Optional[str] = None,
    ) -> Dict:
        document_id = document_metadata["document_id"]
        location = provision["location_label"]
        key = self.provision_key(document_id, location)
        version_id = self._version_id(key, version)
        start = valid_from or document_metadata.get("effective_date")
        return {
            **deepcopy(provision.get("hierarchical_metadata") or {}),
            "location_label": location,
            "document_id": document_id,
            "title": document_metadata.get("title", ""),
            "document_type": document_metadata.get("document_type"),
            "legal_rank": document_metadata.get("legal_rank"),
            "issued_date": document_metadata.get("issued_date"),
            "effective_date": document_metadata.get("effective_date"),
            "expiration_date": document_metadata.get("expiration_date"),
            "status": document_metadata.get("status", "active"),
            "affects_documents": document_metadata.get("affects_documents") or [],
            "affected_by_documents": document_metadata.get("affected_by_documents") or [],
            "source_file": document_metadata.get("source_file", ""),
            "provision_key": key,
            "version_id": version_id,
            "version": version,
            "valid_from": start,
            "valid_to": None,
            "is_latest": True,
            "previous_version_id": None,
            "superseded_by_version_id": None,
            "amended_by_document_id": None,
            "amended_by_source_location": None,
            "change_type": None,
            "applied_operation_ids": [],
        }

    def _build_points_for_provision(
        self,
        provision: Dict,
        version_metadata: Dict,
        sac_tier1: str,
    ) -> List[PointStruct]:
        return self._build_points_for_provisions([
            (provision, version_metadata, sac_tier1),
        ])

    def _build_points_for_provisions(
        self,
        requests: List[Tuple[Dict, Dict, str]],
    ) -> List[PointStruct]:
        """Generate SAC/embeddings in global batches across many provisions."""
        work_items = []
        for provision, version_metadata, sac_tier1 in requests:
            for chunk in self.split_provision(provision):
                work_items.append({
                    "chunk": chunk,
                    "version_metadata": deepcopy(version_metadata),
                    "sac_tier1": sac_tier1,
                })

        if not work_items:
            return []

        tier2_summaries = []
        total_batches = (
            len(work_items) + self.tier2_batch_size - 1
        ) // self.tier2_batch_size
        for index in range(0, len(work_items), self.tier2_batch_size):
            batch_number = index // self.tier2_batch_size + 1
            batch_items = work_items[index:index + self.tier2_batch_size]
            print(
                f"    SAC Tier 2 batch {batch_number}/{total_batches} "
                f"({len(batch_items)} subchunks)"
            )
            self._wait_for_tier2_request_slot()
            tier2_summaries.extend(
                self.generate_tier2_summaries_batch(
                    [item["chunk"] for item in batch_items]
                )
            )

        if len(tier2_summaries) != len(work_items):
            raise ValueError(
                "Số SAC Tier 2 không khớp số subchunk: "
                f"{len(tier2_summaries)} != {len(work_items)}"
            )

        augmented = []
        for item, tier2 in zip(work_items, tier2_summaries):
            chunk = item["chunk"]
            version_metadata = item["version_metadata"]
            sac_tier1 = item["sac_tier1"]
            tier2_text = self._format_tier2_text(
                version_metadata["document_id"],
                chunk["location_label"],
                tier2,
            )
            content = (
                f"{sac_tier1}\n\n{tier2_text}\n\n"
                f"{chunk['original_content']}"
            )
            augmented.append((
                chunk,
                version_metadata,
                sac_tier1,
                tier2,
                tier2_text,
                content,
            ))

        vectors = self.batch_embed([item[5] for item in augmented])
        points = []
        for item, vector in zip(augmented, vectors):
            chunk, version_metadata, sac_tier1, tier2, tier2_text, content = item
            metadata = deepcopy(version_metadata)
            metadata.update(deepcopy(chunk.get("hierarchical_metadata") or {}))
            metadata["location_label"] = chunk["location_label"]
            metadata["sub_chunk_index"] = chunk["sub_chunk_index"]
            metadata["chunk_size"] = chunk["chunk_size"]
            metadata["augmented_size"] = len(content)
            point_id = self._point_id(
                metadata["version_id"],
                chunk["sub_chunk_index"],
            )
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "content": content,
                    "original_content": chunk["original_content"],
                    "provision_content": chunk["provision_content"],
                    "sac_tier1": sac_tier1,
                    "sac_tier2": tier2,
                    "tier2_text": tier2_text,
                    "metadata": metadata,
                },
            ))
        return points

    def ingest_base_document(
        self,
        markdown_file: str,
        summary_file: str,
        metadata_file: str,
    ) -> int:
        with open(markdown_file, "r", encoding="utf-8") as file:
            markdown = file.read()
        with open(summary_file, "r", encoding="utf-8") as file:
            sac_tier1 = json.load(file).get("document_summary", "")
        with open(metadata_file, "r", encoding="utf-8") as file:
            document_metadata = json.load(file)

        document_id = document_metadata.get("document_id")
        if not document_id:
            raise ValueError(f"{metadata_file} thiếu document_id")

        provisions = self.build_provisions(markdown)
        existing_ids = {
            point.id for point in self._scroll_document_points(document_id)
        }
        expected_ids_by_location = {}
        for provision in provisions:
            metadata = self._base_version_metadata(document_metadata, provision)
            expected_ids_by_location[provision["location_label"]] = {
                self._point_id(metadata["version_id"], chunk["sub_chunk_index"])
                for chunk in self.split_provision(provision)
            }

        expected_ids = set().union(*expected_ids_by_location.values())
        if expected_ids and expected_ids.issubset(existing_ids):
            print(f"  Bỏ qua ingest nền đã tồn tại đầy đủ: {document_id}")
            return 0

        missing_locations = {
            location
            for location, point_ids in expected_ids_by_location.items()
            if not point_ids.issubset(existing_ids)
        }
        if existing_ids:
            print(
                f"  Tiếp tục ingest nền chưa đầy đủ: {document_id} "
                f"({len(missing_locations)}/{len(provisions)} provisions thiếu)"
            )

        requests = []
        for provision in provisions:
            if provision["location_label"] not in missing_locations:
                continue
            metadata = self._base_version_metadata(document_metadata, provision)
            requests.append((provision, metadata, sac_tier1))
        points = self._build_points_for_provisions(requests)
        self._upsert_points_batched(points)
        print(
            f"  Đã nạp {document_id}: "
            f"{len(requests)}/{len(provisions)} provisions, {len(points)} points"
        )
        return len(points)

    # ------------------------------------------------------------------
    # Reading and closing versions
    # ------------------------------------------------------------------
    def _latest_groups(self, document_id: str) -> Dict[str, List]:
        groups: Dict[str, List] = {}
        for point in self._scroll_document_points(document_id):
            metadata = (point.payload or {}).get("metadata") or {}
            if not metadata.get("is_latest", True):
                continue
            if metadata.get("valid_to") is not None:
                continue
            key = metadata.get("provision_key")
            if key:
                groups.setdefault(key, []).append(point)
        return groups

    def _group_metadata(self, points: List) -> Dict:
        return deepcopy(((points[0].payload or {}).get("metadata") or {}))

    def _group_content(self, points: List) -> str:
        content = (points[0].payload or {}).get("provision_content")
        if not content:
            raise ValueError("Point thiếu provision_content; hãy ingest lại bằng Step 3 versioned.")
        return str(content)

    def _group_sac_tier1(self, points: List) -> str:
        return str((points[0].payload or {}).get("sac_tier1") or "")

    def _close_group(
        self,
        points: List,
        valid_to: str,
        successor_version_id: Optional[str],
    ) -> None:
        for point in points:
            payload = point.payload or {}
            metadata = deepcopy(payload.get("metadata") or {})
            metadata["valid_to"] = valid_to
            metadata["is_latest"] = False
            metadata["superseded_by_version_id"] = successor_version_id
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload={"metadata": metadata},
                points=[point.id],
                wait=True,
            )

    def _operation_applied(self, groups: Dict[str, List], operation_id: str) -> bool:
        return any(
            operation_id in (self._group_metadata(points).get("applied_operation_ids") or [])
            for points in groups.values()
        )

    def _successor_metadata(
        self,
        old_metadata: Dict,
        operation: Dict,
        location: Optional[str] = None,
        initial_version: bool = False,
    ) -> Dict:
        metadata = deepcopy(old_metadata)
        old_version = int(metadata.get("version") or 1)
        version = 1 if initial_version else old_version + 1
        location = location or metadata["location_label"]
        key = self.provision_key(metadata["document_id"], location)
        version_id = self._version_id(key, version)
        old_start = metadata.get("valid_from") or metadata.get("effective_date")
        operation_date = operation.get("effective_date") or old_start
        valid_from = max(value for value in [old_start, operation_date] if value)
        inherited_operation_ids = [] if initial_version else (
            metadata.get("applied_operation_ids") or []
        )
        return {
            **metadata,
            "location_label": location,
            "provision_key": key,
            "version_id": version_id,
            "version": version,
            "valid_from": valid_from,
            "valid_to": None,
            "is_latest": True,
            "previous_version_id": None if initial_version else old_metadata.get("version_id"),
            "superseded_by_version_id": None,
            "amended_by_document_id": operation.get("van_ban_sua"),
            "amended_by_source_location": operation.get("dieu_khoan_sua"),
            "change_type": operation.get("loai_thay_doi"),
            "applied_operation_ids": list(dict.fromkeys(
                inherited_operation_ids
                + [operation["operation_id"]]
            )),
        }

    def _provision_from_content(self, content: str, metadata: Dict) -> Dict:
        return {
            "location_label": metadata["location_label"],
            "hierarchical_metadata": {
                key: metadata[key]
                for key in ("Chuong", "Dieu", "Khoan")
                if metadata.get(key)
            },
            "provision_content": content.strip(),
        }

    def _align_hierarchy(self, metadata: Dict, location: str, content: str) -> Dict:
        """Align inherited hierarchy when an article gains or loses clauses."""
        aligned = deepcopy(metadata)
        tokens = self._location_tokens(location)
        first_line = content.strip().splitlines()[0] if content.strip() else ""
        if "dieu" in tokens:
            if first_line.startswith("## "):
                aligned["Dieu"] = first_line[3:].strip()
            elif not re.search(
                rf"Điều\s+{re.escape(tokens['dieu'])}\b",
                str(aligned.get("Dieu") or ""),
                re.IGNORECASE,
            ):
                aligned["Dieu"] = f"Điều {tokens['dieu'].lower()}"
        if "khoan" in tokens:
            if first_line.startswith("### "):
                aligned["Khoan"] = first_line[4:].strip()
            else:
                aligned["Khoan"] = f"Khoản {tokens['khoan'].lower()}"
        else:
            aligned.pop("Khoan", None)
        return aligned

    def _insert_successor(
        self,
        old_points: Optional[List],
        content: str,
        operation: Dict,
        template_metadata: Dict,
        sac_tier1: str,
        location: Optional[str] = None,
        initial_version: bool = False,
    ) -> List[PointStruct]:
        return self._insert_successors_batch([{
            "old_points": old_points,
            "content": content,
            "operation": operation,
            "template_metadata": template_metadata,
            "sac_tier1": sac_tier1,
            "location": location,
            "initial_version": initial_version,
        }])

    def _insert_successors_batch(self, plans: List[Dict]) -> List[PointStruct]:
        """Build many successor provisions in shared SAC batches of ten."""
        prepared = []
        requests = []
        for plan in plans:
            metadata = self._successor_metadata(
                plan["template_metadata"],
                plan["operation"],
                location=plan.get("location"),
                initial_version=plan.get("initial_version", False),
            )
            metadata = self._align_hierarchy(
                metadata,
                metadata["location_label"],
                plan["content"],
            )
            provision = self._provision_from_content(plan["content"], metadata)
            requests.append((provision, metadata, plan["sac_tier1"]))
            prepared.append((plan, metadata))

        points = self._build_points_for_provisions(requests)
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        for plan, metadata in prepared:
            if plan.get("old_points"):
                self._close_group(
                    plan["old_points"],
                    metadata["valid_from"],
                    metadata["version_id"],
                )
        return points

    # ------------------------------------------------------------------
    # Deterministic reconstruction
    # ------------------------------------------------------------------
    def _strip_outer_quotes(self, text: str) -> str:
        value = str(text or "").strip()
        if value.startswith("“") and value.endswith("”."):
            return value[1:-2].strip()
        if value.startswith("“") and value.endswith("”"):
            return value[1:-1].strip()
        if value.startswith('"') and value.endswith('".'):
            return value[1:-2].strip()
        if value.startswith('"') and value.endswith('"'):
            return value[1:-1].strip()
        return value

    def _replacement_for_clause(self, operation: Dict, old_content: str) -> str:
        replacement = self._strip_outer_quotes(operation.get("replacement_text"))
        if not replacement:
            raise ValueError("Operation sửa Khoản thiếu replacement_text")
        if replacement.startswith("### "):
            return replacement
        target = self._location_tokens(operation["target_location"])
        number = target.get("khoan")
        if not number:
            raise ValueError("Target Khoản không có số Khoản")
        lines = replacement.splitlines()
        first = lines[0].strip()
        first = re.sub(
            rf"^(?:Khoản\s+)?{re.escape(number)}\s*[\.:]\s*",
            "",
            first,
            flags=re.IGNORECASE,
        )
        lines[0] = f"### Khoản {number.lower()}. {first}".rstrip()
        return "\n".join(lines).strip()

    def _replacement_article_provisions(self, operation: Dict) -> List[Dict]:
        replacement = self._strip_outer_quotes(operation.get("replacement_text"))
        if not replacement:
            raise ValueError("Operation sửa Điều thiếu replacement_text")
        if not re.search(r"^(?:##\s+)?Điều\s+\d+", replacement, re.MULTILINE | re.IGNORECASE):
            target = self._location_tokens(operation["target_location"])
            replacement = f"Điều {target['dieu']}.\n{replacement}"
        markdown = self.markdown_converter.convert_to_markdown(replacement)
        provisions = self.build_provisions(markdown)
        if not provisions:
            raise ValueError("Không phân tích được nội dung Điều thay thế thành provision")
        return provisions

    def _is_article_title_operation(self, operation: Dict) -> bool:
        raw_component = operation.get("target_component")
        component = str(raw_component or "").strip().casefold()
        if component in {"title", "tieu_de", "tiêu_đề", "ten", "tên"}:
            return True
        # Khi Step 1 đã gắn component, giá trị đó là tín hiệu quyết định. Chỉ
        # metadata cũ chưa có field mới được suy luận từ câu chữ nguồn.
        if raw_component is not None and component:
            return False
        evidence = " ".join(
            str(operation.get(key) or "")
            for key in (
                "can_cu_trich_doan",
                "evidence",
                "amendment_source_section",
            )
        )
        return bool(re.search(
            r"sửa\s+đổi\s+(?:tên|tiêu\s+đề)\s+Điều\b",
            evidence,
            re.IGNORECASE,
        ))

    def _replacement_article_heading(self, operation: Dict) -> str:
        target_number = self._location_tokens(
            operation.get("target_location") or ""
        ).get("dieu")
        if not target_number:
            raise ValueError("Operation đổi tên Điều thiếu số Điều đích")

        sources = [
            operation.get("replacement_text"),
            operation.get("amendment_source_section"),
        ]
        pattern = re.compile(
            rf"^Điều\s+{re.escape(target_number)}\s*[\.:]?\s*.+$",
            re.IGNORECASE,
        )
        for source in sources:
            for line in str(source or "").splitlines():
                candidate = re.sub(r"^#{1,3}\s+", "", line.strip())
                candidate = candidate.strip("“”\" ")
                if pattern.match(candidate):
                    return f"## {candidate}"
        raise ValueError(
            f"Không tìm thấy tiêu đề mới của Điều {target_number} trong amendment"
        )

    def _replace_article_heading(self, old_content: str, new_heading: str) -> str:
        lines = old_content.splitlines()
        for index, line in enumerate(lines):
            if re.match(r"^##\s+Điều\s+", line.strip(), re.IGNORECASE):
                lines[index] = new_heading
                return "\n".join(lines).strip()
        return f"{new_heading}\n{old_content}".strip()

    def _replace_article_preamble(
        self,
        old_content: str,
        new_preamble: str,
    ) -> str:
        """Replace only the direct Điều body while preserving its heading."""
        lines = old_content.splitlines()
        if not lines or not re.match(
            r"^##\s+Điều\s+",
            lines[0].strip(),
            re.IGNORECASE,
        ):
            raise ValueError("Provision Điều không có heading hợp lệ để sửa đoạn đầu")
        preamble = self._strip_outer_quotes(new_preamble)
        if not preamble:
            raise ValueError("Operation sửa đoạn đầu Điều thiếu replacement_text")
        return f"{lines[0].strip()}\n{preamble}".strip()

    def _point_spans(self, content: str) -> Dict[str, Tuple[int, int]]:
        matches = list(re.finditer(
            # Older legal documents use both "a)" and "a." for point labels.
            r"(?m)^\s*([a-zđ](?:\d+|(?:\.\d+)+)?)[\.)]\s+",
            content,
            re.IGNORECASE,
        ))
        spans = {}
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            spans[match.group(1).lower()] = (match.start(), end)
        return spans

    def _extract_point_block(self, text: str, label: str) -> str:
        spans = self._point_spans(text)
        if label not in spans:
            raise ValueError(f"Nội dung sửa đổi không chứa Điểm {label}")
        start, end = spans[label]
        return text[start:end].strip()

    def _point_order_key(self, label: str) -> Tuple[int, Tuple[int, ...], str]:
        """Sort Vietnamese legal point labels in their conventional order."""
        order = "a b c d đ e g h i k l m n o p q r s t u v x y"
        labels = order.split()
        normalized = str(label or "").strip().lower()
        match = re.fullmatch(
            r"([a-zđ])((?:\d+|(?:\.\d+)+)?)",
            normalized,
            re.IGNORECASE,
        )
        if not match:
            return len(labels), (), normalized
        base_label, numeric_suffix = match.groups()
        try:
            base_order = labels.index(base_label)
        except ValueError:
            base_order = len(labels)
        suffix_order = tuple(
            int(part)
            for part in numeric_suffix.lstrip(".").split(".")
            if part
        )
        return base_order, suffix_order, normalized

    def _reconstruct_point(self, old_content: str, operation: Dict) -> str:
        label = self._location_tokens(operation["target_location"]).get("diem", "").lower()
        if not label:
            raise ValueError("Operation cấp Điểm thiếu nhãn Điểm")
        spans = self._point_spans(old_content)
        change_type = operation["loai_thay_doi"]

        if change_type == "bo_sung":
            if label in spans:
                raise ValueError(f"Không thể bổ sung: Điểm {label} đã tồn tại")
            new_block = self._extract_point_block(
                operation.get("replacement_text") or "",
                label,
            )
            label_order = self._point_order_key(label)
            greater = sorted(
                (
                    key for key in spans
                    if self._point_order_key(key) > label_order
                ),
                key=self._point_order_key,
            )
            insert_at = spans[greater[0]][0] if greater else len(old_content)
            return (
                old_content[:insert_at].rstrip()
                + "\n\n"
                + new_block
                + "\n\n"
                + old_content[insert_at:].lstrip()
            ).strip()

        if label not in spans:
            raise ValueError(f"Không tìm thấy Điểm {label} trong provision hiện hành")
        start, end = spans[label]
        if change_type == "bai_bo":
            new_block = (
                f"{label}) [Đã bãi bỏ theo {operation.get('dieu_khoan_sua')} "
                f"{operation.get('van_ban_sua')}, hiệu lực từ ngày "
                f"{operation.get('effective_date')}]."
            )
        else:
            new_block = self._extract_point_block(
                operation.get("replacement_text") or "",
                label,
            )
        return (old_content[:start] + new_block + "\n\n" + old_content[end:].lstrip()).strip()

    def _replace_phrase_in_point(self, old_content: str, operation: Dict) -> str:
        """Apply a phrase operation only inside the explicitly targeted point."""
        label = self._location_tokens(operation["target_location"]).get("diem", "").lower()
        if not label:
            raise ValueError("Operation sua_doi_cum_tu cap Diem thieu nhan Diem")

        spans = self._point_spans(old_content)
        if label not in spans:
            raise ValueError(f"Khong tim thay Diem {label} trong provision hien hanh")

        old_text = str(operation.get("noi_dung_cu") or "")
        new_text = str(operation.get("noi_dung_moi") or "")
        if not old_text:
            raise ValueError("Operation sua_doi_cum_tu thieu noi_dung_cu")

        start, end = spans[label]
        point_block = old_content[start:end]
        if old_text not in point_block:
            raise ValueError(
                f"Khong tim thay cum {old_text!r} trong Diem {label} duoc chi dinh"
            )

        updated_block = point_block.replace(old_text, new_text)
        return (old_content[:start] + updated_block + old_content[end:]).strip()

    def _grouped_point_targets(self, operation: Dict) -> Tuple[List[str], str]:
        """Return point targets and require one shared parent provision."""
        targets = operation.get("dieu_khoan_bi_sua") or [
            operation.get("target_location")
        ]
        if isinstance(targets, str):
            targets = [targets]
        targets = [str(target).strip() for target in targets if str(target).strip()]
        if not targets:
            raise ValueError("Operation cấp Điểm không có target")

        parents = []
        for target in targets:
            if "diem" not in self._location_tokens(target):
                raise ValueError(
                    "Operation cấp Điểm chứa target không phải Điểm: "
                    f"{target}"
                )
            parents.append(re.sub(
                r"\s*>\s*Điểm\s+[a-zđ](?:\d+|(?:\.\d+)+)?\s*$",
                "",
                target,
                flags=re.IGNORECASE,
            ).strip())

        canonical_parents = {
            self.canonical_location(parent, include_point=False)
            for parent in parents
        }
        if len(canonical_parents) != 1:
            raise ValueError(
                "Các Điểm trong một operation phải cùng Điều và Khoản: "
                f"{targets}"
            )
        return targets, parents[0]

    def _tombstone(self, location: str, operation: Dict) -> str:
        tokens = self._location_tokens(location)
        if "khoan" in tokens:
            heading = f"### Khoản {tokens['khoan'].lower()}."
        else:
            heading = f"## Điều {tokens.get('dieu', '')}."
        return (
            f"{heading} [Đã bãi bỏ theo {operation.get('dieu_khoan_sua')} "
            f"{operation.get('van_ban_sua')}, hiệu lực từ ngày "
            f"{operation.get('effective_date')}]."
        )

    # ------------------------------------------------------------------
    # Applying operations
    # ------------------------------------------------------------------
    def apply_operation(self, operation: Dict) -> int:
        required = (
            "operation_id",
            "van_ban_sua",
            "van_ban_bi_sua",
            "target_location",
            "target_level",
            "loai_thay_doi",
            "effective_date",
        )
        missing = [key for key in required if not operation.get(key)]
        if missing:
            raise ValueError(f"Operation thiếu trường: {', '.join(missing)}")

        document_id = operation["van_ban_bi_sua"]
        groups = self._latest_groups(document_id)
        if not groups:
            raise ValueError(f"Qdrant chưa có văn bản đích {document_id}")
        level = operation["target_level"]
        change_type = operation["loai_thay_doi"]
        target = operation["target_location"]
        raw_component = operation.get("target_component")
        component = (
            str(raw_component).strip().casefold()
            if raw_component is not None
            else None
        )
        if component not in {None, "", "title", "preamble", "all"}:
            raise ValueError(f"target_component không hợp lệ: {raw_component!r}")
        if change_type == "sua_doi_cum_tu":
            if component not in {None, ""}:
                raise ValueError("sua_doi_cum_tu phải có target_component=null")
        elif component == "title" and (level != "dieu" or change_type != "sua_doi"):
            raise ValueError(
                "target_component='title' chỉ hợp lệ với sua_doi cấp Điều"
            )
        elif component == "preamble" and (
            level != "dieu" or change_type != "sua_doi"
        ):
            raise ValueError(
                "target_component='preamble' chỉ hợp lệ với sua_doi cấp Điều"
            )

        if change_type == "sua_doi_cum_tu":
            if level == "diem":
                point_targets, parent_target = self._grouped_point_targets(operation)
                matches = [
                    points for points in groups.values()
                    if self._same_provision(
                        self._group_metadata(points)["location_label"],
                        parent_target,
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Target Điểm phải match đúng một Khoản: {target}"
                    )

                points = matches[0]
                if operation["operation_id"] in (
                    self._group_metadata(points).get("applied_operation_ids") or []
                ):
                    print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                    return 0

                new_content = self._group_content(points)
                for point_target in point_targets:
                    point_operation = {**operation, "target_location": point_target}
                    new_content = self._replace_phrase_in_point(
                        new_content,
                        point_operation,
                    )
                self._insert_successor(
                    points,
                    new_content,
                    operation,
                    self._group_metadata(points),
                    self._group_sac_tier1(points),
                )
                return 1

            candidates = [
                points for points in groups.values()
                if (
                    self._same_article(self._group_metadata(points)["location_label"], target)
                    if level == "dieu"
                    else self._same_provision(self._group_metadata(points)["location_label"], target)
                )
            ]
            old_text = str(operation.get("noi_dung_cu") or "")
            new_text = str(operation.get("noi_dung_moi") or "")
            candidates = [points for points in candidates if old_text in self._group_content(points)]
            if not candidates:
                if self._operation_applied(groups, operation["operation_id"]):
                    print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                    return 0
                raise ValueError(f"Không tìm thấy cụm {old_text!r} trong {document_id} {target}")
            plans = []
            for points in candidates:
                old_content = self._group_content(points)
                new_content = old_content.replace(old_text, new_text)
                metadata = self._group_metadata(points)
                plans.append({
                    "old_points": points,
                    "content": new_content,
                    "operation": operation,
                    "template_metadata": metadata,
                    "sac_tier1": self._group_sac_tier1(points),
                })
            self._insert_successors_batch(plans)
            return len(plans)

        if level == "diem":
            point_targets, parent_target = self._grouped_point_targets(operation)
            matches = [
                points for points in groups.values()
                if self._same_provision(self._group_metadata(points)["location_label"], parent_target)
            ]
            if len(matches) != 1:
                raise ValueError(f"Target Điểm phải match đúng một Khoản: {target}")
            points = matches[0]
            if operation["operation_id"] in (
                self._group_metadata(points).get("applied_operation_ids") or []
            ):
                print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                return 0
            new_content = self._group_content(points)
            for point_target in point_targets:
                point_operation = {**operation, "target_location": point_target}
                new_content = self._reconstruct_point(new_content, point_operation)
            self._insert_successor(
                points,
                new_content,
                operation,
                self._group_metadata(points),
                self._group_sac_tier1(points),
            )
            return 1

        matches = [
            points for points in groups.values()
            if self._same_provision(self._group_metadata(points)["location_label"], target)
        ]

        if level == "dieu" and self._is_article_title_operation(operation):
            if len(matches) != 1:
                raise ValueError(
                    "Đổi tên Điều cần đúng một provision cấp Điều; "
                    f"hãy ingest lại cấu trúc Chương/Điều/Khoản: {target}"
                )
            points = matches[0]
            if operation["operation_id"] in (
                self._group_metadata(points).get("applied_operation_ids") or []
            ):
                print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                return 0
            new_heading = self._replacement_article_heading(operation)
            new_content = self._replace_article_heading(
                self._group_content(points),
                new_heading,
            )
            self._insert_successor(
                points,
                new_content,
                operation,
                self._group_metadata(points),
                self._group_sac_tier1(points),
            )
            return 1

        if level == "dieu" and component == "preamble":
            if len(matches) != 1:
                raise ValueError(
                    "Sửa đoạn đầu Điều cần đúng một provision cấp Điều: "
                    f"{target}"
                )
            points = matches[0]
            if operation["operation_id"] in (
                self._group_metadata(points).get("applied_operation_ids") or []
            ):
                print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                return 0
            new_content = self._replace_article_preamble(
                self._group_content(points),
                operation.get("replacement_text") or "",
            )
            self._insert_successor(
                points,
                new_content,
                operation,
                self._group_metadata(points),
                self._group_sac_tier1(points),
            )
            return 1

        if level == "khoan":
            if change_type == "bo_sung" and not matches:
                article_groups = [
                    points for points in groups.values()
                    if self._same_article(self._group_metadata(points)["location_label"], target)
                ]
                if not article_groups:
                    raise ValueError(f"Không có Điều cha để bổ sung {target}")
                template_points = article_groups[0]
                template = self._group_metadata(template_points)
                content = self._replacement_for_clause(operation, "")
                self._insert_successor(
                    None,
                    content,
                    operation,
                    template,
                    self._group_sac_tier1(template_points),
                    location=target,
                    initial_version=True,
                )
                return 1
            if len(matches) != 1:
                raise ValueError(f"Target Khoản phải match đúng một provision: {target}")
            points = matches[0]
            if operation["operation_id"] in (
                self._group_metadata(points).get("applied_operation_ids") or []
            ):
                print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                return 0
            if change_type == "bo_sung":
                raise ValueError(f"Không thể bổ sung: provision {target} đã tồn tại")
            if change_type == "bai_bo":
                content = self._tombstone(target, operation)
            else:
                content = self._replacement_for_clause(
                    operation,
                    self._group_content(points),
                )
            self._insert_successor(
                points,
                content,
                operation,
                self._group_metadata(points),
                self._group_sac_tier1(points),
            )
            return 1

        if level == "dieu":
            article_groups = [
                points for points in groups.values()
                if self._same_article(self._group_metadata(points)["location_label"], target)
            ]
            if not article_groups:
                if change_type != "bo_sung":
                    raise ValueError(f"Không tìm thấy Điều đích: {target}")
                replacement_provisions = self._replacement_article_provisions(operation)
                template_points = next(iter(groups.values()))
                template = self._group_metadata(template_points)
                sac_tier1 = self._group_sac_tier1(template_points)
                plans = [
                    {
                        "old_points": None,
                        "content": provision["provision_content"],
                        "operation": operation,
                        "template_metadata": template,
                        "sac_tier1": sac_tier1,
                        "location": provision["location_label"],
                        "initial_version": True,
                    }
                    for provision in replacement_provisions
                ]
                self._insert_successors_batch(plans)
                return len(plans)
            if all(
                operation["operation_id"] in (
                    self._group_metadata(points).get("applied_operation_ids") or []
                )
                for points in article_groups
            ):
                print(f"  Bỏ qua operation đã áp dụng: {operation['operation_id'][:12]}")
                return 0
            if change_type == "bai_bo":
                template_points = next(
                    (
                        points for points in article_groups
                        if self._same_provision(
                            self._group_metadata(points)["location_label"],
                            target,
                        )
                    ),
                    article_groups[0],
                )
                template = self._group_metadata(template_points)
                tombstone = self._tombstone(target, operation)
                successor = self._insert_successor(
                    None,
                    tombstone,
                    operation,
                    template,
                    self._group_sac_tier1(template_points),
                    location=target,
                    initial_version=False,
                )
                successor_id = ((successor[0].payload or {}).get("metadata") or {}).get("version_id")
                for points in article_groups:
                    self._close_group(
                        points,
                        operation["effective_date"],
                        successor_id,
                    )
                return len(article_groups)

            replacement_provisions = self._replacement_article_provisions(operation)
            replacement_by_location = {
                self.canonical_location(item["location_label"], False): item
                for item in replacement_provisions
            }
            existing_by_location = {
                self.canonical_location(self._group_metadata(points)["location_label"], False): points
                for points in article_groups
            }
            template_points = article_groups[0]
            plans = []
            for canonical, provision in replacement_by_location.items():
                old_points = existing_by_location.get(canonical)
                if old_points and operation["operation_id"] in (
                    self._group_metadata(old_points).get("applied_operation_ids") or []
                ):
                    continue
                template = self._group_metadata(old_points or template_points)
                plans.append({
                    "old_points": old_points,
                    "content": provision["provision_content"],
                    "operation": operation,
                    "template_metadata": template,
                    "sac_tier1": self._group_sac_tier1(
                        old_points or template_points
                    ),
                    "location": provision["location_label"],
                    "initial_version": old_points is None,
                })
            if plans:
                self._insert_successors_batch(plans)
            removed_locations = set(existing_by_location) - set(replacement_by_location)
            for canonical in removed_locations:
                self._close_group(
                    existing_by_location[canonical],
                    operation["effective_date"],
                    None,
                )
            return len(plans) + len(removed_locations)

        raise ValueError(f"Chưa hỗ trợ target_level={level!r}")

    def update_existing_qdrant_document_effects(
        self,
        document_effects: List[Dict],
        amendments: List[Dict],
    ) -> int:
        """Apply document-level inverse relations to every version of the target."""
        effects_by_target: Dict[str, List[Dict]] = {}
        for effect in document_effects:
            if not isinstance(effect, dict):
                continue
            target_id = str(effect.get("target_document_id") or "").strip()
            source_id = str(effect.get("source_document_id") or "").strip()
            if target_id and source_id:
                effects_by_target.setdefault(target_id, []).append(effect)

        # Kept in the signature for compatibility with the current main flow.
        # Amendment locations govern provision versioning, not document-level
        # affects_documents/affected_by_documents propagation.
        _ = amendments

        today = date.today().isoformat()
        updated_points = 0
        terminal_types = {"thay_the", "bai_bo"}

        for target_id, target_effects in effects_by_target.items():
            points = self._scroll_document_points(target_id)
            if not points:
                print(f"  ⚠ Qdrant chưa có chunks của {target_id}")
                continue

            source_ids = {
                self._canonical_document_id(effect.get("source_document_id", ""))
                for effect in target_effects
            }
            whole_terminal_dates = [
                str(effect.get("effective_date"))
                for effect in target_effects
                if effect.get("impact_scope") == "whole_document"
                and terminal_types.intersection(effect.get("relation_types") or [])
                and effect.get("effective_date")
            ]
            document_expiration = (
                min(whole_terminal_dates) if whole_terminal_dates else None
            )
            target_updates = 0

            for point in points:
                metadata = deepcopy(((point.payload or {}).get("metadata") or {}))
                existing_affected_by = metadata.get("affected_by_documents") or []
                new_affected_by = [
                    relation
                    for relation in existing_affected_by
                    if self._canonical_document_id(
                        relation.get("source_document_id", "")
                    ) not in source_ids
                ]

                for effect in target_effects:
                    impact_scope = effect.get("impact_scope", "partial")
                    inverse = {
                        "source_document_id": effect.get("source_document_id"),
                        "relation_types": effect.get("relation_types") or [],
                        "impact_scope": impact_scope,
                        "effective_date": effect.get("effective_date"),
                        "status": effect.get("status", "active"),
                    }
                    if inverse not in new_affected_by:
                        new_affected_by.append(inverse)

                changed = new_affected_by != existing_affected_by
                metadata["affected_by_documents"] = new_affected_by

                if document_expiration:
                    if metadata.get("expiration_date") != document_expiration:
                        metadata["expiration_date"] = document_expiration
                        changed = True
                    if (
                        document_expiration <= today
                        and metadata.get("status") != "inactive"
                    ):
                        metadata["status"] = "inactive"
                        changed = True

                    # A whole-document terminal effect closes only the last open
                    # version of each provision. Historical intervals stay intact,
                    # and is_latest remains true because no same-key successor exists.
                    if (
                        metadata.get("is_latest", True)
                        and metadata.get("valid_to") is None
                    ):
                        metadata["valid_to"] = document_expiration
                        changed = True

                if not changed:
                    continue

                self.qdrant_client.set_payload(
                    collection_name=self.collection_name,
                    payload={"metadata": metadata},
                    points=[point.id],
                    wait=True,
                )
                target_updates += 1
                updated_points += 1

            print(
                f"  Updated versioned document effects: {target_id} "
                f"({target_updates}/{len(points)} points)"
            )

        return updated_points


def load_sources() -> Tuple[List[Tuple[str, str, str]], List[Dict], List[Dict]]:
    documents = []
    operations = []
    effects = []
    if not os.path.exists(OUTPUT_DIR):
        return documents, operations, effects

    for path in sorted(Path(OUTPUT_DIR).rglob("*.md")):
        filename = path.name
        if filename.endswith("_full.md"):
            continue
        markdown_file = str(path)
        base = filename[:-3]
        metadata_file = str(path.with_name(f"{base}_metadata.json"))
        summary_file = str(path.with_name(f"{base}_summary.json"))
        if not os.path.exists(metadata_file):
            continue
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        source_issued_date = metadata.get("issued_date") or "9999-12-31"
        for relation in metadata.get("amendment_relations") or []:
            operation = deepcopy(relation)
            operation["source_issued_date"] = source_issued_date
            operations.append(operation)
        for relation in metadata.get("affects_documents") or []:
            effect = deepcopy(relation)
            effect["source_document_id"] = metadata.get("document_id")
            effects.append(effect)

        if not metadata.get("embed_in_qdrant", True):
            continue
        if not os.path.exists(summary_file):
            print(f"⚠ Bỏ qua {filename}: thiếu summary Step 2")
            continue
        documents.append((markdown_file, summary_file, metadata_file))

    operations.sort(key=lambda item: (
        item.get("effective_date") or "9999-12-31",
        item.get("source_issued_date") or "9999-12-31",
        item.get("operation_id") or "",
    ))
    return documents, operations, effects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the version-aware legal corpus in Qdrant."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and count Step 1-2 artifacts without loading models or writing to Qdrant.",
    )
    args = parser.parse_args()

    documents, operations, effects = load_sources()

    print(f"Văn bản nền sẵn sàng: {len(documents)}")
    print(f"Version operations: {len(operations)}")
    print(f"Document-level effects: {len(effects)}")
    if args.validate_only:
        print("Validation complete; Qdrant was not accessed or modified.")
        return

    processor = VersionedEmbeddingQdrantProcessor()
    processor.create_collection(recreate=False)
    total = 0
    for markdown_file, summary_file, metadata_file in documents:
        total += processor.ingest_base_document(
            markdown_file,
            summary_file,
            metadata_file,
        )

    applied = 0
    failed_operations = []
    for operation in operations:
        print(
            f"Áp dụng {operation.get('loai_thay_doi')} | "
            f"{operation.get('van_ban_bi_sua')} | "
            f"{operation.get('target_location')}"
        )
        try:
            applied += processor.apply_operation(operation)
        except ValueError as exc:
            failed_operations.append((operation, str(exc)))
            print(f"  ❌ Không áp dụng được: {exc}")

    # Preserve document-level inverse effects and whole-document status handling.
    if effects:
        processor.update_existing_qdrant_document_effects(effects, operations)
    print(f"Hoàn thành: {total} base points, {applied} provision updates")
    if failed_operations:
        print("\nCác operation cần xử lý lại:")
        for operation, reason in failed_operations:
            print(
                f"- {operation.get('van_ban_bi_sua')} | "
                f"{operation.get('target_location')} | {reason}"
            )
        raise RuntimeError(
            f"Có {len(failed_operations)} operation chưa được áp dụng; "
            "các operation hợp lệ đã được lưu và có thể chạy lại an toàn."
        )


if __name__ == "__main__":
    main()
