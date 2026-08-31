"""Deterministic temporal scopes for legal retrieval.

Agent 2 describes temporal intent. Python validates it and compiles it into
half-open legal interval checks, so downstream agents cannot loosen the scope.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional


SUPPORTED_TEMPORAL_TYPES = {
    "current",
    "exact_date",
    "month",
    "year",
    "date_range",
    "comparison",
}


def _iso_date(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD: {text!r}") from exc


def _range_boundary(
    value: object,
    field_name: str,
    *,
    side: str,
    current_day: str,
) -> tuple[str, Dict]:
    """Compile an Agent-2 range boundary while preserving its source precision."""

    # Backward compatibility for frozen plans created before structured boundaries.
    if not isinstance(value, dict):
        exact = _iso_date(value, field_name)
        return exact, {"granularity": "day", "value": exact}

    granularity = str(value.get("granularity") or "").strip().lower()
    raw = str(value.get("value") or "").strip().lower()
    allowed = {"year", "month", "day"}
    if side == "to":
        allowed.add("current")
    if granularity not in allowed:
        raise ValueError(
            f"{field_name}.granularity must be one of {sorted(allowed)}: "
            f"{granularity!r}"
        )

    if granularity == "current":
        return current_day, {"granularity": "current", "value": "current"}

    if granularity == "day":
        exact = _iso_date(raw, f"{field_name}.value")
        return exact, {"granularity": "day", "value": exact}

    if granularity == "year":
        if not re.fullmatch(r"\d{4}", raw):
            raise ValueError(f"{field_name}.value must use YYYY: {raw!r}")
        year = int(raw)
        datetime.strptime(f"{year:04d}-01-01", "%Y-%m-%d")
        compiled = f"{year:04d}-01-01" if side == "from" else f"{year:04d}-12-31"
        return compiled, {"granularity": "year", "value": f"{year:04d}"}

    try:
        parsed_month = datetime.strptime(raw, "%Y-%m")
    except ValueError as exc:
        raise ValueError(f"{field_name}.value must use YYYY-MM: {raw!r}") from exc
    year, month = parsed_month.year, parsed_month.month
    day = 1 if side == "from" else calendar.monthrange(year, month)[1]
    compiled = f"{year:04d}-{month:02d}-{day:02d}"
    return compiled, {"granularity": "month", "value": f"{year:04d}-{month:02d}"}


def normalize_temporal_scope(
    raw_scope: Optional[Dict],
    *,
    today: Optional[str] = None,
) -> Dict:
    """Validate an Agent-2 scope and produce a deterministic date interval."""

    scope = dict(raw_scope or {})
    temporal_type = str(scope.get("type") or "current").strip().lower()
    if temporal_type not in SUPPORTED_TEMPORAL_TYPES:
        raise ValueError(f"Unsupported temporal type: {temporal_type!r}")

    current_day = _iso_date(today or date.today().isoformat(), "today")
    normalized = {
        "type": temporal_type,
        "date": None,
        "month": None,
        "year": None,
        "from": None,
        "to": None,
        "comparison_scopes": [],
    }

    if temporal_type == "current":
        normalized.update({"date": current_day, "from": current_day, "to": current_day})
        return normalized

    if temporal_type == "exact_date":
        target = _iso_date(scope.get("date"), "temporal.date")
        normalized.update({"date": target, "from": target, "to": target})
        return normalized

    if temporal_type == "month":
        try:
            year = int(scope.get("year"))
            month = int(scope.get("month"))
            last_day = calendar.monthrange(year, month)[1]
        except (TypeError, ValueError) as exc:
            raise ValueError("temporal.month and temporal.year are invalid") from exc
        normalized.update(
            {
                "month": month,
                "year": year,
                "from": f"{year:04d}-{month:02d}-01",
                "to": f"{year:04d}-{month:02d}-{last_day:02d}",
            }
        )
        return normalized

    if temporal_type == "year":
        try:
            year = int(scope.get("year"))
            datetime.strptime(f"{year:04d}-01-01", "%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError("temporal.year is invalid") from exc
        normalized.update(
            {
                "year": year,
                "from": f"{year:04d}-01-01",
                "to": f"{year:04d}-12-31",
            }
        )
        return normalized

    if temporal_type == "date_range":
        start, start_boundary = _range_boundary(
            scope.get("from_boundary") or scope.get("from"),
            "temporal.from",
            side="from",
            current_day=current_day,
        )
        end, end_boundary = _range_boundary(
            scope.get("to_boundary") or scope.get("to"),
            "temporal.to",
            side="to",
            current_day=current_day,
        )
        if start > end:
            raise ValueError(f"temporal.from must be <= temporal.to: {start} > {end}")
        normalized.update(
            {
                "from": start,
                "to": end,
                "from_boundary": start_boundary,
                "to_boundary": end_boundary,
            }
        )
        return normalized

    comparison_scopes = scope.get("comparison_scopes") or []
    if not isinstance(comparison_scopes, list) or len(comparison_scopes) < 2:
        raise ValueError("comparison requires at least 2 comparison_scopes")
    normalized_children: List[Dict] = []
    for index, child in enumerate(comparison_scopes, 1):
        if not isinstance(child, dict):
            raise ValueError(f"comparison_scopes[{index}] must be an object")
        child_scope = normalize_temporal_scope(child, today=current_day)
        if child_scope["type"] == "comparison":
            raise ValueError("Nested comparison scopes are not supported")
        child_scope["label"] = str(child.get("label") or f"Mốc {index}")
        normalized_children.append(child_scope)
    normalized["comparison_scopes"] = normalized_children
    return normalized


def expand_comparison_task(task: Dict, *, today: Optional[str] = None) -> List[Dict]:
    """Turn a comparison Planner task into independent retrieval tasks."""

    normalized_scope = normalize_temporal_scope(task.get("temporal"), today=today)
    if normalized_scope["type"] != "comparison":
        result = dict(task)
        result["temporal"] = normalized_scope
        return [result]

    expanded = []
    for child in normalized_scope["comparison_scopes"]:
        child_scope = dict(child)
        label = child_scope.pop("label")
        parent_task = str(task.get("task") or task.get("query") or "Tra cứu")
        scoped_query = str(task.get("query") or parent_task)
        expanded.append(
            {
                **task,
                # A comparison child is an independent retrieval task. Do not
                # repeat the parent wording that mentions every comparison
                # date, otherwise Agent 3 may try to search the other child.
                "task": f"{scoped_query} [{label}]",
                "query": scoped_query,
                "comparison_parent_task": parent_task,
                "temporal": child_scope,
                "temporal_label": label,
            }
        )
    return expanded


def temporal_scope_key(scope: Dict) -> str:
    normalized = normalize_temporal_scope(scope)
    return "|".join(
        [
            normalized["type"],
            str(normalized.get("from") or ""),
            str(normalized.get("to") or ""),
        ]
    )


def describe_temporal_scope(scope: Dict) -> str:
    normalized = normalize_temporal_scope(scope)
    temporal_type = normalized["type"]
    if temporal_type == "current":
        return f"hiện hành tại ngày {normalized['date']}"
    if temporal_type == "exact_date":
        return f"tại đúng ngày {normalized['date']}"
    if temporal_type == "month":
        return f"trong tháng {normalized['month']:02d}/{normalized['year']}"
    if temporal_type == "year":
        return f"trong năm {normalized['year']}"
    return f"trong khoảng {normalized['from']} đến {normalized['to']}"


def _point_payload(point: object) -> Dict:
    """Read a payload from either a Qdrant point or a plain test dictionary."""

    if isinstance(point, dict):
        payload = point.get("payload", point)
    else:
        payload = getattr(point, "payload", None)
    return payload if isinstance(payload, dict) else {}


def metadata_overlaps_scope(metadata: Dict, scope: Dict) -> bool:
    """Return whether [valid_from, valid_to) overlaps the requested scope."""

    normalized = normalize_temporal_scope(scope)
    if normalized["type"] == "comparison":
        raise ValueError("comparison must be expanded before retrieval")

    valid_from = str(
        metadata.get("valid_from") or metadata.get("effective_date") or "0001-01-01"
    )[:10]
    valid_to_value = metadata.get("valid_to")
    valid_to = str(valid_to_value)[:10] if valid_to_value not in (None, "") else None
    start = normalized["from"]
    end = normalized["to"]

    # Empty or inverted source intervals are audit records, not applicable law.
    if valid_to is not None and valid_to <= valid_from:
        return False

    if normalized["type"] in {"current", "exact_date"}:
        return valid_from <= start and (valid_to is None or valid_to > start)

    return valid_from <= end and (valid_to is None or valid_to > start)


def select_points_for_scope(points: Iterable[object], scope: Dict) -> List[object]:
    """Filter points and collapse to the latest applicable version for point scopes."""

    normalized = normalize_temporal_scope(scope)
    selected = []
    max_versions: Dict[str, int] = {}
    collapse_versions = normalized["type"] in {"current", "exact_date"}

    for point in points:
        metadata = _point_payload(point).get("metadata") or {}
        if not metadata_overlaps_scope(metadata, normalized):
            continue
        selected.append(point)
        if collapse_versions:
            provision_key = str(metadata.get("provision_key") or "").strip()
            if provision_key:
                version = int(metadata.get("version") or 1)
                max_versions[provision_key] = max(max_versions.get(provision_key, 0), version)

    if not collapse_versions:
        return selected

    result = []
    for point in selected:
        metadata = _point_payload(point).get("metadata") or {}
        provision_key = str(metadata.get("provision_key") or "").strip()
        version = int(metadata.get("version") or 1)
        if provision_key and version != max_versions.get(provision_key):
            continue
        result.append(point)
    return result
