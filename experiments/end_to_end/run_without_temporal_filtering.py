"""Run the full proposed pipeline with temporal retrieval control disabled."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DEFAULT_OUTPUT = HERE / "result" / "rerun" / "without_temporal_filtering"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UNBOUNDED_SCOPE = {
    "type": "date_range",
    "from": {"granularity": "day", "value": "0001-01-01"},
    "to": {"granularity": "day", "value": "9999-12-31"},
}


def _argument_value(name: str, default: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def install_no_temporal_filter_patch() -> None:
    """Keep every stored legal version eligible during retrieval."""

    from experiments.runtime.retriever.base import RAGQuestionAnswering
    from experiments.runtime.retriever.temporal import (
        normalize_temporal_scope,
        temporal_scope_key,
    )

    def select_all_points(self: Any, points: list[object]) -> list[object]:
        return list(points)

    def all_versions_are_applicable(
        self: Any, metadata: dict[str, Any]
    ) -> bool:
        return True

    def remember_unbounded_scope(self: Any, scope: dict[str, Any]) -> None:
        del scope
        normalized = normalize_temporal_scope(UNBOUNDED_SCOPE)
        self.temporal_scope = normalized
        self.target_date = None
        self._temporal_scope_key = temporal_scope_key(normalized)

    RAGQuestionAnswering._select_applicable_points = select_all_points
    RAGQuestionAnswering._is_temporally_applicable = all_versions_are_applicable
    RAGQuestionAnswering.set_temporal_scope = remember_unbounded_scope


def main() -> int:
    system = _argument_value("--system", "gemini")
    os.environ["SUBMISSION_RUNTIME_PROFILE"] = system

    # Let argparse render help without importing the Retriever/model stack.
    if "-h" in sys.argv or "--help" in sys.argv:
        from experiments.end_to_end.run_proposed_system import main as run_proposed

        return run_proposed()

    install_no_temporal_filter_patch()
    print("Ablation: temporal filtering and version selection are DISABLED")

    if "--output-dir" not in sys.argv:
        sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT)])
    if "--configuration" not in sys.argv:
        sys.argv.extend(
            ["--configuration", f"proposed_{system}_without_temporal_filtering"]
        )

    from experiments.end_to_end.run_proposed_system import main as run_proposed

    return run_proposed()


if __name__ == "__main__":
    raise SystemExit(main())
