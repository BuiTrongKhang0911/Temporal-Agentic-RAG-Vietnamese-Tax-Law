"""Restore the submitted Qdrant collection snapshot."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import requests

from config import QDRANT_COLLECTION, QDRANT_HOST, QDRANT_PORT


HERE = Path(__file__).resolve().parent
SNAPSHOT_DIR = HERE / "qdrant_snapshot"
SNAPSHOT_INFO = SNAPSHOT_DIR / "Snapshot-GoogleDrive-Link.txt"


def expected_sha256() -> str | None:
    if not SNAPSHOT_INFO.is_file():
        return None
    match = re.search(
        r"SHA-256:\s*([0-9a-fA-F]{64})",
        SNAPSHOT_INFO.read_text(encoding="utf-8"),
    )
    return match.group(1).lower() if match else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=QDRANT_HOST)
    parser.add_argument("--port", type=int, default=QDRANT_PORT)
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow the snapshot to replace an existing collection.",
    )
    args = parser.parse_args()

    snapshot = args.snapshot
    if snapshot is None:
        candidates = sorted(SNAPSHOT_DIR.glob("*.snapshot"))
        if len(candidates) != 1:
            raise ValueError(
                "Expected exactly one .snapshot file in qdrant_snapshot; "
                f"found {len(candidates)}."
            )
        snapshot = candidates[0]
    snapshot = snapshot.resolve()
    if not snapshot.is_file():
        raise FileNotFoundError(snapshot)

    expected_hash = expected_sha256()
    if expected_hash:
        actual_hash = sha256_file(snapshot)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Snapshot SHA-256 mismatch: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
        print(f"Snapshot SHA-256 verified: {actual_hash}")

    base_url = f"http://{args.host}:{args.port}"
    collection_url = f"{base_url}/collections/{args.collection}"
    existing = requests.get(collection_url, timeout=30)
    if existing.status_code == 200 and not args.force:
        raise RuntimeError(
            f"Collection {args.collection!r} already exists. Use --force only "
            "when replacing it is intended."
        )

    upload_url = f"{collection_url}/snapshots/upload"
    with snapshot.open("rb") as handle:
        response = requests.post(
            upload_url,
            params={"wait": "true", "priority": "snapshot"},
            files={"snapshot": (snapshot.name, handle)},
            timeout=3600,
        )
    response.raise_for_status()

    info = requests.get(collection_url, timeout=30)
    info.raise_for_status()
    result = info.json().get("result") or {}
    print(
        f"Restored {args.collection}: status={result.get('status')}, "
        f"points={result.get('points_count')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
