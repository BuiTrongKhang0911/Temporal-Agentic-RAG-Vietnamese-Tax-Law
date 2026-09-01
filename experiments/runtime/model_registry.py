"""Process-wide cache for retrieval models shared by experiment components."""

from __future__ import annotations

import threading

from sentence_transformers import SentenceTransformer


_LOCK = threading.Lock()
_EMBEDDING_MODELS: dict[tuple[str, str], SentenceTransformer] = {}


def get_embedding_model(
    model_name: str,
    *,
    device: str = "cpu",
) -> SentenceTransformer:
    """Return one shared embedding model per model/device pair."""

    key = (model_name, device)
    with _LOCK:
        model = _EMBEDDING_MODELS.get(key)
        if model is None:
            model = SentenceTransformer(
                model_name,
                device=device,
            )
            _EMBEDDING_MODELS[key] = model
    return model
