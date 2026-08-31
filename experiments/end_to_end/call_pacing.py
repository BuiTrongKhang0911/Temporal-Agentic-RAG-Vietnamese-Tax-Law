"""Experiment-only pacing helpers for external language-model calls."""

from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any


def pause_after_call(delay_seconds: float) -> None:
    """Pause after a successful experiment call when pacing is enabled."""

    delay = max(0.0, delay_seconds)
    if delay > 0:
        time.sleep(delay)


@contextmanager
def paced_gemini_calls(delay_seconds: float) -> Iterator[None]:
    """Temporarily pace Gemini SDK calls made inside an experiment block."""

    delay = max(0.0, delay_seconds)
    if delay == 0:
        yield
        return

    import google.generativeai as genai

    original = genai.GenerativeModel.generate_content

    def generate_content(model: Any, *args: Any, **kwargs: Any) -> Any:
        response = original(model, *args, **kwargs)
        time.sleep(delay)
        return response

    genai.GenerativeModel.generate_content = generate_content
    try:
        yield
    finally:
        genai.GenerativeModel.generate_content = original
