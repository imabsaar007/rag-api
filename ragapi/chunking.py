"""Overlapping text chunker that prefers to break on whitespace."""

from __future__ import annotations


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into chunks of about ``size`` characters.

    Consecutive chunks share ``overlap`` characters so that facts straddling a
    boundary stay retrievable. Breaks land on whitespace when one is close to
    the limit, so words are not cut in half.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")

    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # back off to the last space in the final 20% of the window
            window_floor = start + int(size * 0.8)
            space = text.rfind(" ", window_floor, end)
            if space != -1:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = end - overlap
        # always make forward progress
        start = next_start if next_start > start else end
        # start on a word boundary
        if start < n and text[start - 1 : start] != " ":
            nxt = text.find(" ", start)
            if nxt != -1 and nxt < end:
                start = nxt + 1
    return chunks
