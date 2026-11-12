import pytest

from ragapi.chunking import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("hello   world", size=100, overlap=10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\t ", size=100, overlap=10) == []


def test_chunks_respect_size_and_cover_all_words():
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_consecutive_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, size=120, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        shared = set(a.split()) & set(b.split())
        assert shared, "adjacent chunks should share at least one word"


def test_no_words_are_cut_in_half():
    words = [f"token{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), size=90, overlap=20)
    valid = set(words)
    for chunk in chunks:
        assert set(chunk.split()) <= valid


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 20)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size=size, overlap=overlap)


def test_long_word_without_spaces_still_terminates():
    chunks = chunk_text("x" * 1000, size=100, overlap=10)
    assert chunks and all(len(c) <= 100 for c in chunks)
