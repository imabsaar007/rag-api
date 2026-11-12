import pytest

from ragapi.vectorstore import InMemoryStore, build_store, RemoteVectorDBStore


def test_search_returns_closest_first():
    store = InMemoryStore()
    store.add("d", ["a", "b", "c"], [[1, 0], [0, 1], [0.9, 0.1]])
    hits = store.search([1, 0], k=3)
    assert [h.text for h in hits] == ["a", "c", "b"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[0].score > hits[1].score > hits[2].score


def test_k_limits_results():
    store = InMemoryStore()
    store.add("d", ["a", "b", "c"], [[1, 0], [0, 1], [1, 1]])
    assert len(store.search([1, 0], k=2)) == 2


def test_empty_store_returns_no_hits():
    assert InMemoryStore().search([1, 0], k=3) == []


def test_reingest_replaces_old_chunks():
    store = InMemoryStore()
    store.add("d", ["old1", "old2"], [[1, 0], [0, 1]])
    store.add("d", ["new"], [[1, 0]])
    assert store.stats() == {"documents": 1, "chunks": 1}
    assert store.search([1, 0], k=5)[0].text == "new"


def test_delete_doc():
    store = InMemoryStore()
    store.add("a", ["x"], [[1, 0]])
    store.add("b", ["y"], [[0, 1]])
    assert store.delete_doc("a") == 1
    assert store.delete_doc("a") == 0
    assert store.stats()["documents"] == 1


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        InMemoryStore().add("d", ["a", "b"], [[1, 0]])


def test_zero_vector_does_not_crash():
    store = InMemoryStore()
    store.add("d", ["a"], [[0, 0]])
    assert store.search([1, 0], k=1)[0].score == 0.0


def test_build_store_backends():
    assert isinstance(build_store("memory", ""), InMemoryStore)
    assert isinstance(build_store("vectordb", "http://x:1"), RemoteVectorDBStore)
    with pytest.raises(ValueError):
        build_store("nope", "")
