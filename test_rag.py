import numpy as np

from index import IncrementalIndex
from rag import RAG

VECS = {"alpha": [1, 0, 0], "beta": [0, 1, 0], "gamma": [0, 0, 1]}
DOCS = {"1": ("T1", "alpha"), "2": ("T2", "beta"), "3": ("T3", "gamma")}


def embed(texts):
    return np.array([VECS[t] for t in texts], dtype=np.float32)


def make_rag(tmp_path, answer_fn, rewrite_fn=None, **kw):
    index = IncrementalIndex(tmp_path, embed, "t")
    index.sync({i: text for i, (_, text) in DOCS.items()})
    return RAG(index, lambda t: np.array(VECS[t], dtype=np.float32), DOCS,
               answer_fn, rewrite_fn, **kw)


def ok(citations):
    return lambda q, p: {"verdict": "supported", "explanation": "x", "citations": citations}


def test_valid_citation_is_returned(tmp_path):
    result = make_rag(tmp_path, ok(["1"]), top_k=2).ask("alpha")
    assert result.verdict == "supported" and result.citations == ["1"]
    assert result.passages[0][0] == "1"


def test_invented_citation_is_withheld(tmp_path):
    assert make_rag(tmp_path, ok(["99"]), top_k=2).ask("alpha").verdict == "insufficient"


def test_answer_without_citations_is_withheld(tmp_path):
    assert make_rag(tmp_path, ok([]), top_k=2).ask("alpha").verdict == "insufficient"


def test_refusal_passes_through(tmp_path):
    refuse = lambda q, p: {"verdict": "insufficient", "explanation": "no", "citations": []}
    assert make_rag(tmp_path, refuse).ask("alpha").verdict == "insufficient"


def test_unparseable_reply_is_an_error(tmp_path):
    assert make_rag(tmp_path, lambda q, p: None).ask("alpha").verdict == "error"


def test_followup_is_rewritten_but_first_turn_is_not(tmp_path):
    seen = []

    def rewrite(history, message):
        seen.append((list(history), message))
        return "beta"

    rag = make_rag(tmp_path, ok(["2"]), rewrite, top_k=1)
    first = rag.ask("alpha")
    assert seen == [] and first.query == "alpha"
    second = rag.ask("what about it?", ["alpha"])
    assert second.query == "beta" and second.passages[0][0] == "2"


def test_history_is_truncated(tmp_path):
    seen = []
    rag = make_rag(tmp_path, ok(["1"]), lambda h, m: seen.append(list(h)) or "alpha", max_history=2)
    rag.ask("beta", ["alpha", "beta", "gamma", "alpha"])
    assert seen == [["gamma", "alpha"]]