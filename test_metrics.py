import math

from metrics import mrr_at_k, ndcg_at_k, recall_at_k


def test_recall():
    assert recall_at_k(["a", "b", "c"], {"b"}, 1) == 0.0
    assert recall_at_k(["a", "b", "c"], {"b"}, 2) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5


def test_mrr():
    assert mrr_at_k(["a", "b", "c"], {"a"}, 10) == 1.0
    assert mrr_at_k(["a", "b", "c"], {"b"}, 10) == 0.5
    assert mrr_at_k(["a", "b", "c"], {"z"}, 10) == 0.0
    assert mrr_at_k(["a", "b", "c"], {"c"}, 2) == 0.0  # beyond the cutoff


def test_ndcg():
    assert ndcg_at_k(["a", "b"], {"a"}, 10) == 1.0
    assert math.isclose(ndcg_at_k(["a", "b"], {"b"}, 10), 1 / math.log2(3))
    assert ndcg_at_k(["a", "b"], {"z"}, 10) == 0.0