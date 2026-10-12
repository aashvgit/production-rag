import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant document in the top k, else 0."""
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Rewards relevant documents near the top (binary relevance)."""
    dcg = sum(1 / math.log2(r + 1) for r, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0