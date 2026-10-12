import json
import random
from pathlib import Path
import time
from metrics import mrr_at_k, ndcg_at_k, recall_at_k

METRICS = {
    "Recall@10": lambda r, rel: recall_at_k(r, rel, 10),
    "Recall@100": lambda r, rel: recall_at_k(r, rel, 100),
    "MRR@10": lambda r, rel: mrr_at_k(r, rel, 10),
    "nDCG@10": lambda r, rel: ndcg_at_k(r, rel, 10),
}


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% range for the mean, by resampling the queries."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def evaluate(name, rank_fn, queries, qrels):
    """rank_fn(query_text) -> ranked list of document ids, best first (at least 100)."""
    qids = sorted(qrels)
    per_query = {m: [] for m in METRICS}
    elapsed = 0.0
    for qid in qids:
        start = time.perf_counter()
        ranked = rank_fn(queries[qid])
        elapsed += time.perf_counter() - start
        for m, fn in METRICS.items():
            per_query[m].append(fn(ranked, qrels[qid]))
    ms = 1000 * elapsed / len(qids)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(
        json.dumps({"qids": qids, "per_query": per_query, "ms_per_query": ms})
    )
    print(f"\n{name} ({len(qids)} queries, {ms:.0f} ms per query)")
    for m, vals in per_query.items():
        lo, hi = bootstrap_ci(vals)
        print(f"  {m}: {sum(vals) / len(vals):.3f}  (95% range {lo:.3f} to {hi:.3f})")