from data import load_scifact
from eval_lib import evaluate
from metrics import ndcg_at_k
from retrieval import Retriever, rrf

corpus, queries, qrels_test = load_scifact(split="test")
_, _, qrels_train = load_scifact(split="train")
ret = Retriever(corpus)

train_ids = sorted(qrels_train)
dense_r = {q: ret.dense(queries[q]) for q in train_ids}
sparse_r = {q: ret.sparse(queries[q]) for q in train_ids}

print(f"tuning on {len(train_ids)} TRAIN queries (the test queries are not used here)")
best_w, best_score = None, -1.0
for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:  # BM25 weight; the dense weight stays 1
    score = sum(
        ndcg_at_k(rrf([dense_r[q], sparse_r[q]], [1.0, w]), qrels_train[q], 10)
        for q in train_ids
    ) / len(train_ids)
    print(f"  BM25 weight {w:<4}: train nDCG@10 {score:.3f}")
    if score > best_score:
        best_w, best_score = w, score
print(f"chosen BM25 weight: {best_w}")

evaluate("hybrid", lambda query: ret.hybrid(query, w_bm25=best_w), queries, qrels_test)