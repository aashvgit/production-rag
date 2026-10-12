from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_scifact

MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 5  # how many passages the generator would see

corpus, queries, qrels_test = load_scifact(split="test")
_, _, qrels_train = load_scifact(split="train")
doc_ids = list(corpus)
index_of = {d: i for i, d in enumerate(doc_ids)}
doc_emb = np.load(Path("data") / f"emb_{MODEL.replace('/', '_')}.npy")
model = SentenceTransformer(MODEL)


def signals(qrels):
    qids = sorted(qrels)
    q_emb = model.encode([queries[q] for q in qids], normalize_embeddings=True)
    scores = q_emb @ doc_emb.T
    rows = []
    for qid, s in zip(qids, scores):
        rel = [index_of[d] for d in qrels[qid] if d in index_of]
        top = np.argsort(-s)[:TOP_K]
        removed = s.copy()
        removed[rel] = -1.0  # delete this query's answer documents from the index
        rows.append({
            "ans_score": float(s[top[0]]),
            "ans_has_evidence": bool(set(top) & set(rel)),
            "rem_score": float(removed.max()),
        })
    return rows


def rates(rows, tau):
    answered = [r for r in rows if r["ans_score"] >= tau]
    answers_answerable = len(answered) / len(rows)
    false_answers = sum(r["rem_score"] >= tau for r in rows) / len(rows)
    evidence = sum(r["ans_has_evidence"] for r in answered) / max(1, len(answered))
    return answers_answerable, false_answers, evidence


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


train, test = signals(qrels_train), signals(qrels_test)
print(f"AUC (1.0 = perfect separation, 0.5 = no better than a coin): "
      f"train {auc([r['ans_score'] for r in train], [r['rem_score'] for r in train]):.3f}, "
      f"test {auc([r['ans_score'] for r in test], [r['rem_score'] for r in test]):.3f}")

all_scores = [r["ans_score"] for r in train] + [r["rem_score"] for r in train]
taus = np.quantile(all_scores, np.linspace(0.05, 0.95, 19))

print("\nTRAIN queries (the cutoff is chosen here)")
print("cutoff | answers answerable | answers unanswerable (bad) | evidence in context when answered")
best_tau, best_j = None, -1.0
for tau in taus:
    a, f, e = rates(train, tau)
    print(f"{tau:.3f}  |       {a:.0%}         |         {f:.0%}          |        {e:.0%}")
    if a - f > best_j:
        best_tau, best_j = tau, a - f

a, f, e = rates(test, best_tau)
print(f"\nChosen cutoff on train (largest gap between the two rates): {best_tau:.3f}")
print(f"TEST at that cutoff: answers {a:.0%} of answerable, answers {f:.0%} of unanswerable, "
      f"evidence present in {e:.0%} of the answered ones")