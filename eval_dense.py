from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import doc_text, load_scifact
from metrics import mrr_at_k, ndcg_at_k, recall_at_k

MODEL = "BAAI/bge-small-en-v1.5"

corpus, queries, qrels = load_scifact()
doc_ids = list(corpus)
model = SentenceTransformer(MODEL)
print("the model reads at most", model.max_seq_length, "tokens per input")

cache = Path("data") / f"emb_{MODEL.replace('/', '_')}.npy"
if cache.exists():
    doc_emb = np.load(cache)
else:
    texts = [doc_text(corpus[i]) for i in doc_ids]
    doc_emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    np.save(cache, doc_emb)

qids = sorted(qrels)
q_emb = model.encode([queries[q] for q in qids], normalize_embeddings=True)
scores = q_emb @ doc_emb.T  # cosine similarity, since the vectors are normalized

totals = {"Recall@10": 0.0, "Recall@100": 0.0, "MRR@10": 0.0, "nDCG@10": 0.0}
for row, qid in zip(scores, qids):
    ranked = [doc_ids[i] for i in np.argsort(-row)[:100]]
    rel = qrels[qid]
    totals["Recall@10"] += recall_at_k(ranked, rel, 10)
    totals["Recall@100"] += recall_at_k(ranked, rel, 100)
    totals["MRR@10"] += mrr_at_k(ranked, rel, 10)
    totals["nDCG@10"] += ndcg_at_k(ranked, rel, 10)

print(f"\n{MODEL} on SciFact ({len(qids)} test queries)")
for name, total in totals.items():
    print(f"  {name}: {total / len(qids):.3f}")