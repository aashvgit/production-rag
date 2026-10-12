from sentence_transformers import CrossEncoder

from data import doc_text, load_scifact
from eval_lib import evaluate
from retrieval import Retriever
import os

W_BM25 = 0.25
CANDIDATES = int(os.environ.get("CANDIDATES", "50"))
RERANKER = os.environ.get("RERANKER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RUN_NAME = os.environ.get("RUN_NAME", "rerank")

corpus, queries, qrels = load_scifact(split="test")
ret = Retriever(corpus)
reranker = CrossEncoder(RERANKER, max_length=512)


def rank(query):
    full = ret.hybrid(query, w_bm25=W_BM25, k=100)
    candidates, rest = full[:CANDIDATES], full[CANDIDATES:]
    scores = reranker.predict([(query, doc_text(corpus[d])) for d in candidates], batch_size=32)
    reranked = [d for d, _ in sorted(zip(candidates, scores), key=lambda x: -x[1])]
    return reranked + rest  # the rest stays behind, so Recall@100 stays defined


evaluate(RUN_NAME, rank, queries, qrels)