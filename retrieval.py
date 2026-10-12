import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from data import doc_text

MODEL = "BAAI/bge-small-en-v1.5"


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings, weights, rrf_k=60, k=100):
    """Reciprocal rank fusion of several ranked lists of document ids."""
    fused = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + weight / (rrf_k + rank)
    return sorted(fused, key=lambda d: (-fused[d], d))[:k]


class Retriever:
    def __init__(self, corpus):
        self.doc_ids = list(corpus)
        self.model = SentenceTransformer(MODEL)
        self.doc_emb = np.load(Path("data") / f"emb_{MODEL.replace('/', '_')}.npy")
        self.bm25 = BM25Okapi([tokenize(doc_text(corpus[i])) for i in self.doc_ids])

    def dense(self, query, k=100):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        return [self.doc_ids[i] for i in np.argsort(-(self.doc_emb @ q))[:k]]

    def sparse(self, query, k=100):
        scores = self.bm25.get_scores(tokenize(query))
        return [self.doc_ids[i] for i in np.argsort(-scores)[:k]]

    def hybrid(self, query, w_bm25=1.0, rrf_k=60, k=100):
        return rrf([self.dense(query), self.sparse(query)], [1.0, w_bm25], rrf_k, k)