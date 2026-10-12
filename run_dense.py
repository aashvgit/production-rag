from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_scifact
from eval_lib import evaluate

MODEL = "BAAI/bge-small-en-v1.5"
corpus, queries, qrels = load_scifact()
doc_ids = list(corpus)
model = SentenceTransformer(MODEL)
doc_emb = np.load(Path("data") / f"emb_{MODEL.replace('/', '_')}.npy")


def rank(query):
    q = model.encode([query], normalize_embeddings=True)[0]
    return [doc_ids[i] for i in np.argsort(-(doc_emb @ q))[:100]]


evaluate("dense", rank, queries, qrels)