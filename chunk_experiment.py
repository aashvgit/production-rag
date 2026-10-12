import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_scifact
from eval_lib import evaluate

MODEL = "BAAI/bge-small-en-v1.5"
VARIANTS = {
    "whole": dict(window=1000, stride=1000, with_title=True),
    "sent1": dict(window=1, stride=1, with_title=True),
    "win3": dict(window=3, stride=2, with_title=True),
    "win5": dict(window=5, stride=3, with_title=True),
    "win3_notitle": dict(window=3, stride=2, with_title=False),
}


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]  # rough splitter


def make_chunks(corpus, window, stride, with_title):
    texts, owner = [], []
    for idx, d in enumerate(corpus.values()):
        sents = sentences(d["text"]) or [d["title"]]
        for start in range(0, len(sents), stride):
            piece = " ".join(sents[start:start + window])
            texts.append(f'{d["title"]}. {piece}' if with_title else piece)
            owner.append(idx)
            if start + window >= len(sents):
                break
    return texts, np.array(owner)


split = sys.argv[1]
names = sys.argv[2:] or list(VARIANTS)

corpus, queries, qrels = load_scifact(split=split)
doc_ids = list(corpus)
model = SentenceTransformer(MODEL)

for name in names:
    texts, owner = make_chunks(corpus, **VARIANTS[name])
    cache = Path("data") / f"chunks_{name}.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        print(f"{name}: embedding {len(texts)} chunks...")
        emb = model.encode(texts, batch_size=32, normalize_embeddings=True)
        np.save(cache, emb)

    def rank(query, emb=emb, owner=owner):
        q = model.encode([query], normalize_embeddings=True)[0]
        doc_scores = np.full(len(doc_ids), -1.0)
        np.maximum.at(doc_scores, owner, emb @ q)  # a document scores as its best chunk
        return [doc_ids[i] for i in np.argsort(-doc_scores)[:100]]

    evaluate(f"chunk_{name}_{split}", rank, queries, qrels)