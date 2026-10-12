import re

import numpy as np
from rank_bm25 import BM25Okapi

from data import doc_text, load_scifact
from eval_lib import evaluate


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


corpus, queries, qrels = load_scifact()
doc_ids = list(corpus)
bm25 = BM25Okapi([tokenize(doc_text(corpus[i])) for i in doc_ids])


def rank(query):
    scores = bm25.get_scores(tokenize(query))
    return [doc_ids[i] for i in np.argsort(-scores)[:100]]


evaluate("bm25", rank, queries, qrels)