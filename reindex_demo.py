import time

from sentence_transformers import SentenceTransformer

from data import doc_text, load_scifact
from index import IncrementalIndex

MODEL = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL)
embed = lambda texts: model.encode(texts, batch_size=32, normalize_embeddings=True)

corpus, _, _ = load_scifact()
docs = {i: doc_text(d) for i, d in corpus.items()}
index = IncrementalIndex("data/index_demo", embed, config=MODEL)


def timed(label, d):
    start = time.perf_counter()
    stats = index.sync(d)
    print(f"{label}: {stats}  ({time.perf_counter() - start:.1f} s)")


timed("first build", docs)
timed("nothing changed", docs)

edited = dict(docs)
ids = sorted(edited)
for i in ids[:5]:
    edited[i] += " (revised)"
del edited[ids[10]]
edited["new-1"] = "A newly added abstract about the effects of exercise on sleep quality."
timed("5 edited, 1 deleted, 1 added", edited)