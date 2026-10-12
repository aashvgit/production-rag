import json
import os
import random
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from data import load_scifact
from grounded import answer, cited_ids_valid

SPLIT = os.environ.get("SPLIT", "train")
N = int(os.environ.get("N", "25"))
DELAY = float(os.environ.get("DELAY", "7"))
TOP_K = 5
MODEL = "BAAI/bge-small-en-v1.5"
OUT = Path(f"results/grounding_{SPLIT}.jsonl")

corpus, queries, qrels = load_scifact(split=SPLIT)
doc_ids = list(corpus)
index_of = {d: i for i, d in enumerate(doc_ids)}
doc_emb = np.load(Path("data") / f"emb_{MODEL.replace('/', '_')}.npy")
model = SentenceTransformer(MODEL)

qids = sorted(qrels)
random.Random(0).shuffle(qids)
qids = qids[:N]

OUT.parent.mkdir(exist_ok=True)
done = set()
if OUT.exists():
    for line in OUT.read_text().splitlines():
        r = json.loads(line)
        done.add((r["qid"], r["condition"]))

with OUT.open("a") as f:
    for qid in qids:
        scores = doc_emb @ model.encode([queries[qid]], normalize_embeddings=True)[0]
        rel = [index_of[d] for d in qrels[qid]]
        for condition in ("answerable", "removed"):
            if (qid, condition) in done:
                continue
            s = scores.copy()
            if condition == "removed":
                s[rel] = -1.0  # delete this claim's relevant abstracts from the index
            top = np.argsort(-s)[:TOP_K]
            passages = [(doc_ids[i], corpus[doc_ids[i]]["title"], corpus[doc_ids[i]]["text"]) for i in top]
            result = answer(queries[qid], passages)
            rec = {"qid": qid, "condition": condition, "provided": [p[0] for p in passages],
                   "relevant": sorted(qrels[qid]), "result": result}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            verdict = result["verdict"] if result else "UNPARSEABLE"
            print(f"{condition:<10} {verdict:<13} {queries[qid][:70]}")
            time.sleep(DELAY)

records = [json.loads(line) for line in OUT.read_text().splitlines()]
records = [r for r in records if r["qid"] in set(qids)]
print(f"\n=== {SPLIT} split, {len(qids)} claims ===")
for condition in ("answerable", "removed"):
    rs = [r for r in records if r["condition"] == condition]
    parsed = [r for r in rs if r["result"]]
    answered = [r for r in parsed if r["result"]["verdict"] != "insufficient"]
    print(f"\n{condition}: {len(rs)} runs, {len(rs) - len(parsed)} unparseable replies")
    print(f"  answered (did not refuse): {len(answered)}/{len(parsed)}")
    print(f"  refused: {len(parsed) - len(answered)}/{len(parsed)}")
    if answered:
        valid = sum(cited_ids_valid(r["result"]["citations"], r["provided"]) for r in answered)
        print(f"  answers whose citations are all real and non-empty: {valid}/{len(answered)}")
        if condition == "answerable":
            hit = sum(bool(set(r["result"]["citations"]) & set(r["relevant"])) for r in answered)
            print(f"  answers citing at least one truly relevant abstract: {hit}/{len(answered)}")