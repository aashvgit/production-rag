import json
from pathlib import Path

from data import load_scifact

_, queries, _ = load_scifact()


def load(name):
    return json.loads(Path(f"results/{name}.json").read_text())


dense, bm25 = load("dense"), load("bm25")
qids = dense["qids"]
diff = sorted(
    (bm25["per_query"]["nDCG@10"][i] - dense["per_query"]["nDCG@10"][i], qids[i])
    for i in range(len(qids))
)
print("DENSE much better than BM25:")
for delta, q in diff[:5]:
    print(f"  {delta:+.2f}  {queries[q]}")
print("\nBM25 much better than dense:")
for delta, q in diff[-5:][::-1]:
    print(f"  {delta:+.2f}  {queries[q]}")