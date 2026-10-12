import json
from pathlib import Path


def load(name):
    return json.loads(Path(f"results/{name}.json").read_text())["per_query"]["nDCG@10"]


d, b = load("dense"), load("bm25")
best = [max(x, y) for x, y in zip(d, b)]
print(f"dense {sum(d) / len(d):.3f}, bm25 {sum(b) / len(b):.3f}, "
      f"best of the two per query {sum(best) / len(best):.3f}")