import json
import random
import sys
from pathlib import Path


def load(name):
    return json.loads(Path(f"results/{name}.json").read_text())


a, b = load(sys.argv[1]), load(sys.argv[2])
metric = sys.argv[3] if len(sys.argv) > 3 else "nDCG@10"
assert a["qids"] == b["qids"]
diffs = [y - x for x, y in zip(a["per_query"][metric], b["per_query"][metric])]
n = len(diffs)
rng = random.Random(0)
boots = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(2000))
print(f"{metric}: {sys.argv[2]} minus {sys.argv[1]} = {sum(diffs) / n:+.3f} "
      f"(95% range {boots[50]:+.3f} to {boots[1950]:+.3f})")
print(f"{sys.argv[2]} better on {sum(d > 0 for d in diffs)} queries, "
      f"worse on {sum(d < 0 for d in diffs)}, tied on {sum(d == 0 for d in diffs)}")
print("If the range excludes 0, the difference is probably real.")