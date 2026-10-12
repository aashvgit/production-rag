import json
import os
import random
import time
from pathlib import Path

from data import load_scifact
from eval_lib import evaluate
from followups import REWRITE_HISTORY_PROMPT
from llm import generate
from retrieval import Retriever

SPLIT = os.environ.get("SPLIT", "train")
DELAY = float(os.environ.get("DELAY", "7"))
OUT = Path(f"results/switch_{SPLIT}.jsonl")

corpus, queries, qrels = load_scifact(split=SPLIT)
records = [json.loads(l) for l in Path(f"results/followups_{SPLIT}.jsonl").read_text().splitlines()]
records = [r for r in records if r["overlap"] <= 1]
all_qids = sorted(qrels)

cache = {}
if OUT.exists():
    for line in OUT.read_text().splitlines():
        d = json.loads(line)
        cache[d["qid"]] = d

with OUT.open("a") as f:
    for r in records:
        if r["qid"] in cache:
            continue
        rng = random.Random(f"switch-{r['qid']}")  # same unrelated claim on every run
        earlier = queries[rng.choice([q for q in all_qids if q != r["qid"]])]
        history = f"1. {earlier}\n2. {r['claim']}"
        start = time.perf_counter()
        rewrite = generate(REWRITE_HISTORY_PROMPT.format(history=history, followup=r["followup"]))
        seconds = time.perf_counter() - start
        rewrite = rewrite.strip().strip('"')
        d = {"qid": r["qid"], "earlier": earlier, "rewrite": rewrite, "seconds": round(seconds, 2)}
        f.write(json.dumps(d) + "\n")
        f.flush()
        cache[r["qid"]] = d
        print(f"{r['followup']}  ->  {rewrite}")
        time.sleep(DELAY)

sub_qrels = {r["qid"]: qrels[r["qid"]] for r in records}
concat_all = {r["qid"]: f'{cache[r["qid"]]["earlier"]} {r["claim"]} {r["followup"]}' for r in records}
rewrite_all = {r["qid"]: cache[r["qid"]]["rewrite"] for r in records}
ret = Retriever(corpus)
evaluate(f"sw_{SPLIT}_concat", ret.dense, concat_all, sub_qrels)
evaluate(f"sw_{SPLIT}_rewrite", ret.dense, rewrite_all, sub_qrels)

import statistics

times = [d["seconds"] for d in cache.values() if "seconds" in d]
if times:
    print(f"median rewrite call: {statistics.median(times):.2f} s over {len(times)} calls")