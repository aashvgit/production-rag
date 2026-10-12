import json
import os
import random
import time
from pathlib import Path

from data import doc_text, load_scifact
from followups import GEN_PROMPT, REWRITE_PROMPT, overlap
from llm import generate

SPLIT = os.environ.get("SPLIT", "train")
N = int(os.environ.get("N", "30"))
DELAY = float(os.environ.get("DELAY", "7"))
OUT = Path(f"results/followups_{SPLIT}.jsonl")

corpus, queries, qrels = load_scifact(split=SPLIT)
qids = sorted(qrels)
random.Random(1).shuffle(qids)
qids = qids[:N]

OUT.parent.mkdir(exist_ok=True)
done = {json.loads(l)["qid"] for l in OUT.read_text().splitlines()} if OUT.exists() else set()
with OUT.open("a") as f:
    for qid in qids:
        if qid in done:
            continue
        claim = queries[qid]
        abstract = doc_text(corpus[sorted(qrels[qid])[0]])
        followup = generate(GEN_PROMPT.format(claim=claim, abstract=abstract)).strip().strip('"')
        time.sleep(DELAY)
        rewrite = generate(REWRITE_PROMPT.format(claim=claim, followup=followup)).strip().strip('"')
        time.sleep(DELAY)
        rec = {"qid": qid, "claim": claim, "followup": followup, "rewrite": rewrite,
               "overlap": overlap(claim, followup)}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        print(f"[overlap {rec['overlap']}] {followup}  ->  {rewrite}")