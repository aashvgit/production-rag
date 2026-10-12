import json
import os
from pathlib import Path

from data import load_scifact
from eval_lib import evaluate
from retrieval import Retriever

SPLIT = os.environ.get("SPLIT", "train")
MAX_OVERLAP = 1  # fixed in advance: a follow-up repeating the topic words is not context-dependent

corpus, queries, qrels = load_scifact(split=SPLIT)
records = [json.loads(l) for l in Path(f"results/followups_{SPLIT}.jsonl").read_text().splitlines()]
kept = [r for r in records if r["overlap"] <= MAX_OVERLAP]
print(f"{len(kept)} of {len(records)} follow-ups kept (at most {MAX_OVERLAP} content word shared with the first message)")

sub_qrels = {r["qid"]: qrels[r["qid"]] for r in kept}
variants = {
    "original": {r["qid"]: r["claim"] for r in kept},                       # the first message alone
    "raw": {r["qid"]: r["followup"] for r in kept},                          # follow-up only
    "concat": {r["qid"]: r["claim"] + " " + r["followup"] for r in kept},    # history + follow-up
    "rewrite": {r["qid"]: r["rewrite"] for r in kept},                       # model rewrite
}
ret = Retriever(corpus)
for name, qs in variants.items():
    evaluate(f"fu_{SPLIT}_{name}", ret.dense, qs, sub_qrels)