import json
from collections import Counter
from pathlib import Path
import os
from data import load_scifact

SPLIT = os.environ.get("SPLIT", "train")
corpus, queries, _ = load_scifact(split=SPLIT)
records = [json.loads(line) for line in Path(f"results/grounding_{SPLIT}.jsonl").read_text().splitlines()]

counts = Counter()
over_refusals = []
for r in records:
    if r["condition"] != "answerable" or not r["result"]:
        continue
    has_evidence = bool(set(r["provided"]) & set(r["relevant"]))
    answered = r["result"]["verdict"] != "insufficient"
    if answered and has_evidence:
        counts["answered, labeled evidence was in the passages"] += 1
    elif answered:
        counts["answered, labeled evidence was NOT in the passages"] += 1
    elif has_evidence:
        counts["refused although the labeled evidence WAS in the passages (over-refusal)"] += 1
        over_refusals.append(r)
    else:
        counts["refused, labeled evidence NOT in the passages (a retrieval miss, so correct)"] += 1
for name, n in counts.items():
    print(f"{n:>3}  {name}")

print("\n=== OVER-REFUSALS: what the model refused to use ===")
for r in over_refusals:
    print("-" * 70)
    print("CLAIM:", queries[r["qid"]])
    print("MODEL SAID:", r["result"]["explanation"])
    for doc in r["relevant"]:
        if doc in r["provided"]:
            print(f"LABELED ABSTRACT [{doc}]:", corpus[doc]["title"], "|", corpus[doc]["text"][:300])

print("\n=== ANSWERED EVEN THOUGH THE LABELED ABSTRACT WAS REMOVED ===")
for r in records:
    if r["condition"] == "removed" and r["result"] and r["result"]["verdict"] != "insufficient":
        print("-" * 70)
        print("CLAIM:", queries[r["qid"]])
        print("VERDICT:", r["result"]["verdict"], "|", r["result"]["explanation"])
        for cid in r["result"]["citations"]:
            if cid in corpus:
                print(f"CITED [{cid}]:", corpus[cid]["title"], "|", corpus[cid]["text"][:300])