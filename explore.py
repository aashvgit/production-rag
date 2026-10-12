import csv
import json
from pathlib import Path

ROOT = Path("data/scifact")

corpus = {}
for line in (ROOT / "corpus.jsonl").read_text().splitlines():
    d = json.loads(line)
    corpus[d["_id"]] = d

queries = {}
for line in (ROOT / "queries.jsonl").read_text().splitlines():
    d = json.loads(line)
    queries[d["_id"]] = d["text"]

qrels = {}  # query id -> set of relevant document ids (test split)
with open(ROOT / "qrels" / "test.tsv") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if int(row["score"]) > 0:
            qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])

lengths = [len((d["title"] + " " + d["text"]).split()) for d in corpus.values()]
print(len(corpus), "documents; average", round(sum(lengths) / len(lengths)), "words, longest", max(lengths))
print(len(queries), "queries in total;", len(qrels), "have relevance labels in the test split")
print("relevant documents per test query:", round(sum(len(v) for v in qrels.values()) / len(qrels), 2))

qid = sorted(qrels)[0]
print("\nQUERY:", queries[qid])
for did in qrels[qid]:
    print("\nRELEVANT DOCUMENT:", corpus[did]["title"])
    print(corpus[did]["text"][:400], "...")