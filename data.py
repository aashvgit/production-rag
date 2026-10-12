import csv
import json
from pathlib import Path

def load_scifact(root: str = "data/scifact", split: str = "test"):
    """Return (corpus, queries, qrels) for one split of SciFact."""
    root = Path(root)
    corpus = {}
    for line in (root / "corpus.jsonl").read_text().splitlines():
        d = json.loads(line)
        corpus[d["_id"]] = d
    queries = {}
    for line in (root / "queries.jsonl").read_text().splitlines():
        d = json.loads(line)
        queries[d["_id"]] = d["text"]
    qrels = {}  # query id -> set of relevant document ids
    with open(root / "qrels" / f"{split}.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    return corpus, queries, qrels

def doc_text(doc: dict) -> str:
    return f'{doc["title"]}. {doc["text"]}'