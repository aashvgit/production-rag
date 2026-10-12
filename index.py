import hashlib
import json
import os
from pathlib import Path

import numpy as np


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class IncrementalIndex:
    def __init__(self, directory, embed_fn, config: str):
        self.dir = Path(directory)
        self.embed_fn = embed_fn  # list of texts -> array of vectors
        self.config = config      # model name plus chunking settings; a change forces a rebuild
        self.ids: list[str] = []
        self.hashes: dict[str, str] = {}
        self.emb = np.zeros((0, 0), dtype=np.float32)
        self._load()

    def _load(self):
        manifest = self.dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            if m["config"] == self.config:
                self.ids, self.hashes = m["ids"], m["hashes"]
                self.emb = np.load(self.dir / "emb.npy")

    def _save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "emb.tmp"
        with open(tmp, "wb") as f:
            np.save(f, self.emb)
        os.replace(tmp, self.dir / "emb.npy")  # replace whole files, so a crash cannot leave half a write
        manifest_tmp = self.dir / "manifest.tmp"
        manifest_tmp.write_text(json.dumps(
            {"config": self.config, "ids": self.ids, "hashes": self.hashes}))
        os.replace(manifest_tmp, self.dir / "manifest.json")

    def sync(self, docs: dict[str, str]) -> dict:
        """docs: id -> text. Embed only new or changed texts; drop removed ones."""
        old = {doc_id: row for row, doc_id in enumerate(self.ids)}
        new_hashes = {i: text_hash(t) for i, t in docs.items()}
        todo = [i for i in docs if i not in old or self.hashes[i] != new_hashes[i]]
        fresh = dict(zip(todo, self.embed_fn([docs[i] for i in todo]))) if todo else {}
        ids = sorted(docs)  # a fixed order, so the result does not depend on history
        rows = [fresh[i] if i in fresh else self.emb[old[i]] for i in ids]
        stats = {
            "added": sum(i not in old for i in todo),
            "updated": sum(i in old for i in todo),
            "removed": len(set(old) - set(docs)),
            "unchanged": len(docs) - len(todo),
            "embedded": len(todo),
        }
        self.ids, self.hashes = ids, new_hashes
        self.emb = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
        self._save()
        return stats

    def search(self, query_vec, k: int = 10) -> list[str]:
        scores = self.emb @ query_vec
        return [self.ids[i] for i in np.argsort(-scores)[:k]]