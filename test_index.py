import zlib

import numpy as np

from index import IncrementalIndex

calls = []


def fake_embed(texts):
    calls.append(len(texts))
    out = []
    for t in texts:
        v = np.random.default_rng(zlib.crc32(t.encode())).normal(size=8)
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


DOCS = {"a": "alpha text", "b": "beta text", "c": "gamma text"}


def test_first_sync_embeds_everything(tmp_path):
    calls.clear()
    stats = IncrementalIndex(tmp_path, fake_embed, "m1").sync(DOCS)
    assert stats["embedded"] == 3 and calls == [3]


def test_unchanged_sync_embeds_nothing(tmp_path):
    index = IncrementalIndex(tmp_path, fake_embed, "m1")
    index.sync(DOCS)
    calls.clear()
    stats = index.sync(DOCS)
    assert stats["embedded"] == 0 and calls == [] and stats["unchanged"] == 3


def test_incremental_equals_full_rebuild(tmp_path):
    incremental = IncrementalIndex(tmp_path / "inc", fake_embed, "m1")
    incremental.sync(DOCS)
    edited = {"a": "alpha text REVISED", "c": "gamma text", "d": "delta text"}  # edit a, drop b, add d
    calls.clear()
    stats = incremental.sync(edited)
    assert stats == {"added": 1, "updated": 1, "removed": 1, "unchanged": 1, "embedded": 2}
    assert calls == [2]  # only the changed and the new text were embedded

    full = IncrementalIndex(tmp_path / "full", fake_embed, "m1")
    full.sync(edited)
    assert incremental.ids == full.ids
    assert np.allclose(incremental.emb, full.emb)


def test_config_change_forces_full_rebuild(tmp_path):
    IncrementalIndex(tmp_path, fake_embed, "m1").sync(DOCS)
    calls.clear()
    stats = IncrementalIndex(tmp_path, fake_embed, "m2").sync(DOCS)  # a different model name
    assert stats["embedded"] == 3


def test_index_survives_restart(tmp_path):
    IncrementalIndex(tmp_path, fake_embed, "m1").sync(DOCS)
    calls.clear()
    stats = IncrementalIndex(tmp_path, fake_embed, "m1").sync(DOCS)  # a new object loads from disk
    assert stats["embedded"] == 0 and calls == []