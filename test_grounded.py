import grounded


def test_parse_plain_json():
    r = grounded.parse('{"verdict": "supported", "explanation": "x", "citations": [12, "7"]}')
    assert r["verdict"] == "supported" and r["citations"] == ["12", "7"]


def test_parse_fenced_json():
    r = grounded.parse('```json\n{"verdict": "insufficient", "explanation": "", "citations": []}\n```')
    assert r["verdict"] == "insufficient"


def test_parse_rejects_garbage():
    assert grounded.parse("I think it is supported") is None
    assert grounded.parse('{"verdict": "maybe", "citations": []}') is None


def test_citation_check():
    assert grounded.cited_ids_valid(["1", "2"], ["1", "2", "3"])
    assert not grounded.cited_ids_valid(["9"], ["1", "2"])
    assert not grounded.cited_ids_valid([], ["1"])  # an answer must cite something


def test_answer_with_fake_model(monkeypatch):
    fake = '{"verdict": "insufficient", "explanation": "", "citations": []}'
    monkeypatch.setattr(grounded, "generate", lambda prompt: fake)
    assert grounded.answer("claim", [("1", "T", "text")])["verdict"] == "insufficient"