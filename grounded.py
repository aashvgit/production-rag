import json
import re

from llm import generate

PROMPT = """You are a careful research assistant. You are given a scientific claim and
numbered passages (paper abstracts). Use ONLY the passages. Do not use outside knowledge.

Decide whether the passages contain evidence that supports or contradicts the claim.
- If they do, answer "supported" or "contradicted", explain in one or two sentences, and
  cite the ids of the passages you used.
- If the passages do not address the claim, answer "insufficient" and cite nothing.

Reply with JSON only, in exactly this form:
{{"verdict": "supported" or "contradicted" or "insufficient", "explanation": "...", "citations": ["id", "id"]}}

Claim: {claim}

Passages:
{passages}
"""


def format_passages(passages):
    """passages: list of (doc_id, title, text)."""
    return "\n\n".join(f"[{doc_id}] {title}. {text}" for doc_id, title, text in passages)


def parse(reply: str):
    """Return the reply as a dict, or None if it is not the JSON we asked for."""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if data.get("verdict") not in {"supported", "contradicted", "insufficient"}:
        return None
    cites = data.get("citations")
    data["citations"] = [str(c) for c in cites] if isinstance(cites, list) else []
    return data


def cited_ids_valid(cited, provided) -> bool:
    """An answer must cite something, and every cited id must be one we provided."""
    return bool(cited) and all(c in provided for c in cited)


def answer(claim, passages):
    reply = generate(PROMPT.format(claim=claim, passages=format_passages(passages)))
    return parse(reply)  # None means the reply could not be parsed