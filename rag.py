from dataclasses import dataclass

from followups import REWRITE_HISTORY_PROMPT
from grounded import cited_ids_valid
from llm import generate


@dataclass
class Answer:
    verdict: str          # "supported", "contradicted", "insufficient" or "error"
    explanation: str
    citations: list
    query: str            # the standalone query that was actually searched
    passages: list        # (doc_id, title, text) shown to the model


def rewrite(history, message):
    lines = "\n".join(f"{i}. {m}" for i, m in enumerate(history, start=1))
    reply = generate(REWRITE_HISTORY_PROMPT.format(history=lines, followup=message))
    return reply.strip().strip('"')


class RAG:
    def __init__(self, index, embed_query, docs, answer_fn, rewrite_fn=None,
                 top_k=5, max_history=4):
        self.index, self.embed_query, self.docs = index, embed_query, docs
        self.answer_fn, self.rewrite_fn = answer_fn, rewrite_fn
        self.top_k, self.max_history = top_k, max_history

    def ask(self, message, history=()):
        history = list(history)[-self.max_history:]
        query = self.rewrite_fn(history, message) if history and self.rewrite_fn else message
        ids = self.index.search(self.embed_query(query), self.top_k)
        passages = [(i, *self.docs[i]) for i in ids]
        result = self.answer_fn(query, passages)
        if result is None:
            return Answer("error", "The model reply could not be parsed.", [], query, passages)
        if result["verdict"] != "insufficient" and not cited_ids_valid(result["citations"], ids):
            # grounding gate: an answer with missing or invented citations is withheld
            return Answer("insufficient", "The answer's citations could not be verified, "
                          "so it was withheld.", [], query, passages)
        return Answer(result["verdict"], result.get("explanation", ""),
                      result["citations"], query, passages)