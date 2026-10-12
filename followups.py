import re

GEN_PROMPT = """A user asked this question about a scientific paper:
"{claim}"

Here is the paper's abstract:
{abstract}

Write ONE short follow-up question (at most 12 words) that the user might ask next. Rules:
- It must only make sense given the first question: do NOT name the specific topic, gene,
  drug, disease or method; use words like "it", "that", "this" or "what about ..." instead.
- It must be answerable from the abstract.
Reply with the follow-up question only."""

REWRITE_PROMPT = """Rewrite the user's follow-up as one standalone search query that contains
all the context needed to understand it, using the earlier message. Reply with the query only.

Earlier message: {claim}
Follow-up: {followup}
Standalone query:"""

REWRITE_HISTORY_PROMPT = """Rewrite the user's last message as one standalone search query that
contains all the context needed to understand it. The conversation may have changed topic: use
only the earlier messages that the last message actually refers to. Reply with the query only.

Earlier messages (oldest first):
{history}

Last message: {followup}
Standalone query:"""

STOP = set(
    "a an the of in on at to for from by with and or not is are was were be been what which who "
    "how does do did can it its this that these those about their there than as more most "
    "also any all some".split()
)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOP}


def overlap(claim: str, followup: str) -> int:
    return len(content_words(claim) & content_words(followup))