# production-rag

A retrieval-augmented system for checking scientific claims against paper abstracts
(BEIR SciFact: 5,183 abstracts, 300 labeled test claims), built and measured step by step:
hybrid retrieval, reranking, chunking, query rewriting, grounded answers with a
refusal path, and incremental reindexing.

## Retrieval (300 test claims; 95% ranges by bootstrap over queries)

| System | nDCG@10 | Recall@10 | Time per query |
|---|---|---|---|
| BM25 | 0.652 | 0.776 | [fill] |
| Dense (bge-small-en-v1.5) | 0.721 | 0.840 | 11 ms |
| Hybrid (reciprocal rank fusion, BM25 weight 0.25 tuned on 809 train claims) | 0.719 | 0.843 | [fill] |
| Hybrid + MiniLM cross-encoder reranker | 0.696 | 0.819 | 425 ms |
| Hybrid + bge-reranker-base | 0.720 | 0.849 | 3,103 ms |

Findings: dense beat BM25 by 0.069 (paired range 0.032 to 0.106). Hybrid did not
beat dense (difference -0.003, range -0.020 to +0.015), and neither of two rerankers
helped while costing 39 to 280 times more time per query. The reranker is therefore off by default.
Hypothesis (untested): bge-small already handles most claim-to-abstract paraphrases.

## Chunking (nDCG@10, dense; variant chosen on 809 train claims, then reported on test)

| Variant | Train | Test |
|---|---|---|
| Whole abstract (chosen) | 0.757 | 0.721 |
| 1 sentence | 0.737 | 0.716 |
| 3-sentence windows | 0.756 | 0.729 |
| 5-sentence windows | 0.756 | 0.725 |
| 3-sentence windows, no title | 0.726 | 0.720 |

Chunking made no detectable difference, because SciFact abstracts are short (about 215 words).
I kept whole abstracts after choosing on train and did not switch after seeing test.

## Refusal and grounding

A similarity-score threshold separates answerable from unanswerable claims poorly (AUC 0.68).
One model call returning JSON (verdict, explanation, citations), with "insufficient" as the refusal
path, does much better. "Unanswerable" claims are made by deleting the claim's labeled abstracts
from the index. Prompt frozen on 60 train claims; test run once.

| 100 test claims | Result (95% range) |
|---|---|
| Answerable claims answered | 61% (51 to 70%) |
| Answerable claims refused | 39% |
| Answers whose citations were all real (code-checked) | 61/61 |
| Answers citing a labeled relevant abstract | 50/61 (71 to 90%) |
| Claims with the answer removed that were wrongly answered | 20% (13 to 29%) |

I read [N] of the 20 wrongly answered cases: [X] were defensible (the cited abstract addresses the
claim despite the missing label) and [Y] were unsupported. The pipeline also withholds any answer
whose citations were not among the retrieved passages. It never triggered here (0 of 99 answers).

## Conversational follow-ups (54 test claims, model-written follow-ups)

| nDCG@10 | Normal | Unrelated earlier message |
|---|---|---|
| Raw follow-up alone | 0.236 | |
| Claim + follow-up (concat) | 0.776 | 0.570 |
| Model rewrite | 0.773 | 0.738 |

Rewriting beat concatenation by 0.169 (range 0.092 to 0.246) when history contained a topic
change, at a median of 1.18 s per rewrite call. Caveat: the follow-ups were generated from the
relevant abstract, which leaks vocabulary (concat scored above the original claim), so absolute
numbers are inflated and only the comparisons between methods are informative.

## Incremental reindexing

Documents are hashed and only new or changed ones are re-embedded; the index stores a config string,
so a model change forces a full rebuild. On SciFact: first build 64.8 s (5,183 documents),
no changes 0.0 s, and 5 edited + 1 deleted + 1 added documents 0.4 s (6 embedded).
A test proves the result equals a full rebuild. BM25 is still rebuilt from scratch.

## Limitations

- Single corpus (short biomedical abstracts); chunking and reranking results may differ elsewhere.
- LLM-based results use 60 to 100 claims, so ranges are wide; model is an API alias that may change.
- Free-text answer quality is not scored (no gold answers), only retrieval, citation validity and refusal.
- 39% of answerable claims were refused; I did not separate retrieval misses from over-refusals on test.
- BEIR labels are incomplete, so some "wrong" answers may be defensible.

## Tests

`uv run pytest -v` runs 20 tests with no model and no API key (metrics, parsing and citation checks,
incremental index equivalence, pipeline logic with fakes).

## Reproduce

[numbered commands]

Data: BEIR SciFact, not included in the repository.