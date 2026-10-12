from data import load_scifact
from eval_lib import evaluate
from retrieval import Retriever

corpus, queries, qrels = load_scifact(split="test")
ret = Retriever(corpus)
evaluate("hybrid", lambda q: ret.hybrid(q, w_bm25=0.25), queries, qrels)