from sentence_transformers import SentenceTransformer

from data import doc_text, load_scifact
from grounded import answer
from index import IncrementalIndex
from rag import RAG, rewrite

MODEL = "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(MODEL)
corpus, _, _ = load_scifact()

index = IncrementalIndex(
    "data/index",
    lambda texts: model.encode(texts, batch_size=32, normalize_embeddings=True),
    config=MODEL,
)
print("syncing index:", index.sync({i: doc_text(d) for i, d in corpus.items()}))
rag = RAG(
    index,
    lambda t: model.encode([t], normalize_embeddings=True)[0],
    {i: (d["title"], d["text"]) for i, d in corpus.items()},
    answer,
    rewrite,
)

history = []
print("Type a scientific claim. An empty line quits.")
while True:
    message = input("\nYou: ").strip()
    if not message:
        break
    result = rag.ask(message, history)
    history.append(message)
    print(f"[searched: {result.query}]")
    print(f"{result.verdict.upper()}: {result.explanation}")
    for doc_id, title, _ in result.passages:
        if doc_id in result.citations:
            print(f"  cited [{doc_id}] {title}")