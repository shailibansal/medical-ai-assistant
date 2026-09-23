from sentence_transformers import CrossEncoder


# Cross-encoder used for query-document relevance scoring
model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_documents(query, papers, top_k=5):
    """
    Rerank retrieved papers based on query-document relevance.

    Args:
        query: User's question
        papers: Candidate papers retrieved from ChromaDB
        top_k: Number of papers to return

    Returns:
        Reranked list of papers
    """

    if not papers:
        return []

    pairs = []

    for paper in papers:
        document = f"{paper['title']}\n\n{paper['content']}"
        pairs.append(
            (
                query,
                document
            )
        )

    scores = model.predict(pairs)

    for paper, score in zip(papers, scores):
        paper["rerank_score"] = float(score)

    papers = sorted(
        papers,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return papers[:top_k]