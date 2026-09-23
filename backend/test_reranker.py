from services.vectordb import search_papers
from services.reranker import rerank_documents


query = "What are the molecular mechanisms involved in diabetic retinopathy?"


print("=" * 70)
print("INITIAL VECTOR RETRIEVAL")
print("=" * 70)

papers = search_papers(
    query,
    n_results=10
)

for i, paper in enumerate(papers, start=1):
    print(
        f"{i}. {paper['pmid']} | "
        f"distance={paper['distance']:.4f}"
    )


print("\n")
print("=" * 70)
print("AFTER RERANKING")
print("=" * 70)

reranked = rerank_documents(
    query,
    papers,
    top_k=5
)

for i, paper in enumerate(reranked, start=1):
    print(
        f"{i}. {paper['pmid']} | "
        f"rerank_score={paper['rerank_score']:.4f}"
    )