import chromadb
from services.embeddings import get_embedding

client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection(
    name="medical_papers"
)


def add_papers(papers):
    """
    Store papers in ChromaDB.
    """

    for paper in papers:

        text = paper["title"] + "\n\n" + paper["abstract"]

        embedding = get_embedding(text)

        try:
            collection.add(
                ids=[paper["pmid"]],
                documents=[text],
                embeddings=[embedding],
                metadatas=[
                    {
                        "pmid": paper["pmid"],
                        "title": paper["title"],
                        "journal": paper["journal"],
                        "authors": ", ".join(paper["authors"])
                    }
                ]
            )
        except Exception:
            print(f"Paper {paper['pmid']} already exists. Skipping...")

def search_papers(query, n_results=5):

    embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    papers = []

    documents = results["documents"][0]
    metadata = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, distance in zip(
        documents,
        metadata,
        distances
    ):

        papers.append({
            "pmid": meta["pmid"],
            "title": meta["title"],
            "journal": meta["journal"],
            "authors": meta["authors"],
            "content": doc,
            "distance": distance
        })

    return papers