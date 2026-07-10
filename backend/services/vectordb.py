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

    for i, paper in enumerate(papers):

        text = paper["title"] + "\n\n" + paper["abstract"]

        embedding = get_embedding(text)

        collection.add(
            ids=[str(i)],
            documents=[text],
            embeddings=[embedding],
            metadatas=[
                {
                    "title": paper["title"],
                    "journal": paper["journal"]
                }
            ]
        )

def search_papers(query, n_results=3):

    embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    papers = []

    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    for doc, meta in zip(documents, metadata):

        papers.append({
            "title": meta["title"],
            "journal": meta["journal"],
            "content": doc
        })

    return papers