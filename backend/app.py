from fastapi import FastAPI
from services.embeddings import get_embedding
from services.pubmed import search_pubmed, fetch_papers
from services.vectordb import add_papers, search_papers
from services.rag import answer_question

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Medical AI Assistant is running"}


@app.get("/embedding")
def embedding(text: str):

    vector = get_embedding(text)

    return {
        "dimension": len(vector),
        "first10": vector[:10]
    }

@app.get("/ingest")
def ingest(query: str):

    ids = search_pubmed(query)

    papers = fetch_papers(ids)

    add_papers(papers)

    return {
        "message": f"{len(papers)} papers stored successfully."
    }

@app.get("/semantic-search")
def semantic_search(query: str):

    results = search_papers(query)

    return results

@app.get("/chat")
def chat(question: str):

    return answer_question(question)