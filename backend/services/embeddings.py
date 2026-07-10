from sentence_transformers import SentenceTransformer

# Load the model once when the app starts
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    """
    Convert text into an embedding vector.
    """
    embedding = model.encode(text)

    return embedding.tolist()