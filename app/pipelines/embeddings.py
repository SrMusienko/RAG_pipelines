from sentence_transformers import SentenceTransformer
import pickle

def compute_embeddings(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2"
) -> list[dict]:

    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = vectors[i].tolist()

    with open("artifacts/embeddings.pkl", "wb") as f:
        pickle.dump(vectors, f)

    print(f"Сгенерировано {len(vectors)} эмбеддингов, размерность {vectors.shape[1]}")

    return chunks