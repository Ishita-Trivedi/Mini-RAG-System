

"""
Implement cosine similarity search (use sklearn.metrics.pairwise.cosine_similarity)
Or use Faiss for production efficiency: pip install faiss-cpu
Return top-k (k=5) most similar chunks ranked by score
Include similarity scores in results
"""
import numpy as np
import faiss
from task2_embeddings import embeddings
from task1_loader import all_chunks
from sentence_transformers import SentenceTransformer

#d represents dimension
# WRONG (kept for learning): d = embeddings.shape
# Reason: this returns a tuple (num_chunks, dim). FAISS expects a single int dimension.
def retrieve_top_k(query: str, k: int = 5) -> list[dict]:
    d = embeddings.shape[1]

    #cast to float32: FAISS expects vectors in 32-bit float format (float32) for indexing and search.
    emb_matrix = embeddings.astype("float32")
    #normalize vectors
    faiss.normalize_L2(emb_matrix)
    #build the index

    # WRONG (kept for learning): index = faiss.IndexFlatL2(d)
    # Reason: with L2-normalized vectors, IndexFlatIP is preferred for cosine-style similarity search.
    index = faiss.IndexFlatIP(d)

    #store vectors as per index
    index.add(emb_matrix)


    #search a sentence in this db aka vector search
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vec = model.encode([query]).astype("float32")

    #normalize the vectors
    # You are likely using IndexFlatIP (inner product).
    # For unit-normalized vectors, inner product equals cosine similarity:
    # So normalization lets FAISS inner-product search behave like cosine search.
    faiss.normalize_L2(query_vec)
    #search for nearest neighbors

    # WRONG (kept for learning): k = 5
    # Reason: if available chunks are fewer than k, FAISS pads with invalid neighbors (-1) and extreme scores.
    #shape represent the number of chunks so retrieved k should be within the limits
    k = min(k, emb_matrix.shape[0])
    scores, indices = index.search(query_vec, k=k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        # Safety check: skip invalid FAISS placeholders if any appear.
        if idx < 0 or idx >= len(all_chunks):
            continue

        chunk = all_chunks[idx]
        results.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "source": chunk.get("source"),
                "position": chunk.get("position"),
                "text": chunk.get("text"),
                "score": float(score),
            }
        )

    return results


if __name__ == "__main__":
    out = retrieve_top_k("What is RAG?", k=5)
    for i, item in enumerate(out, start=1):
        print(f"{i}. score={item['score']:.4f} source={item['source']}")
        print(f"   {item['text']}")


