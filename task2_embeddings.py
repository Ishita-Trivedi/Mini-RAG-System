

"""
Use sentence-transformers library (install: pip install sentence-transformers)
Load model: sentence-transformers/all-MiniLM-L6-v2 (384-dim embeddings)
Generate embeddings for all chunks from Task 1
Store as numpy array for efficient retrieval

Input: List of chunks
Output: np.ndarray of shape (num_chunks, 384)
"""
from task1_loader import all_chunks
from sentence_transformers import SentenceTransformer
import numpy as np
# Initialize the model (auto-downloads)
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#extract text from all_chunks
chunk_texts=[c["text"] for c in all_chunks]
#compute embeddings
embeddings=model.encode(chunk_texts)

# Store as numpy array for efficient retrieval
embeddings=np.array(embeddings)
print(embeddings)
print(embeddings.shape)


