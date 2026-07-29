# Mini RAG System - Hands-On 

A progressive implementation of a Retrieval-Augmented Generation (RAG) system from scratch.

## Overview

This project walks you through building a complete RAG pipeline in 6 tasks, from document loading to end-to-end question answering with evaluation.

## Project Structure

```text
Test1/
├── documents/              # Sample .txt documents to retrieve from
├── task1_loader.py         # Load and chunk documents
├── task2_embeddings.py     # Generate embeddings for chunks
├── task3_retrieval.py      # Implement vector search
├── task4_prompt.py         # Build prompts from retrieved context
├── task5_rag.py            # End-to-end RAG pipeline
├── task6_eval.py           # Evaluate retrieval quality
└── test_queries.txt        # Sample queries to test with
```

## Tasks

### **Task 1: Document Loader & Chunker** (task1_loader.py)
Load text documents and split them into overlapping chunks.

**Quick library list (Task 1):**
- `langchain-text-splitters` (`RecursiveCharacterTextSplitter`)
- `nltk`
- `tiktoken`
- plus stdlib (`pathlib`, string methods)

**Libraries/Tools to use:**
- `pathlib` or `os` for reading files from `documents/`
- For chunking (pick one):
  - `langchain-text-splitters` -> `RecursiveCharacterTextSplitter` (recommended for beginners)
  - `nltk` tokenization (`word_tokenize`) + custom overlap logic
  - `tiktoken` for token-aware chunk sizes close to LLM tokens
- Core Python string methods (`split`, slicing, `join`) for a no-library chunking baseline
- Optional: `re` for basic text cleanup
- Optional: `typing` (`List`, `Dict`) for readable type hints

**Install for this task:**
- Minimum: no install required (standard library only)
- Recommended chunking lib: `pip install langchain-text-splitters`
- Alternatives: `pip install nltk` or `pip install tiktoken`

**Objectives:**
- Read text files from `documents/` folder
- Split into chunks of ~256 tokens with 50-token overlap
- Return list of chunk objects with metadata (chunk_id, source_file, position)

**Topics learned:**
- File handling with `pathlib`
- Basic text chunking and overlap logic
- Designing chunk metadata for later retrieval

**Go deeper on your own:**
- Compare character-based, word-based, and token-based chunking
- Study `RecursiveCharacterTextSplitter` internals and separator hierarchy
- Explore why overlap improves retrieval quality

**Input:** Text files  
**Output:** `List[Dict]` with keys: `chunk_id`, `source`, `position`, `text`

---

### **Task 2: Embedding Pipeline** (task2_embeddings.py)
Convert chunks into numerical embeddings using a pre-trained model.

**Libraries/Tools to use:**
- `sentence_transformers` (`SentenceTransformer`) to create embeddings
- `numpy` to store/save embedding arrays
- Optional: `pickle` or `json` for saving chunk metadata alongside embeddings

**Install for this task:**
- `pip install sentence-transformers numpy`

**Objectives:**
- Use `sentence-transformers` library (install: `pip install sentence-transformers`)
- Load model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings)
- Generate embeddings for all chunks from Task 1
- Store as numpy array for efficient retrieval

**Topics learned:**
- What embeddings represent
- Batch encoding with a pretrained model
- Saving vectors for fast reuse

**Go deeper on your own:**
- Learn cosine space and vector similarity intuition
- Compare semantic embeddings with TF-IDF vectors
- Study embedding dimensionality and model trade-offs

**Input:** List of chunks  
**Output:** `np.ndarray` of shape (num_chunks, 384)

---

### **Task 3: Vector Search** (task3_retrieval.py)
Build a simple vector similarity search engine.

**Libraries/Tools to use:**
- `sklearn.metrics.pairwise.cosine_similarity` for baseline similarity search
- `numpy` for vector operations and top-k sorting
- Optional (faster/production style): `faiss-cpu` for indexing and retrieval

**Install for this task:**
- Baseline retrieval: `pip install scikit-learn numpy`
- Optional fast index: `pip install faiss-cpu`

**Objectives:**
- Implement cosine similarity search (use `sklearn.metrics.pairwise.cosine_similarity`)
- Or use Faiss for production efficiency: `pip install faiss-cpu`
- Return top-k (k=5) most similar chunks ranked by score
- Include similarity scores in results

**Topics learned:**
- Similarity scoring for retrieval
- Top-k ranking and sorting
- Baseline vs indexed retrieval

**Go deeper on your own:**
- Compare cosine similarity with dot product and Euclidean distance
- Learn how FAISS indexes speed up nearest-neighbor search
- Study hybrid retrieval and reranking strategies

**Input:** Query string, embeddings, chunks  
**Output:** `List[Dict]` with top-5 chunks and similarity scores

---

### **Task 4: Prompt Builder** (task4_prompt.py)
Create prompts that combine query + retrieved context.

**Libraries/Tools to use:**
- Core Python string formatting (`f-strings`) for prompt templates
- Optional: `textwrap` for cleaner multi-line formatting
- Optional: `jinja2` if you want reusable prompt templates (not required)

**Install for this task:**
- Minimum: no install required (standard library only)
- Optional template engine: `pip install jinja2`

**Objectives:**
- Build a prompt template that includes:
  - System instruction (role of assistant)
  - Retrieved context chunks
  - User query
  - Output format instruction
- Return formatted prompt string ready for LLM

**Topics learned:**
- Prompt structure and instruction design
- Context injection
- Formatting inputs for generation models

**Go deeper on your own:**
- Study prompt constraints, citations, and grounding
- Compare few-shot, zero-shot, and retrieval-augmented prompts
- Explore prompt templates for different answer styles

**Input:** Query, retrieved chunks  
**Output:** Formatted prompt string

**Example template:**
```
You are a helpful assistant. Answer the user's question based ONLY on the provided context.

Context:
{context_here}

Question: {query}

Answer:
```

---

### **Task 5: End-to-End RAG** (task5_rag.py)
Integrate all tasks into one working pipeline.

**Libraries/Tools to use:**
- Your own modules from Tasks 1-4 (`task1_loader`, `task2_embeddings`, etc.)
- One LLM option:
  - `transformers` (Hugging Face local pipeline) or
  - `requests` (for API call style) or
  - `ollama` CLI/local runtime (if installed)
- `time` to measure basic latency per stage

**Install for this task:**
- Mock LLM path: no extra install required
- Hugging Face local path: `pip install transformers torch`
- API call path: `pip install requests`
- Ollama path: install Ollama app locally and run a model (for example `ollama run llama3`)

**Objectives:**
- Chain Tasks 1-4 into a single `rag(query: str) -> str` function
- Implement a simple LLM call (options):
  - **Mock LLM:** Return a deterministic answer based on context
  - **Hugging Face API:** Use free inference with `transformers` library
  - **Local Ollama:** If running locally on your machine
- Test with 3-5 sample queries from `test_queries.txt`

**Topics learned:**
- End-to-end pipeline orchestration
- Module composition and data flow
- Basic latency awareness

**Go deeper on your own:**
- Study pipeline error handling and fallbacks
- Compare mock, local, and hosted model execution
- Learn how to cache embeddings and retrieval results

**Input:** User query  
**Output:** Generated answer grounded in retrieved context

---

### **Task 6: Evaluation Metric** (task6_eval.py)
Measure retrieval quality.

**Libraries/Tools to use:**
- `collections.Counter` or set logic for keyword overlap scoring
- `sklearn.feature_extraction.text.TfidfVectorizer` for TF-IDF similarity
- `sklearn.metrics.pairwise.cosine_similarity` for semantic-style score
- `pandas` (optional) to generate a clean evaluation report table

**Install for this task:**
- Baseline evaluation: `pip install scikit-learn`
- Optional reporting table: `pip install pandas`

**Objectives:**
- Implement relevance scoring: does retrieved context contain answer clues?
- Simple approach: keyword overlap between query and context
- Or use semantic similarity (TF-IDF overlap)
- Score each retrieval: 0 (irrelevant) to 1 (highly relevant)
- Generate evaluation report for all test queries

**Topics learned:**
- Retrieval quality scoring
- Simple evaluation metrics
- Reporting and comparison of results

**Go deeper on your own:**
- Learn precision, recall, MRR, and nDCG for retrieval systems
- Study how to build a small gold-label evaluation set
- Compare keyword overlap with semantic evaluation

**Input:** Query, retrieved chunks  
**Output:** Relevance score (0-1), report

---

## Sample Documents

Create a `documents/` folder with 5-10 sample .txt files. Examples:

**doc1.txt** (Machine Learning 101)
```
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
Three main types exist: supervised, unsupervised, and reinforcement learning.
Supervised learning uses labeled data to train models for prediction tasks.
```

**doc2.txt** (Neural Networks)
```
Neural networks are inspired by biological neurons in the human brain.
They consist of interconnected layers: input, hidden, and output layers.
Backpropagation is used to update weights during training.
```

**doc3.txt** (RAG Systems)
```
RAG combines retrieval and generation for better answers.
First, relevant documents are retrieved based on the query.
Then, these documents augment the prompt to the language model.
This approach reduces hallucination in AI responses.
```

---

## Getting Started

### 1. Install Dependencies
```bash
cd Test1
pip install sentence-transformers faiss-cpu scikit-learn numpy
```

### 2. Create Sample Documents
Create `documents/` folder and add the sample .txt files above.

### 3. Run Tasks Sequentially
```bash
python task1_loader.py        # Load and chunk documents
python task2_embeddings.py    # Generate embeddings
python task3_retrieval.py     # Test vector search
python task4_prompt.py        # Build prompts
python task5_rag.py           # Run end-to-end RAG
python task6_eval.py          # Evaluate results
```

### 4. Test Queries
Create `test_queries.txt` with sample questions:
```
What is machine learning?
How do neural networks work?
What is RAG and why is it useful?
Explain backpropagation.
```

---

## How To Approach This For Maximum Learning

Use this workflow while implementing each task:

1. **Predict first (2-3 min):** Before coding, write what you expect the output to look like.
2. **Implement small:** Build the smallest working version (no optimization yet).
3. **Inspect outputs deeply:** Print intermediate values:
  - chunk counts and sample chunks
  - embedding shape and sample vector stats
  - top-k retrieval results with scores
  - final prompt sent to the LLM
4. **Break it on purpose:** Try edge cases (empty query, unrelated query, tiny document, repeated text).
5. **Fix with reason:** For each bug, write one line: root cause + fix.
6. **Upgrade once baseline works:**
  - cosine -> hybrid retrieval
  - fixed prompt -> better instruction + citation format
  - single metric -> add latency and hit-rate
7. **Reflect after each task (5 min):**
  - What did I build?
  - Why does it work?
  - Where can it fail in production?

### Practice Mode (recommended)

- **Round 1 (learning mode):** Use print statements and keep code simple.
- **Round 2 (engineering mode):** Refactor into reusable functions and add type hints.
- **Round 3 (interview mode):** Rebuild Task 1-3 from memory without looking.

### What to track in a notebook

- Query -> top-5 chunks -> answer quality
- Retrieval mistakes (wrong chunk at rank 1)
- Prompt changes and their effect on answer quality
- Average latency per stage: chunking, embedding, retrieval, generation

If you follow this cycle, you will not only complete the project but also understand the reasoning behind each RAG design choice.

---

## Success Criteria

✅ **Task 1:** Successfully load 5+ documents and chunk them with metadata  
✅ **Task 2:** Generate 384-dim embeddings for all chunks  
✅ **Task 3:** Retrieve top-5 relevant chunks for sample queries  
✅ **Task 4:** Generate well-formatted prompts combining context + query  
✅ **Task 5:** End-to-end pipeline produces coherent answers  
✅ **Task 6:** Relevance scores correlate with actual context quality  

---

## Tips & Hints

- **Tokenization:** Use `len(text.split())` as a rough token count (1 token ≈ 1 word)
- **Embeddings:** Pre-computed embeddings should be cached in a `.npy` file for speed
- **Similarity:** Cosine similarity is standard: `(A · B) / (||A|| × ||B||)`
- **Mock LLM:** Start with a template-based answer to focus on retrieval logic first
- **Debugging:** Print retrieved chunks at each stage to verify correctness

---

## Expected Output

After running all tasks:
```
Query: "What is RAG?"

Retrieved Chunks:
1. [similarity: 0.89] "RAG combines retrieval and generation..."
2. [similarity: 0.76] "First, relevant documents are retrieved..."
3. [similarity: 0.71] "This approach reduces hallucination..."

Generated Answer:
"RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and 
generation for better answers. It works by first retrieving relevant documents based 
on the query, then augmenting the prompt to the language model with these documents. 
This approach reduces hallucination in AI responses."

Relevance Score: 0.85
```

---

## Time Estimate

- Task 1: 30-45 min
- Task 2: 20-30 min
- Task 3: 30-45 min
- Task 4: 15-20 min
- Task 5: 45-60 min
- Task 6: 20-30 min

**Total: 3-4 hours**

---

## Resources

- [Sentence Transformers Docs](https://www.sbert.net/)
- [Faiss Documentation](https://github.com/facebookresearch/faiss)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)

---

**Good luck! Start with Task 1 and work sequentially. Each task builds on the previous one.**
