# Measure retrieval quality."""Implement relevance scoring: does retrieved context contain answer clues?Simple approach: keyword overlap between query and contextOr use semantic similarity (TF-IDF overlap)Score each retrieval: 0 (irrelevant) to 1 (highly relevant)Generate evaluation report for all test queries"""# Input: Query, retrieved chunks# Output: Relevance score (0-1), report
from pathlib import Path  # For loading test_queries.txt fileimport re  # For regex tokenization (extracting words from text)from statistics import mean  # For computing average metrics in report
from sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.metrics.pairwise import cosine_similarity
from task3_retrieval import retrieve_top_k

# Quick concept notes:# TF-IDF: gives higher weight to informative words and lower weight to very common words.# Cosine similarity: compares direction of vectors (semantic closeness), independent of length.# Jaccard overlap: lexical overlap ratio = shared terms / total unique terms.

# Common words removed so overlap focuses on meaningful terms.STOPWORDS = {    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",    "with", "what", "how", "why", "when", "where", "which", "who", "whom", "this",}

def tokenize(text: str) -> set[str]:    # Lowercase + simple alphanumeric tokenization.    # re.findall(r"[a-zA-Z0-9]+", ...) extracts all letter/number groups and ignores punctuation.    # Example: "RAG-2026 is great!" -> ["rag", "2026", "is", "great"] after lowercasing.    words = re.findall(r"[a-zA-Z0-9]+", text.lower())    # Drop stopwords and one-character tokens.    return {w for w in words if w not in STOPWORDS and len(w) > 1}

def keyword_overlap_score(query: str, retrieved_chunks: list[dict]) -> float:    """Jaccard-style overlap between query keywords and retrieved context keywords."""    if not retrieved_chunks:        return 0.0
    # Compare query terms against all retrieved text merged together.    query_tokens = tokenize(query)    context_text = " ".join(chunk.get("text", "") for chunk in retrieved_chunks)    context_tokens = tokenize(context_text)
    if not query_tokens or not context_tokens:        return 0.0
    # Jaccard overlap = shared terms / total unique terms.    intersection = query_tokens.intersection(context_tokens)    union = query_tokens.union(context_tokens)    return len(intersection) / len(union)

def tfidf_semantic_score(query: str, retrieved_chunks: list[dict]) -> float:    """Compute average TF-IDF cosine similarity between query and retrieved chunks."""    if not retrieved_chunks:        return 0.0
    # First document is query, remaining are retrieved chunk texts.    docs = [query] + [chunk.get("text", "") for chunk in retrieved_chunks]    # Build TF-IDF vectors from this small corpus.    # Each text becomes a vector where important terms get higher weights.    vectorizer = TfidfVectorizer()    tfidf = vectorizer.fit_transform(docs)
    query_vec = tfidf[0:1]    chunk_vecs = tfidf[1:]    # Similarity of query against each chunk.    # Higher cosine score means chunk text is closer to query intent.    sims = cosine_similarity(query_vec, chunk_vecs)[0]
    # Use average similarity as a single semantic relevance signal.    return float(sims.mean()) if len(sims) else 0.0

def combined_relevance_score(query: str, retrieved_chunks: list[dict], alpha: float = 0.5) -> dict:    """Blend keyword and TF-IDF scores into one 0-1 relevance score."""    keyword = keyword_overlap_score(query, retrieved_chunks)    tfidf = tfidf_semantic_score(query, retrieved_chunks)    # Weighted blend between lexical and semantic signals.    # Purpose: combine exact term matching (keyword overlap) with meaning similarity (TF-IDF cosine)    # so the final score is more robust than using either signal alone.    # Formula: R = alpha*K + (1-alpha)*S    # where R=final relevance, K=keyword overlap score, S=tfidf semantic score.    # alpha closer to 1.0 -> trust keyword overlap more.    # alpha closer to 0.0 -> trust semantic similarity more.    combined = alpha * keyword + (1 - alpha) * tfidf    return {        "keyword_overlap": round(keyword, 4),        "tfidf_similarity": round(tfidf, 4),        "relevance": round(combined, 4),    }

def load_test_queries() -> list[str]:    """Load queries from test_queries.txt if present, otherwise fallback defaults."""    query_file = Path(__file__).parent / "test_queries.txt"    if query_file.exists():        # Keep non-empty lines only.        lines = [line.strip() for line in query_file.read_text(encoding="utf-8").splitlines()]        queries = [q for q in lines if q]        if queries:            return queries
    # Fallback examples so script still runs without external query file.    return [        "What is RAG?",        "How do neural networks work?",        "What is machine learning?",    ]

def evaluate_queries(queries: list[str], k: int = 5) -> list[dict]:    report_rows = []    for query in queries:        # Retrieve chunks from Task 3 and score this retrieval set.        retrieved = retrieve_top_k(query, k=k)        scores = combined_relevance_score(query, retrieved)
        # Store one row per query for final reporting.        report_rows.append(            {                "query": query,                "retrieved_count": len(retrieved),                "top_source": retrieved[0]["source"] if retrieved else "none",                "top_score": round(retrieved[0]["score"], 4) if retrieved else 0.0,                **scores,            }        )    return report_rows

def print_report(rows: list[dict]) -> None:    print("=" * 88)    print("Task 6 Retrieval Evaluation Report")    print("=" * 88)
    for i, row in enumerate(rows, start=1):        # Per-query detailed metrics.        print(f"\n{i}. Query: {row['query']}")        print(f"   Retrieved chunks : {row['retrieved_count']}")        print(f"   Top source       : {row['top_source']}")        print(f"   Top retrieval    : {row['top_score']}")        print(f"   Keyword overlap  : {row['keyword_overlap']}")        print(f"   TF-IDF similarity: {row['tfidf_similarity']}")        print(f"   Final relevance  : {row['relevance']}")
    if rows:        # Aggregate metrics across all evaluated queries.        avg_keyword = mean(row["keyword_overlap"] for row in rows)        avg_tfidf = mean(row["tfidf_similarity"] for row in rows)        avg_rel = mean(row["relevance"] for row in rows)
        print("\n" + "-" * 88)        print("Summary")        print(f"Average keyword overlap  : {avg_keyword:.4f}")        print(f"Average TF-IDF similarity: {avg_tfidf:.4f}")        print(f"Average relevance score  : {avg_rel:.4f}")

if __name__ == "__main__":    # Run full evaluation pipeline.    queries = load_test_queries()    report = evaluate_queries(queries, k=5)    print_report(report)
