

# Create prompts that combine query + retrieved context.
"""
Build a prompt template that includes:
System instruction (role of assistant)
Retrieved context chunks
User query
Output format instruction
Return formatted prompt string ready for LLM
"""
# Input: Query, retrieved chunks
# Output: Formatted prompt string

def build_prompt(
        query: str,
        retrieved_chunks: list[dict]
) -> str:
        # Hint 1: Each item in retrieved_chunks is expected to have keys like
        # text, source, score (score is optional).
        if not retrieved_chunks:
                formatted_context = "No relevant context retrieved."
        else:
                context_lines = []
                for i, chunk in enumerate(retrieved_chunks, start=1):
                        source = chunk.get("source", "unknown")
                        score = chunk.get("score", None)
                        text = chunk.get("text", "")

                        # Hint 2: You can change this formatting to match your preferred
                        # citation style for Task 5.
                        if score is None:
                                context_lines.append(f"[{i}] source={source}\n{text}")
                        else:
                                context_lines.append(f"[{i}] source={source} score={score:.4f}\n{text}")

                formatted_context = "\n\n".join(context_lines)

        # Hint 3: Keep instructions explicit to reduce hallucinations.
#we join the li
        prompt = f"""You are a helpful AI assistant.
Answer ONLY using the context provided below.
If the answer is not in the context, say: 'I could not find that in the provided context.'

Context:
{formatted_context}

User Question:
{query}

Output format:
1) Direct answer
2) Supporting evidence from context
3) Source references like [1], [2]
"""

        return prompt


if __name__ == "__main__":
        # Tiny local test
        sample_query = "What is RAG?"
        sample_chunks = [
                {
                        "source": "doc3rag.txt",
                        "score": 0.91,
                        "text": "RAG combines retrieval and generation for better answers."
                },
                {
                        "source": "doc1ml.txt",
                        "score": 0.34,
                        "text": "Machine learning is a subset of artificial intelligence."
                },
        ]

        print(build_prompt(sample_query, sample_chunks))
    


