

# Load and chunk documents
"""
Task 1: Document Loader & Chunker (task1_loader.py)
Load text documents and split them into overlapping chunks.

Libraries/Tools to use:
- pathlib (or os) for reading files from documents/
- For chunking (pick one):
    - langchain-text-splitters (RecursiveCharacterTextSplitter)
    - nltk
    - tiktoken
- Stdlib fallback: Python string methods (split, slicing, join)

Objectives:

Read text files from documents/ folder
Split into chunks of ~256 tokens with 50-token overlap
Return list of chunk objects with metadata (chunk_id, source_file, position)
Input: Text files
Output: List[Dict] with keys: chunk_id, source, position, text
"""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Read text files from documents/ folder
# folder_path=Path("./documents")
folder_path=Path(__file__).parent/"documents"

#initialize text splitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50,
    length_function=len,
    add_start_index=True  # Automatically adds 'start_index' to chunk metadata
)
all_chunks=[]
for file_path in folder_path.glob("*.txt"):
    text=file_path.read_text(encoding="utf-8")
    #Split into chunks of ~256 tokens with 50-token overlap
    texts=text_splitter.split_text(text)
    for i,chunk_text in enumerate(texts):
        all_chunks.append({
            "chunk_id":f"{file_path.stem}_{i}",
            "source": file_path.name,
            "position":i,
            "text":chunk_text
        })
print(f"Total chunks:{len(all_chunks)}")
print(all_chunks[0])






