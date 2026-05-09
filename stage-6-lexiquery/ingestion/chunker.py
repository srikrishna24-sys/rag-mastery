"""
chunker.py — splits documents into overlapping chunks

WHY THIS EXISTS:
Documents are too long to embed or fit in a prompt.
We split them into smaller pieces with overlap so:
  - Each chunk is semantically meaningful
  - Boundary sentences are captured by adjacent chunks
  - Embeddings represent focused concepts not entire docs

STRATEGY: Recursive character splitting
  Try to split on: paragraphs → sentences → words → characters
  This respects natural text boundaries unlike fixed-size splitting

INNOVATION OPPORTUNITIES:
  - Replace with LlamaParse for layout-aware PDF parsing (tables, headers)
  - Use semantic splitting: embed sentences, split where similarity drops
  - Add document metadata (source, page number, section) to each chunk
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import tiktoken

# ── Constants ──────────────────────────────────────────────────────────────
# 512 tokens: sweet spot for dense technical text
# Too small (<128): chunks lack context
# Too large (>1024): embeddings average too much meaning, retrieval suffers
CHUNK_SIZE = 512

# 50 tokens overlap: captures sentences at chunk boundaries
# Rule of thumb: overlap = ~10% of chunk size
CHUNK_OVERLAP = 50

# We count tokens not characters because:
# - LLMs think in tokens not characters
# - "ChatGPT" is 1 token, "a" is 1 token — very different character lengths
# - Token-based splitting gives consistent semantic density per chunk
ENCODING = "cl100k_base"  # OpenAI's encoding used by GPT-4, text-embedding-3


def count_tokens(text: str) -> int:
    """
    Count how many tokens a string contains.

    WHY: we want chunks of consistent TOKEN size, not character size.
    A 512-character chunk might be 100 tokens or 400 tokens depending
    on the words used. Token counting gives us consistency.
    """
    enc = tiktoken.get_encoding(ENCODING)
    return len(enc.encode(text))


def chunk_document(
    text: str,
    source: str = "unknown",
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """
    Split a document into overlapping chunks.

    Args:
        text: raw document text
        source: filename or URL — stored as metadata on each chunk
        chunk_size: target size in tokens
        chunk_overlap: overlap between adjacent chunks in tokens

    Returns:
        List of LangChain Document objects, each with:
          - page_content: the chunk text
          - metadata: source, chunk_index, token_count

    INNOVATION: in production you'd also store:
        - page_number (from PDF parser)
        - section_heading (from document structure)
        - document_date (for time-sensitive retrieval)
    """
    splitter = RecursiveCharacterTextSplitter(
        # Try splitting on these separators in order:
        # 1. Double newline (paragraph boundary) — best split point
        # 2. Single newline (line boundary)
        # 3. Period+space (sentence boundary)
        # 4. Single space (word boundary) — last resort
        # 5. Empty string (character boundary) — absolute last resort
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Use our token counter not character counter
        length_function=count_tokens,
        # Keep the separator attached to the preceding chunk
        # so sentences don't start with orphaned punctuation
        is_separator_regex=False,
    )

    # Split the raw text into chunks
    raw_chunks = splitter.split_text(text)

    # Wrap each chunk in a LangChain Document with metadata
    # Metadata travels with the chunk through the entire pipeline
    # — retrieval, reranking, generation, evaluation
    documents = []
    for i, chunk_text in enumerate(raw_chunks):
        documents.append(Document(
            page_content=chunk_text,
            metadata={
                "source": source,        # where did this chunk come from?
                "chunk_index": i,        # position in the original document
                "token_count": count_tokens(chunk_text),  # actual size
                "total_chunks": len(raw_chunks),  # how many chunks total
            }
        ))

    return documents


def chunk_documents(docs: List[dict]) -> List[Document]:
    """
    Chunk multiple documents at once.

    Args:
        docs: list of {"text": "...", "source": "filename.pdf"}

    Returns:
        Flat list of all chunks across all documents

    WHY FLAT LIST: the vector store doesn't care which document
    a chunk came from — it just stores and retrieves chunks.
    The source metadata on each chunk tells us provenance.
    """
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            text=doc["text"],
            source=doc.get("source", "unknown")
        )
        all_chunks.extend(chunks)
        print(f"  {doc.get('source', 'unknown')}: {len(chunks)} chunks")

    print(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Quick test — run with: python3 ingestion/chunker.py
    sample_text = """
    Retrieval Augmented Generation (RAG) is a technique that combines
    the strengths of retrieval-based and generative models. Instead of
    relying solely on the model's parametric memory, RAG retrieves
    relevant documents from an external corpus and uses them as context.

    The retrieval component uses dense embeddings or sparse methods like
    BM25 to find relevant passages. These passages are then concatenated
    with the query and passed to the generative model as context.

    This approach has several advantages over pure generative models.
    First, it allows the model to access up-to-date information beyond
    its training cutoff. Second, it reduces hallucinations by grounding
    responses in retrieved evidence. Third, it provides verifiable
    citations for generated content.
    """

    chunks = chunk_document(sample_text, source="test_document.txt")

    print(f"Created {len(chunks)} chunks\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"  Tokens: {chunk.metadata['token_count']}")
        print(f"  Source: {chunk.metadata['source']}")
        print(f"  Text preview: {chunk.page_content[:100]}...")
        print()
