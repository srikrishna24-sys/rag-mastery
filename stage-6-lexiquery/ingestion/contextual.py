"""
contextual.py — adds LLM-generated context to each chunk

WHY THIS EXISTS:
Chunks extracted from documents are decontextualised fragments.
A chunk saying "the ratio shall not fall below the threshold"
has no idea it came from an RBI circular about NBFC liquidity.

This file fixes that by prepending a 1-2 sentence LLM-generated
summary to each chunk BEFORE embedding. The summary situates the
chunk within its source document — section, topic, identifiers.

RESULT: embeddings are rich with context, not decontextualised.
Anthropic reported 49% reduction in retrieval failures with this.

COST: one LLM call per chunk at ingestion time only.
      With gpt-4o-mini: ~$0.05 for 200 chunks total.

INNOVATION OPPORTUNITIES:
  - Use prompt caching (Anthropic API) to cache the full document
    and only pay for the chunk tokens per call — cuts cost 90%
  - Run chunk context generation in parallel with asyncio
  - Store context separately and ablate: does it actually help?
    Run RAGAS with and without contextual retrieval and compare.
"""

import os
from openai import OpenAI
from langchain_core.documents import Document
from typing import List
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Prompt ─────────────────────────────────────────────────────────────────
# This prompt is the heart of contextual retrieval.
# Key design decisions:
#   - We pass the FULL document so the LLM understands the complete context
#   - We ask for SPECIFIC details: section, topic, identifiers, dates
#   - We explicitly say "output ONLY the context" — no preamble
#   - We cap at 2 sentences — longer context dilutes the chunk embedding
CONTEXT_PROMPT = """<document>
{document}
</document>

Here is a specific chunk from the above document:
<chunk>
{chunk}
</chunk>

Write 1-2 sentences that situate this chunk within the document.
Be specific — include:
- Document type and title if identifiable
- Section or topic this chunk belongs to
- Any relevant identifiers (dates, circular numbers, section numbers)

Output ONLY the context sentences. No preamble, no explanation."""


def add_context_to_chunk(
    full_document: str,
    chunk_text: str,
    source: str = "unknown"
) -> str:
    """
    Generate a contextual prefix for a single chunk.

    Args:
        full_document: the complete document text (for LLM context)
        chunk_text: the specific chunk to contextualise
        source: filename — used as fallback if LLM gives empty response

    Returns:
        contextualised chunk text = LLM context + original chunk

    WHY WE PASS THE FULL DOCUMENT:
        The LLM needs to understand where this chunk sits in the
        overall document. Without the full document, it can only
        describe the chunk itself — not its role in the document.

    TRUNCATION TO 15000 chars:
        LLMs have context limits. We pass the first 15000 characters
        of the document which covers most of the structure and headings.
        Full document for a 50-page PDF might be 100k chars — too long.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            # cheap + fast — perfect for bulk ingestion
            # gpt-4o would be more accurate but 10x more expensive
            # for contextual prefixes, gpt-4o-mini is sufficient

            max_tokens=150,
            # 1-2 sentences = 30-80 tokens
            # 150 is generous upper bound
            # caps cost per call

            temperature=0,
            # temperature=0 means deterministic output
            # we want consistent, factual context not creative writing
            # same chunk should always get same context

            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document analyst. "
                               "Your job is to situate document chunks "
                               "within their source document."
                },
                {
                    "role": "user",
                    "content": CONTEXT_PROMPT.format(
                        # Truncate document to first 15000 chars
                        # Covers document structure, headings, intro
                        # without hitting context limits or costing too much
                        document=full_document[:15000],
                        chunk=chunk_text
                    )
                }
            ]
        )

        context = response.choices[0].message.content.strip()

        # Combine: LLM context + blank line + original chunk
        # The blank line helps the embedding model treat them as
        # related but distinct pieces of information
        return f"{context}\n\n{chunk_text}"

    except Exception as e:
        # If LLM call fails, return original chunk unchanged
        # Better to have a decontextualised chunk than no chunk
        print(f"  Warning: context generation failed for chunk "
              f"from {source}: {e}")
        return chunk_text


def add_context_to_chunks(
    chunks: List[Document],
    full_document: str
) -> List[Document]:
    """
    Add contextual prefixes to a list of chunks.

    Args:
        chunks: list of Document objects from chunker.py
        full_document: complete document text for context generation

    Returns:
        same list of Documents with enriched page_content

    NOTE ON COST:
        This makes one LLM API call per chunk.
        For 100 chunks at gpt-4o-mini pricing (~$0.00015/1k tokens):
        ~100 calls × ~500 tokens each = 50k tokens = ~$0.0075
        Essentially free for the retrieval quality improvement you get.
    """
    print(f"  Adding context to {len(chunks)} chunks...")

    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"  Processing chunk {i+1}/{len(chunks)}...", end="\r")

        enriched_text = add_context_to_chunk(
            full_document=full_document,
            chunk_text=chunk.page_content,
            source=chunk.metadata.get("source", "unknown")
        )

        # Create new Document with enriched text
        # Keep all original metadata — add context_added flag
        enriched_chunks.append(Document(
            page_content=enriched_text,
            metadata={
                **chunk.metadata,           # keep all original metadata
                "context_added": True,      # flag: this chunk was enriched
                "original_text": chunk.page_content  # store original for debugging
            }
        ))

    print(f"\n  Done. {len(enriched_chunks)} chunks enriched.")
    return enriched_chunks


if __name__ == "__main__":
    # Quick test with a sample RBI document chunk
    sample_doc = """
    Reserve Bank of India
    Master Circular on Know Your Customer (KYC) Norms
    RBI/2024-25/16

    1. Introduction
    These directions shall be called the Reserve Bank of India
    (Know Your Customer (KYC)) Directions, 2016.

    2. Scope
    These Directions are applicable to all Regulated Entities (REs).

    3. Customer Due Diligence
    3.1 REs shall carry out CDD measures while establishing
    an account-based relationship.
    3.2 The beneficial owner threshold shall be 10 percent
    for companies and 15 percent for partnerships.
    """

    sample_chunk = ("The beneficial owner threshold shall be 10 percent "
                   "for companies and 15 percent for partnerships.")

    print("Testing contextual retrieval...")
    print(f"Original chunk:\n{sample_chunk}\n")

    enriched = add_context_to_chunk(
        full_document=sample_doc,
        chunk_text=sample_chunk,
        source="rbi_kyc_circular.pdf"
    )

    print(f"Enriched chunk:\n{enriched}")
