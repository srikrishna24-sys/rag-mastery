# Stage 2: Naive RAG — The Baseline

## Overview

Two completely separate phases — ingestion (runs once) and query (runs every time). Never conflate them — common interview mistake.

---

## Phase 1: Ingestion Pipeline

### Chunking

- **What it is:** splitting documents into smaller pieces (256–1024 tokens) that can be embedded and retrieved individually
- Too small (64 tokens): chunk lacks enough context to be meaningful
- Too large (2048 tokens): embedding averages too much meaning, retrieval becomes imprecise
- Sweet spot: 512 tokens with 50-token overlap for dense technical text, 256 for conversational
- **Why overlap exists:** key sentences at chunk boundaries get captured by both adjacent chunks — hard splits lose boundary information
- **Interview answer for chunk size:** "Depends on document type. Always ablate — run RAGAS eval across chunk sizes and pick best faithfulness score."

### Embeddings

- **What it is:** text → vector of 768 or 1536 floats
- **Key property:** semantic proximity — similar meaning = close vectors
- Cosine similarity = (A · B) / (|A| × |B|)
- **Why cosine not Euclidean:** cosine measures direction (meaning), Euclidean measures magnitude (text length). We care about meaning.
- **CRITICAL:** must use same embedding model for chunks AND queries — different models = different vector spaces = meaningless scores

### Vector Store

- **What it is:** database built specifically for similarity search
- **HNSW index:** graph-based approximate nearest neighbour search — O(log n) vs brute force O(n) — this is what makes it fast
- **FAISS:** in-memory library, local dev, small datasets (<1M vectors)
- **Qdrant:** open source, self-hostable, production-grade, hybrid search
- **Pinecone:** managed cloud, scales to billions, costs money

---

## Phase 2: Query Pipeline

### Query Embedding

- User question goes through the SAME embedding model used at ingestion
- Produces a vector in the same space as the stored chunks

### Similarity Search

- Vector store finds k chunks with highest cosine similarity to query
- k = 3 to 5 in naive RAG
- Too small: miss relevant context
- Too large: bloat prompt, increase cost, degrade generation quality

### Prompt Construction

- Retrieved chunks injected into a template as context
- "Answer ONLY from the context below" is load-bearing — without it LLM blends retrieved context with parametric memory and hallucinates
- If answer not in context, LLM should say so — not fabricate

### Generation

- LLM reads context and answers from it — not from its weights
- Model's job shifts from "remember everything" to "read and answer"
- This is the core mechanism that prevents hallucination in RAG

---

## Key Interview Questions

**Q: What is the difference between ingestion and query phase?**
A: Ingestion runs once to build the index — chunk, embed, store. Query runs on every user request — embed query, retrieve top-k, construct prompt, generate answer.

**Q: Why use cosine similarity not Euclidean distance?**
A: Cosine measures the angle between vectors — direction represents meaning. Euclidean measures raw distance which is affected by text length. We care about semantic meaning not document length.

**Q: Why must chunk and query use the same embedding model?**
A: Both must exist in the same vector space for similarity scores to be meaningful. Different models produce different spaces — coordinates become incomparable.

**Q: What is HNSW?**
A: Graph-based approximate nearest neighbour index. Searches in O(log n) by navigating a multi-layer graph from coarse to fine. Used internally by Pinecone, Qdrant, Weaviate.

**Q: What chunk size would you use?**
A: Depends on document type — 512 tokens with 50 overlap for dense technical text, 256 for conversational. Always ablate — run RAGAS eval across chunk sizes and pick best faithfulness score.

**Q: Why is "answer only from context" load-bearing in the prompt?**
A: Without it the LLM blends retrieved context with its parametric memory. When they conflict, hallucinations creep in. The explicit constraint forces the model to stay grounded in the retrieved docs.
