# Stage 3: Where Naive RAG Breaks

## Overview

Naive RAG works for demos. It fails in production for 4 specific reasons. Interviewers don't ask "how does RAG work" — they ask "where does RAG break." These 4 failure modes are the answer.

---

## Failure Mode 1 — Decontextualised chunks

### The problem

- Chunks are fragments with no memory of their source document, section, or surrounding context
- Fixed-size chunking breaks tables, code blocks, and lists mid-way

### Concrete example

- RBI Master Circular chunk: "The ratio shall not fall below the prescribed threshold at any point during the financial year."
- Read in isolation: what ratio? prescribed by whom? for what entity?
- Matches semantically but LLM cannot answer correctly from it

### Why the embedding doesn't save you

- Decontextualised chunk → decontextualised embedding
- May match queries it shouldn't, miss queries it should

### Interview answer

"Naive chunking creates decontextualised fragments with no awareness of source document or section. Combined with fixed-size splits that break tables and lists, the LLM receives incomplete context even when retrieval rank is correct. Contextual retrieval and layout-aware chunking fix this."

---

## Failure Mode 2 — Semantic retrieval blind spots

### The problem

- Dense embeddings capture meaning but lose exact token identity
- Identifiers, codes, citations, proper nouns have no semantic meaning
- Embedding model compresses them into generic regional vectors

### Why it happens

- Embedding models trained on semantic similarity — synonyms, paraphrases, related concepts
- "WO-2024-MUM-4521" becomes a vector near "maintenance work orders" generally — not the specific work order
- "FLOC LB765" → dense returns generic asset management chunks, BM25 finds LB765 instantly because it's a rare exact token

### Real examples

- Work order IDs (WO-2024-MUM-4521)
- FLOC codes (LB765) — Shell maintenance systems
- Legal citations (Section 17(3)(b) Companies Act)
- Financial identifiers (ISIN INE002A01018)
- Medical codes (ICD-10 E11.65)
- RBI circular numbers (RBI/2024/103)

### Why BM25 fixes this

- BM25 scores by term frequency × inverse document frequency
- Rare tokens like "LB765" get extremely high IDF score
- Finds the exact document immediately

### Dense vs BM25 comparison

```
query: "FLOC LB765 maintenance history"

Dense rank 1: generic asset maintenance procedures (0.81)
Dense rank 4: actual LB765 document (0.71) ← buried
BM25 rank 1:  actual LB765 document (0.97) ← instant exact match
```

### Interview answer

"Dense retrieval fails on exact-match queries — IDs, codes, citations, names — because embedding models dilute token identity during semantic compression. BM25 handles these via term frequency matching on exact tokens. Production RAG always uses hybrid search — dense for semantic understanding, BM25 for exact match, merged via Reciprocal Rank Fusion."

---

## Failure Mode 3 — No reranking

### The problem

- Embedding models optimised for recall — find broadly relevant chunks
- NOT optimised to rank which chunk best answers the specific question
- Most relevant chunk often sits at rank 4 or 5, not rank 1

### Bi-encoder vs cross-encoder — critical concept

**Bi-encoder (embedding models):**
- Query and chunk encoded INDEPENDENTLY
- Each produces a vector in isolation — never see each other
- Similarity computed by dot product AFTER encoding
- Fast: chunk vectors pre-computed at ingestion, O(1) lookup
- Imprecise: meaning is context-dependent, vector is a compromise

**Cross-encoder (rerankers):**
- Query and chunk CONCATENATED and fed together as one input
- Model reads both simultaneously — reasons about their relationship
- Outputs a single relevance score directly — no vectors, no dot product
- Slower: one forward pass per (query, chunk) pair — O(k) calls
- Accurate: sees full context, reasons about specific relationship

### Why cross-encoder has better accuracy

- Bi-encoder: "What are the charges?" encoded without seeing any chunk
  → must pick a direction in vector space (banking? legal? physics?)
  → produces a compromise vector averaging all possible meanings
- Cross-encoder: "What are the charges? [SEP] The defendant faces three counts of fraud..."
  → immediately understands legal context
  → reasons: yes, this is relevant. Score: 0.94
- Contextual reasoning about the specific pair = better accuracy

### The production pattern

```
retrieve top-50 with bi-encoder   → broad recall, fast
rerank to top-8 with cross-encoder → precise relevance, slower
send only top-8 to LLM            → better precision, fewer tokens, lower cost
```

### Interview answer

"Bi-encoders are fast because they compare pre-computed vectors but can't reason about the query-chunk relationship. Cross-encoders read both together and reason about their specific relationship — more accurate but slower. Use bi-encoders for retrieval speed and cross-encoders for reranking precision. Retrieve top-50, rerank to top-8, send top-8 to LLM."

---

## Failure Mode 4 — No evaluation loop

### The problem

- RAG systems shipped based on manual testing and vibes
- No quantitative measurement of quality
- No way to detect if a change made things worse

### Why this is a production killer

Real scenario: engineer tweaks prompt template to sound more conversational. Faithfulness drops from 0.87 to 0.71. Nobody knows because there is no measurement. Users get hallucinated answers. Three weeks later a customer complains. Nobody can trace it to the prompt change.

### What a proper eval harness looks like

- Golden dataset: 100-200 QA pairs with known correct answers
- RAGAS metrics computed on every PR via GitHub Actions
- Merge-blocking threshold: faithfulness must stay above 0.85
- Production monitoring: Langfuse traces for drift detection

### Interview answer

"Without an eval harness you're flying blind — any change could hurt quality without you knowing. The fix is a golden dataset of labelled QA pairs, RAGAS metrics on every PR, and a merge-blocking faithfulness threshold. That's the minimum for a production RAG."

---

## Key interview questions for this stage

**Q: Why does a decontextualised chunk cause retrieval failure?**
A: The chunk has no awareness of its source document or section. The embedding reflects this incomplete context — it may match queries it shouldn't and miss queries it should.

**Q: Give an example where dense retrieval fails but BM25 succeeds.**
A: Searching for a FLOC code like LB765 in a maintenance system. Dense returns generic asset management chunks. BM25 finds LB765 instantly because it's a rare token with a high IDF score.

**Q: What is the difference between a bi-encoder and cross-encoder?**
A: Bi-encoders encode query and chunk independently and compare vectors — fast but imprecise. Cross-encoders read both together and output a relevance score directly — slower but accurate. Use bi-encoders for retrieval, cross-encoders for reranking.

**Q: Why does a cross-encoder produce a score instead of a vector?**
A: Because both query and chunk are passed together — the model has enough information to directly reason about their relationship and output a single relevance number. No vector comparison needed.

**Q: Why is an eval harness non-negotiable in production?**
A: Without it, quality regressions from prompt changes, model updates, or data drift go undetected. A golden dataset with RAGAS metrics and CI gates is the minimum viable safety net.
