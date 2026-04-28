# Stage 4: Advanced RAG — The Fixes

> **Goal:** Understand the four targeted fixes that turn a failing naive RAG
> into a production-grade system. Each fix maps directly to one failure mode
> from Stage 3.

---

## Overview

| Failure Mode (Stage 3) | Advanced RAG Fix |
|---|---|
| 1 — Decontextualised chunks | Contextual Retrieval (Anthropic, Sept 2024) |
| 2 — Semantic retrieval blind spots | Hybrid Search + Reciprocal Rank Fusion |
| 3 — No reranking | Cross-Encoder Reranking |
| 4 — No evaluation loop | RAGAS Eval Harness (covered in depth in Stage 5) |

These fixes are **additive** — each one stacks on top of naive RAG. A
production system deploys all four together. Understanding why each fix exists
(the failure it solves) is more important in interviews than reciting the
mechanics.

```
ADVANCED RAG — FULL PIPELINE

INGESTION (offline)
──────────────────────────────────────────────────────────
Documents → Layout-aware chunking
         → LLM context generation (cached, once per chunk)
         → Contextual chunk = context summary + original chunk
         → Embedding → Vector Store (dense index)
         → BM25 indexing (sparse index)

QUERY (online, per request)
──────────────────────────────────────────────────────────
User Query
  │
  ├─► Dense retrieval  (top-50 by cosine)  ─┐
  │                                          ├─► RRF merge → top-50 merged
  └─► BM25 retrieval   (top-50 by TF-IDF)  ─┘
                                              │
                                    Cross-encoder rerank (top-50 → top-8)
                                              │
                                    Prompt construction
                                              │
                                            LLM
                                              │
                                    Grounded Answer
                                              │
                                    RAGAS evaluation (async)
```

---

## Fix 1 — Contextual Retrieval (Anthropic, September 2024)

### The problem it solves

Naive chunking creates decontextualised fragments. A chunk like:

> *"The ratio shall not fall below the prescribed threshold at any point
> during the financial year."*

Read in isolation: what ratio? Prescribed by whom? For what entity? Effective
when? The chunk's embedding reflects this incompleteness — it is a vague,
underspecified vector that may match queries it shouldn't and miss queries it
should.

### How it works

Before embedding each chunk, make **one LLM call** to generate a 1-2 sentence
contextual summary that situates the chunk within the document — capturing the
source document name, section, topic, relevant identifiers, and date. Then
**prepend this summary to the chunk** before embedding.

The critical detail: **the LLM call happens at ingestion time, not query
time.** You pay for it once per chunk when building the index. The query
pipeline is unchanged.

```
FOR EACH CHUNK at ingestion:

  prompt = f"""
  Document: {full_document}        ← cached prefix
  -----
  Chunk: {chunk_text}              ← only this changes per call

  Generate a 1-2 sentence context that situates this chunk
  within the document — include section name, topic, identifiers.
  """

  context_summary = llm.generate(prompt)
  contextual_chunk = context_summary + "\n\n" + chunk_text
  vector = embedding_model.embed(contextual_chunk)
  vector_store.upsert(vector, metadata={"text": contextual_chunk})
```

### Example transformation

**Before (naive chunk):**
> *"The ratio shall not fall below the prescribed threshold at any point
> during the financial year."*

**After (contextual chunk):**
> *"From RBI Master Circular 2024/103, Section 4.2 on NBFC prudential norms.
> This chunk covers the minimum Liquidity Coverage Ratio (LCR) requirement
> effective January 2025.*
>
> *The ratio shall not fall below the prescribed threshold at any point
> during the financial year."*

Now the embedding captures the document, the section, the regulation, the
metric name, and the effective date — not just the vague sentence. A query
about "RBI NBFC liquidity ratio 2025" finds this chunk precisely.

### Cost optimisation with prompt caching

The naive implementation of contextual retrieval is expensive: one LLM call
per chunk, with the full document in every prompt. For a 500-chunk document,
that's 500 × (full document tokens) = enormous cost.

**The fix: prompt caching.**

The full document is the **cached prefix** — it stays constant across all
chunk calls for the same document. Only the chunk text changes. With Anthropic
prompt caching:

1. The full document tokens are cached on the first call.
2. Subsequent calls (different chunks, same document) hit the cache —
   charged at ~10% of normal input token cost.
3. Net ingestion LLM cost drops ~90%.

**Model choice:** Use `claude-3-haiku` for bulk ingestion. It is 10-20× faster
and cheaper than Sonnet for this task. The contextual summary doesn't require
deep reasoning — haiku is sufficient.

### The result

Anthropic's internal benchmarks reported a **49% reduction in retrieval
failures** versus naive chunking on the same corpus.

### Interview answer

> **"Before embedding each chunk, I prepend an LLM-generated 1-2 sentence
> contextual summary — document name, section, topic, identifiers — to make
> the chunk self-contained. The LLM call happens at ingestion time, paid once.
> I use prompt caching on the full document so the document tokens are cached
> across all chunks — ingestion LLM cost drops ~90%. Anthropic reported 49%
> fewer retrieval failures. It's the highest-ROI single improvement you can
> make to a naive RAG system."**

---

## Fix 2 — Hybrid Search with Reciprocal Rank Fusion

### The problem it solves

Dense-only retrieval fails on **exact-match queries** — work order IDs, FLOC
codes, legal citations, financial identifiers, medical codes. Embedding models
compress text into semantic direction vectors, diluting token identity in the
process. A query for "FLOC LB765" returns generic asset management chunks,
because "LB765" is just a rare token that gets averaged into a regional
semantic concept.

BM25 handles these cases natively — it scores by exact term frequency and
inverse document frequency. Rare tokens like "LB765" receive extremely high
IDF scores and are found instantly.

The challenge: you cannot simply run both retrievers and add their scores.

### Why you cannot add scores directly

BM25 and dense similarity operate on completely different scales:

- BM25 score for the top result: might be 18.4
- Dense cosine similarity for the top result: might be 0.87

These numbers are not comparable. Naively summing them — `18.4 + 0.87 = 19.27`
— gives BM25 an overwhelming weight advantage regardless of relevance. The
merged ranking would just be BM25's ranking with a small dense perturbation.

**You need a scale-free merge mechanism.**

### Reciprocal Rank Fusion — the solution

RRF converts scores to **ranks** and applies the formula:

```
RRF_score(document d) = Σ  1 / (k + rank(d, list))
                       lists
```

Where `k = 60` is a damping constant (empirically chosen, widely used). The
`k` prevents top-ranked documents from completely dominating — a document at
rank 1 gets `1/61 = 0.0164`, not an infinitely large score.

Ranks are always on the same scale (1, 2, 3, ...) regardless of the original
scoring system. RRF needs no normalisation, no calibration, no tuning of
relative weights.

### RRF worked example

**Query:** `"FLOC LB765 maintenance history"`

```
Dense retrieval results:
  Rank 1: Generic maintenance procedures doc      (cosine 0.81)
  Rank 2: Asset management policy doc             (cosine 0.79)
  Rank 3: Preventive maintenance schedule doc     (cosine 0.77)
  Rank 4: LB765 specific maintenance history doc  (cosine 0.71)  ← buried

BM25 retrieval results:
  Rank 1: LB765 specific maintenance history doc  (score 21.4)   ← exact match
  Rank 2: LB765 incident report 2023              (score 18.9)
  Rank 3: FLOC maintenance procedures overview    (score 12.1)
  Rank 8: Generic maintenance procedures doc      (score 4.2)

RRF calculation (k=60):

  LB765 maintenance history doc:
    Dense rank 4  →  1 / (60 + 4)  = 0.01563
    BM25  rank 1  →  1 / (60 + 1)  = 0.01639
    RRF score     =  0.01563 + 0.01639 = 0.03202  ← rises to rank 1

  Generic maintenance procedures doc:
    Dense rank 1  →  1 / (60 + 1)  = 0.01639
    BM25  rank 8  →  1 / (60 + 8)  = 0.01471
    RRF score     =  0.01639 + 0.01471 = 0.03110  ← drops to rank 2
```

The LB765 document, buried at rank 4 in dense retrieval, rises to rank 1 in
the merged list because BM25 strongly identified it via exact token matching.

### Implementation

```python
def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results, start=1):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank)
    for rank, doc in enumerate(bm25_results, start=1):
        scores[doc.id] = scores.get(doc.id, 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### Production notes

- **OpenSearch** supports both HNSW (dense) and BM25 natively — one database
  for both indices. This is the standard production choice for hybrid RAG.
- **Qdrant** supports hybrid search with sparse vectors (BM25/SPLADE) built-in.
- Run dense and BM25 retrieval in parallel — they are independent.
- Retrieve top-50 from each before merging. After RRF, pass the merged top-50
  to the reranker.

### Interview answer

> **"I can't add BM25 and dense scores directly — they're on completely
> different scales. RRF solves this by converting both to ranks and applying
> `1 / (k + rank)`. Ranks are always comparable regardless of the original
> scoring system. k=60 is the standard constant — it dampens top-rank
> dominance. BM25 handles exact match (IDs, codes, citations), dense handles
> semantic meaning, RRF merges both. For the merge I retrieve top-50 from
> each, run RRF, pass the merged top-50 to the reranker."**

---

## Fix 3 — Cross-Encoder Reranking

### The problem it solves

After hybrid search, you have a merged list of the top-50 most likely relevant
chunks. But "most likely" is doing a lot of work. Embedding models (bi-encoders)
are optimised for **recall** — find broadly relevant candidates quickly. They
are not optimised to rank which chunk best answers the *specific query*. The
most relevant chunk might be at rank 12 in the merged list.

Sending ranks 1-8 to the LLM without reranking risks missing the most
relevant chunk entirely.

### The bi-encoder vs cross-encoder distinction

This is one of the most important architectural concepts in production RAG.

**Bi-encoder (embedding models — used in retrieval):**

```
Query: "What are the overdraft charges?"
  │
  └─► Embedding Model ─► Query vector [0.23, -0.71, ...]

Chunk: "The bank charges a 2% fee on overdraft amounts..."
  │
  └─► Embedding Model ─► Chunk vector [0.19, -0.68, ...]

Similarity: cosine(query_vector, chunk_vector) = 0.87
```

The query and chunk are encoded **independently**. They never see each other
during encoding. The model produces a vector for the query in isolation —
it must capture a direction in semantic space that covers all possible meanings
of "charges" (banking fees? criminal charges? electrical charge?). It produces
a compromise vector. The final similarity score is a dot product of two
independently-computed vectors.

**Fast:** chunk vectors are pre-computed at ingestion. Query embedding is one
forward pass. Similarity is a dot product — O(1) per chunk.

**Imprecise:** meaning is context-dependent. An isolated query vector cannot
fully capture the query's intent in the context of a specific chunk.

---

**Cross-encoder (reranker — used after retrieval):**

```
Input: "What are the overdraft charges? [SEP] The bank charges a 2% fee
        on overdraft amounts exceeding the account balance..."
  │
  └─► Cross-Encoder Model ─► Relevance Score: 0.94
```

The query and chunk are **concatenated** and fed together as a single input.
The model reads both simultaneously — it reasons about their specific
relationship. It doesn't produce vectors. It outputs a single relevance score
directly.

**Slow:** one forward pass per (query, chunk) pair. For 50 candidates, that's
50 forward passes. Cannot be pre-computed.

**Accurate:** the model has full context — it knows the query, it knows the
chunk, and it reasons about how well this specific chunk answers this specific
query. No vector compression, no contextual averaging.

### Why "top-50 then rerank to top-8" — not just "top-8 directly"

Recall and precision are in tension. You cannot optimise both in a single
retrieval step.

```
Option A: retrieve top-8 directly with bi-encoder
  - Fast
  - If the correct chunk is at true rank 12, it never enters the pipeline
  - The LLM never sees the most relevant chunk
  - Generation quality suffers

Option B: retrieve top-50, rerank to top-8 with cross-encoder
  - Retrieval: wide net ensures correct chunk is in the pool (even if at rank 12)
  - Reranking: cross-encoder reads all 50 pairs, promotes rank-12 chunk to rank 1
  - Send top-8 (now correctly ranked) to LLM
  - Better precision, fewer tokens, lower LLM cost
```

The two stages serve different purposes and cannot be collapsed:

| Stage | Model type | Purpose | Speed | Optimised for |
|---|---|---|---|---|
| Retrieval (top-50) | Bi-encoder | Cast wide net | Fast | Recall |
| Reranking (top-8) | Cross-encoder | Promote best result | Slow | Precision |

### The hiring analogy

> Top-50 retrieval = HR screening 500 résumés down to 50 using keyword
> matching. Fast, broad, some false positives acceptable. The goal is to
> ensure the right candidate is in the pile — not to rank them perfectly.
>
> Reranking = the hiring manager interviewing all 50 candidates one by one,
> asking probing questions, evaluating each against the specific role
> requirements. Slow, deep, highly accurate.
>
> You need both stages. Skipping HR screening means the hiring manager drowns
> in 500 candidates. Skipping the interview means you hire based on keywords.

### Production reranker options

| Model | Type | Notes |
|---|---|---|
| **Cohere Rerank 3** | Hosted API | No infra, multilingual, production-grade |
| **BGE-reranker-v2-m3** | Open source | Self-hosted, strong multilingual performance |
| **ms-marco-MiniLM-L-6-v2** | Open source | Fast, English-only, good for low-latency |

**Latency consideration:** 50 cross-encoder forward passes add ~200-400ms to
query latency. Run retrieval and reranking in the critical path; run RAGAS
evaluation async after returning the response to the user.

### Interview answer

> **"Bi-encoders optimise for recall — they cast a wide net but can't reason
> about specific query-chunk relationships. Cross-encoders read query and chunk
> jointly and output a direct relevance score — accurate but slow. The pattern
> is: retrieve top-50 with the bi-encoder to ensure the correct chunk is in
> the candidate pool (it might be at rank 12), rerank with a cross-encoder to
> promote it to rank 1, then send only the top-8 to the LLM. This gives you
> both recall and precision without paying cross-encoder cost on millions of
> chunks."**

---

## Fix 4 — RAGAS Eval Harness

The evaluation layer is covered in full depth in Stage 5. The core concept:

Without measurement, any change to the RAG pipeline — prompt edits, chunk size
changes, model upgrades — could silently degrade quality. A RAGAS eval harness
runs on every PR, measures faithfulness and context recall against a golden
dataset, and blocks merges that regress below threshold.

Covered in detail: [Stage 5 — RAG Evaluation with RAGAS](../stage-5-evaluation/notes.md)

---

## Key Interview Questions

---

**Q: What does contextual retrieval add and when does the LLM call happen?**

> **A:** Contextual retrieval prepends a 1-2 sentence LLM-generated summary to
> each chunk before embedding. The summary captures the source document, section,
> topic, and identifiers — making the chunk self-contained and the embedding
> precise. The LLM call happens at ingestion time, once per chunk, not at query
> time. Using prompt caching on the full document reduces the cost by ~90%.
> Anthropic reported 49% fewer retrieval failures in their benchmarks.

---

**Q: Why can't you add BM25 and dense scores directly?**

> **A:** They use completely different scoring scales — BM25 might score 0-25,
> dense cosine similarity 0-1. Adding them directly gives BM25 overwhelming
> weight advantage regardless of relevance. RRF solves this by converting both
> lists to ranks and applying `1 / (k + rank)`. Ranks are always on the same
> scale (integers starting at 1) regardless of the original scoring system —
> no normalisation needed.

---

**Q: Walk through an RRF calculation.**

> **A:** Query: "contract clause 17b". Dense rank 2, BM25 rank 6, k=60:
> - Dense: `1 / (60 + 2) = 0.01613`
> - BM25: `1 / (60 + 6) = 0.01515`
> - RRF score: `0.01613 + 0.01515 = 0.03128`
>
> Compare to a document at dense rank 1, BM25 rank 15:
> - Dense: `1 / (60 + 1) = 0.01639`
> - BM25: `1 / (60 + 15) = 0.01333`
> - RRF score: `0.01639 + 0.01333 = 0.02972`
>
> The first document wins despite being rank 2 in dense, because BM25 strongly
> identified it as an exact match.

---

**Q: Why retrieve top-50 before reranking to top-8?**

> **A:** Recall and precision are in tension — you can't optimise both in one
> step. Retrieving top-50 ensures the correct chunk is in the candidate pool
> even if the bi-encoder ranks it 12th. The cross-encoder then reads all 50
> (query, chunk) pairs jointly and promotes the truly relevant chunk to rank 1.
> If you retrieve only top-8 directly, you might never see the most relevant
> chunk — it's gone before reranking even starts.

---

**Q: Bi-encoder vs cross-encoder?**

> **A:** Bi-encoder encodes query and chunk independently, compares via dot
> product — fast (pre-computable), optimised for recall. Cross-encoder reads
> query and chunk jointly, outputs a direct relevance score — slow (one forward
> pass per pair, cannot be pre-computed), optimised for precision. Use
> bi-encoder for retrieval (scale to millions of chunks), cross-encoder for
> reranking (only runs on the top-50 candidates).

---

**Q: Why does a cross-encoder produce a score instead of a vector?**

> **A:** Because it receives both query and chunk as a concatenated input — it
> has enough information to directly reason about their specific relationship
> and output a single relevance number. There is no need to produce vectors or
> compute a dot product. This is also why it's more accurate: the model isn't
> trying to summarise the query or chunk independently — it's evaluating the
> specific pair as a unit.

---

## Summary — What Each Fix Buys You

| Fix | Failure it solves | Mechanism | Benchmark impact |
|---|---|---|---|
| Contextual Retrieval | Decontextualised chunks | LLM-generated context prepended at ingestion | −49% retrieval failures (Anthropic) |
| Hybrid Search + RRF | Semantic blind spots on exact-match queries | Dense + BM25 merged via rank-based fusion | Catches ID/code queries that dense misses |
| Cross-Encoder Reranking | Imprecise ranking from bi-encoders | Joint (query, chunk) reasoning post-retrieval | Promotes correct chunk from rank 12 to rank 1 |
| RAGAS Eval Harness | No measurement, silent regressions | Faithfulness/recall metrics on golden dataset | Merge-blocking safety net for all future changes |

> These four fixes are not optional in production. Each one addresses a specific
> failure mode that will hurt real users. A system without contextual retrieval
> fails on complex documents. A system without hybrid search fails on any query
> containing an identifier. A system without reranking sends noise to the LLM.
> A system without evaluation is one prompt edit away from silent regression.
