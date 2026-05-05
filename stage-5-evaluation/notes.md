# Stage 5: RAG Evaluation with RAGAS

## Overview

The core question: "How do you know your RAG system is working?" Manual testing is the wrong answer. RAGAS gives quantitative scores for each part of your pipeline — not vibes.

---

## The 4 RAGAS Metrics

### Metric 1 — Faithfulness

- **Measures:** does the answer contain only information from retrieved context?
- **Diagnosis:** low score = LLM is hallucinating from parametric memory
- **Calculation:** break answer into claims, check each against chunks
  ```
  faithfulness = supported claims / total claims
  ```
- **Fix:** tighten prompt with "answer only from context", add "if not in context say I don't know", lower temperature
- **Production threshold:** faithfulness >= 0.85

### Metric 2 — Answer Relevancy

- **Measures:** is the answer relevant to the question asked?
- **Diagnosis:** low score = answer drifted off-topic
- **Calculation:** generate 3-5 hypothetical questions from the answer, measure cosine similarity with original question
- **Why hypothetical questions:** can't compare answer to question directly — different lengths and token distributions. Converting answer to question-space makes cosine comparison meaningful.
- **Fix:** better reranking, more focused prompt

### Metric 3 — Context Precision

- **Measures:** of all chunks retrieved, what fraction were actually useful?
- **Diagnosis:** low score = too much noise in retrieved chunks
- **Calculation:** relevant chunks / total retrieved chunks
- Also penalises poor ranking — relevant chunks buried at bottom score worse
- **Fix:** tune hybrid search weights, improve reranker
- **Example:** retrieve 5 chunks, only 3 used in answer → precision = 0.60

### Metric 4 — Context Recall

- **Measures:** of all facts needed to answer correctly, how many were retrieved?
- **Diagnosis:** low score = retrieval missing relevant chunks
- **Requires:** reference answer in golden dataset
- **Calculation:** facts supported by retrieved chunks / total facts needed
- **Fix:** improve chunking, add contextual retrieval, increase k
- **Example:** answer needs 4 facts, chunks only contain 3 → recall = 0.75

---

## Precision vs Recall — concrete example

Question: "What are the safety procedures for high voltage equipment?"

Reference answer needs 4 facts:
```
Fact 1 — wear insulated gloves
Fact 2 — isolate the circuit first
Fact 3 — use a buddy system
Fact 4 — carry a permit-to-work form
```

Retrieved 5 chunks:
```
Chunk 1 — insulated gloves     ✓ relevant, used
Chunk 2 — circuit isolation    ✓ relevant, used
Chunk 3 — general safety intro ✗ noise, not used
Chunk 4 — buddy system         ✓ relevant, used
Chunk 5 — fire safety          ✗ noise, not used

Context Precision = 3/5 = 0.60  (noisy retrieval)
Context Recall    = 3/4 = 0.75  (missing permit-to-work fact)
```

High precision + low recall scenario:
```
Retrieve only 2 chunks (both relevant, no noise)
Precision = 2/2 = 1.0   (perfect, no noise)
Recall    = 2/4 = 0.50  (missed half the needed facts)
→ retrieval is clean but incomplete
```

---

## Metric Diagnosis Table

| Metric | Low Score Means | Fix |
|---|---|---|
| Faithfulness | LLM hallucinating | Tighten prompt, lower temp |
| Answer Relevancy | Answer off-topic | Better reranking, focused prompt |
| Context Precision | Too much noise | Tune hybrid search, reranker |
| Context Recall | Missing chunks | Better chunking, contextual retrieval, increase k |

---

## The Golden Dataset

- 100-200 QA pairs with reference answers and reference contexts
- Written by domain experts for production
- Synthetically generated + manually reviewed for projects
- Minimum 100 questions for statistically reliable metrics

Each entry structure:
```json
{
  "question": "...",
  "reference_answer": "...",
  "reference_contexts": ["chunk text..."]
}
```

---

## The CI/CD Eval Gate

- GitHub Actions runs RAGAS on every PR touching the RAG pipeline
- PR blocked if: faithfulness < 0.85, answer_relevancy < 0.80
- Catches regressions from prompt changes, chunking changes, model updates
- Answer to: "how do you prevent regressions in production RAG?"

---

## LLM-as-Judge

- Used when no golden dataset exists
- LLM scores (question, context, answer) on faithfulness and relevancy
- Critical caveat: validate against human labels on 50+ examples
- Target >= 90% agreement with humans before trusting in CI
- Bias: judge model prefers answers from similar models

---

## Production Monitoring

- Langfuse traces every production query
- Logs: retrieved chunks, reranker scores, RAGAS metrics, latency, cost
- Alerts when faithfulness drops below 0.80 over rolling 100-query window
- Catches data drift when document corpus changes silently

---

## The Full Eval Stack

```
Development  → RAGAS on golden dataset (200 QA pairs)
CI/CD        → GitHub Actions — block PR if metrics drop
Production   → Langfuse tracing + rolling metric alerts
Ad-hoc       → LLM-as-judge for new query categories
```

---

## Key Interview Questions

**Q: Faithfulness is 0.60 — what does it mean and what do you fix?**
A: 40% of claims in the answer are not supported by retrieved context — LLM is hallucinating from memory. Fix: add "answer only from context", add "if not in context say I don't know", lower temperature.

**Q: Difference between context precision and context recall?**
A: Precision = of what I retrieved, how much was useful (no noise). Recall = of what was needed, how much did I retrieve (no gaps). High precision + low recall = clean but incomplete retrieval. Fix precision → tune reranker. Fix recall → fix chunking, increase k.

**Q: Why generate hypothetical questions for answer relevancy?**
A: Can't compare answer to question directly — different lengths and token distributions make cosine similarity meaningless. Converting the answer to question-space makes comparison valid.

**Q: Colleague says "I'll manually test 10 queries before deploy."**
A: 10 queries can't cover the distribution of real user queries. A prompt change that improves those 10 might silently hurt 200 others. Without automated metrics you have no regression protection. One bad deploy takes weeks to diagnose. RAGAS with 200 questions and CI gates takes one afternoon to set up and protects you forever.
