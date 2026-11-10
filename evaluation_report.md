# DocuMind AI — Evaluation Benchmark Report

## 1. Executive Summary
- **Evaluation Date**: 2026-11-10
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- **Vector Store**: FAISS IndexFlatIP (Cosine Similarity)
- **Text Splitter**: RecursiveCharacterTextSplitter (chunk_size=900, chunk_overlap=150)
- **Top-1 Retrieval Accuracy**: 100.0%
- **Average Retrieval Latency**: 24.98 ms
- **Edge Cases Tested**: Corrupted PDF, Empty PDF, Zero-length stream, Out-of-Scope queries, Prompt Injection

## 2. Benchmark Query Results

| Query ID | Expected Page | Top Retrieved Page | Top Similarity Score | Status |
|---|---|---|---|---|
| `Q1_OVERVIEW` | Page 1 | Page 1 | 81.6% | PASS |
| `Q2_FINANCIALS` | Page 2 | Page 2 | 54.4% | PASS |
| `Q3_TECH_STACK` | Page 3 | Page 3 | 53.8% | PASS |
| `Q4_COMPLIANCE` | Page 4 | Page 4 | 52.2% | PASS |
| `Q5_OUT_OF_SCOPE` | Page None | Page None | 0.0% | PASS |
| `Q6_PROMPT_INJECTION_DEFENSE` | Page 2 | Page 2 | 29.4% | PASS |

## 3. Grounding & Anti-Hallucination Guardrails
1. **Verifiable Citations**: LLM responses are parsed and verified against actual retrieved document chunks and page numbers.
2. **Refusal Mechanism**: Queries lacking document support or scoring below similarity threshold trigger the strict refusal: *"I couldn't find sufficient information in the uploaded documents."*
3. **Prompt Injection Defense**: Retrieved passages are delimited within clear data boundary blocks and marked as untrusted data in the system prompt.
