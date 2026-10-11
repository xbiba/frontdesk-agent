# Recorded evaluations

Runs with real models, kept as evidence for ADR-0004, ADR-0005 and later comparisons. The M3 live turn runs and the embedder experiment are launched from the private configuration repository; their summaries are added here. Offline runs with the hash embedder and the demo models check contracts only and are not recorded here.

| Date | Business | What | Embedder / model | File |
|---|---|---|---|---|
| 2026-10 | demo-clinic | retrieval, k = 1, 3, 5 | `fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | [2026-10-retrieval-demo-clinic.md](2026-10-retrieval-demo-clinic.md) |
| 2026-10 | demo-clinic, demo-shop | live turns after tuning (M3 Task 14): exit criteria, by language, model requests | `gemini-3.1-flash-lite` (Vertex AI `eu`) | [2026-10-m3-live-demo.md](2026-10-m3-live-demo.md) |
