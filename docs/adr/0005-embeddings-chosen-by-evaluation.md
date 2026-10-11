# ADR-0005: Embeddings and reranking chosen by evaluation

## Status
Accepted on 2026-10-11, for M5 as well. Decided after the retrieval evaluation runs on real models (M1 harness; approved clinic data from M3; decided before production in M5). M3 runs a bounded experiment (`frontdesk eval embedders`): the demo's model against `text-multilingual-embedding-002` and `gemini-embedding-001`, both on Vertex AI `europe-west1` (neither is served on the `eu` multi-region), on at least 30 clinic questions in Albanian, Italian and English written before the run. On 2026-10-11 it ran on 36 Clinica MCU questions, 12 per language, under the rule fixed before the run: a candidate replaces the reference only if it raises the mean recall@3 over the languages by at least 10 points and lowers it in no language. The reference reached a mean recall@3 of 0.97 (Albanian 1.00, Italian 0.92, English 1.00) and both candidates 1.00, a gain of 2.8 points, so the reference stays. The candidates put the right document first more often in Albanian (recall@1 0.75 and 0.67 against 0.50); that matters only if replies come to rely on the first result alone. The table is in `runs/2026-10-11-1437-embedders/embedders.md` of the clinic's configuration repository.

## Context
Retrieval must work in Albanian, Italian and English, including a question in one language against a document in another. Query embeddings derive from patient messages, so they must run on approved EU routes.

## Options considered
- Vertex `gemini-embedding-001` (EU).
- Open multilingual models hosted in our EU project (multilingual-e5, BGE-M3).
- Cohere or Voyage, only if available on an approved EU route.
- Reranking and contextual retrieval as optional additions.

## Decision
Production keeps `paraphrase-multilingual-MiniLM-L12-v2`, run locally (384 dimensions), as the demo does; CI uses a deterministic hash embedder that measures nothing about quality. No reranker and no contextual retrieval: at a recall@3 of 0.97 on the clinic's questions they have nothing to win. A new candidate is measured the same way, by recall@k and MRR per language, with latency and cost.

## Consequences
- The vector column is 384-dimensional; a model with another dimension means a migration.
- Each knowledge version records which embedder computed its vectors. A deployment refuses to start when the configured embedder differs, until `frontdesk knowledge reembed` recomputes them: vectors from two models are never compared.

## Exit plan
Embedders implement one protocol; changing model is configuration plus `frontdesk knowledge reembed`, and a migration when the dimension differs.
