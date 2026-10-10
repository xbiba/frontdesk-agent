# ADR-0005: Embeddings and reranking chosen by evaluation

## Status
Proposed. Decided after the retrieval evaluation runs on real models (M1 harness; approved clinic data from M3; decided before production in M5). M3 runs a bounded experiment (`frontdesk eval embedders`): the demo's model against `text-multilingual-embedding-002` and `gemini-embedding-001`, both on Vertex AI `europe-west1` (neither is served on the `eu` multi-region), on at least 30 clinic questions in Albanian, Italian and English written before the run. The rule is fixed before the run: a candidate replaces the reference only if it raises the mean recall@3 over the languages by at least 10 points and lowers it in no language; at parity the reference stays.

## Context
Retrieval must work in Albanian, Italian and English, including a question in one language against a document in another. Query embeddings derive from patient messages, so they must run on approved EU routes.

## Options considered
- Vertex `gemini-embedding-001` (EU).
- Open multilingual models hosted in our EU project (multilingual-e5, BGE-M3).
- Cohere or Voyage, only if available on an approved EU route.
- Reranking and contextual retrieval as optional additions.

## Decision
Current decision: the demo uses `paraphrase-multilingual-MiniLM-L12-v2` locally (384 dimensions); CI uses a deterministic hash embedder that measures nothing about quality. The production model, the reranker and contextual retrieval are chosen by recall@k and MRR per language, with latency and cost.

## Consequences
- The vector column is 384-dimensional; a model with another dimension means a migration.
- Each knowledge version records which embedder computed its vectors. A deployment refuses to start when the configured embedder differs, until `frontdesk knowledge reembed` recomputes them: vectors from two models are never compared.

## Exit plan
Embedders implement one protocol; changing model is configuration plus `frontdesk knowledge reembed`, and a migration when the dimension differs.
