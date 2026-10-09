# Retrieval evaluation: demo-clinic with the demo embedder

The first retrieval evaluation with a real embedder (the M1 acceptance). It runs the labelled questions of `businesses/demo-clinic/evals/retrieval.yaml` through the hybrid search that answers a customer's question (full text plus vectors, fused with Reciprocal Rank Fusion). The knowledge was embedded by the same model as the questions, on an empty database.

| | |
|---|---|
| Date | 2026-10-09 |
| Business | demo-clinic (synthetic data) |
| Questions | 8: English 3, Italian 3, Albanian 2. Tags: cross-lingual 2, no-diacritics 2, paraphrase 1 |
| Knowledge | 5 documents (Italian 2, Albanian 2, English 1), 9 chunks |
| Embedder identity | `fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, as the report prints it and as the knowledge version records it |
| Model files | Hugging Face `Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`, revision `faf4aa4225822f3bc6376869cb1164e8e3feedd0`; `model_optimized.onnx`, 235,052,644 bytes, sha256 `634d0f66c29dc934c8fa72b8a4fe91dd4d420a22f1d82a241058d4316e659a99`; 384 dimensions, inputs cut at 128 tokens (fastembed's description of the model) |
| Library | fastembed 0.9.0 on onnxruntime 1.30.0 |
| Code | the commit that adds this file |

Commands, from an empty database volume:

```bash
uv sync --extra demo
docker compose up -d postgres
uv run frontdesk db upgrade --env-file deploy/demo-clinic.env
FRONTDESK_EMBEDDING_BACKEND=fastembed uv run frontdesk knowledge bootstrap-demo --env-file deploy/demo-clinic.env
FRONTDESK_EMBEDDING_BACKEND=fastembed uv run frontdesk eval retrieval --env-file deploy/demo-clinic.env --k 1   # and --k 3, --k 5
```

## Reading the numbers

- For all 8 questions the first document returned is a relevant one: MRR is 1.00 at every k.
- The one row below 1.00 at k = 1 is `physio-prepare-en`. It has two relevant documents (the Italian and the Albanian one), so its recall@1 cannot exceed 0.50. Mean recall@1 of 0.94 and cross-lingual recall@1 of 0.75 are the highest values this set allows at k = 1, not misses. At k = 3 both reach 1.00.
- The knowledge is 5 documents in 9 chunks. At k = 5 the ranking reaches every document, so recall@5 is 1.00 for any embedder; recall@3 only asks for the relevant document to be among the first three of five. Read k = 1 and MRR; recall@3 and recall@5 are recorded for completeness.
- The same questions with the hash embedder of the keyless demo (a run that is not recorded separately) give MRR 0.75 at k = 1, 0.79 at k = 3 and 0.82 at k = 5. All of the difference is in the two cross-lingual questions, which the hash embedder misses at k = 1 (recall@1 0.00): it shares no words between languages. So the set does tell a multilingual model from a bag of words.
- 8 questions and 5 documents are too few to compare models or to choose the production embedder (ADR-0005). They show that the demo model relates the three languages, which is all the demo needs.
- Neither the cross-lingual nor the no-diacritics questions come out low: both tags reach everything the set allows at every k. There is no finding to add to ADR-0005's Context, and the dataset was not changed to reach these numbers.

## Results

The reports are the unedited output of `frontdesk eval retrieval`; only each report's title line is demoted from a level-1 to a level-4 heading.

### k = 1

#### Retrieval evaluation — demo-clinic

Embedder: fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 · k = 1 · cases: 8
Mean recall@1: 0.94 · MRR: 1.00

| case | language | recall@1 | RR |
|---|---|---|---|
| parking-en | en | 1.00 | 1.00 |
| first-visit-it | it | 1.00 | 1.00 |
| physio-prepare-it | it | 1.00 | 1.00 |
| physio-prepare-sq-plain | sq | 1.00 | 1.00 |
| payments-sq | sq | 1.00 | 1.00 |
| accessibility-en | en | 1.00 | 1.00 |
| physio-prepare-en | en | 0.50 | 1.00 |
| payments-it | it | 1.00 | 1.00 |

| group | recall | MRR |
|---|---|---|
| lang:en | 0.83 | 1.00 |
| lang:it | 1.00 | 1.00 |
| lang:sq | 1.00 | 1.00 |
| tag:cross-lingual | 0.75 | 1.00 |
| tag:no-diacritics | 1.00 | 1.00 |
| tag:paraphrase | 1.00 | 1.00 |

### k = 3

#### Retrieval evaluation — demo-clinic

Embedder: fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 · k = 3 · cases: 8
Mean recall@3: 1.00 · MRR: 1.00

| case | language | recall@3 | RR |
|---|---|---|---|
| parking-en | en | 1.00 | 1.00 |
| first-visit-it | it | 1.00 | 1.00 |
| physio-prepare-it | it | 1.00 | 1.00 |
| physio-prepare-sq-plain | sq | 1.00 | 1.00 |
| payments-sq | sq | 1.00 | 1.00 |
| accessibility-en | en | 1.00 | 1.00 |
| physio-prepare-en | en | 1.00 | 1.00 |
| payments-it | it | 1.00 | 1.00 |

| group | recall | MRR |
|---|---|---|
| lang:en | 1.00 | 1.00 |
| lang:it | 1.00 | 1.00 |
| lang:sq | 1.00 | 1.00 |
| tag:cross-lingual | 1.00 | 1.00 |
| tag:no-diacritics | 1.00 | 1.00 |
| tag:paraphrase | 1.00 | 1.00 |

### k = 5

#### Retrieval evaluation — demo-clinic

Embedder: fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 · k = 5 · cases: 8
Mean recall@5: 1.00 · MRR: 1.00

| case | language | recall@5 | RR |
|---|---|---|---|
| parking-en | en | 1.00 | 1.00 |
| first-visit-it | it | 1.00 | 1.00 |
| physio-prepare-it | it | 1.00 | 1.00 |
| physio-prepare-sq-plain | sq | 1.00 | 1.00 |
| payments-sq | sq | 1.00 | 1.00 |
| accessibility-en | en | 1.00 | 1.00 |
| physio-prepare-en | en | 1.00 | 1.00 |
| payments-it | it | 1.00 | 1.00 |

| group | recall | MRR |
|---|---|---|
| lang:en | 1.00 | 1.00 |
| lang:it | 1.00 | 1.00 |
| lang:sq | 1.00 | 1.00 |
| tag:cross-lingual | 1.00 | 1.00 |
| tag:no-diacritics | 1.00 | 1.00 |
| tag:paraphrase | 1.00 | 1.00 |
