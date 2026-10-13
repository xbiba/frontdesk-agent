# ADR-0004: A model on Vertex AI in the EU, chosen by evaluation

## Status
Accepted (2026-10-07); amended 2026-10-10 for M3. Live evaluations in M3; first real messages in M4 (shadow mode); tests and the demo use deterministic models.

## Context
Patient messages can contain health data, so provider routes are approved per data class: `patient_derived` for any text derived from a patient's message, original or rewritten, and `evaluation` for synthetic questions and the business's approved public information. In October 2026 the EU quota for Claude Haiku was refused to the project, and Claude Opus 5.5 costs far more than closed-form classification and quoting need.

## Options considered
- Anthropic API: `inference_geo` offers `us` or `global` only.
- Claude Opus 5.5 on Vertex AI `eu`: the first choice; its cost does not fit a receptionist's closed forms.
- Claude Haiku on Vertex AI `eu`: no quota for the project yet.
- Gemini 2.5 Flash-Lite on `europe-west1`: Vertex AI retires it on 2026-10-20.
- Gemini 3.1 Flash-Lite on the `eu` multi-region: generally available since 2026-05-07, retiring "May 7, 2027 or later"; not served on `europe-west1`.

## Decision
The model is the one the M3 live evaluations choose on Vertex AI in the EU; today route `vertex-eu-gemini-flash-lite`: `gemini-3.1-flash-lite` on the `eu` multi-region (`https://aiplatform.eu.rep.googleapis.com`, API `v1`), with minimal thinking and output caps per agent (600 tokens for the classifier, 1,200 for the responder; thinking counts in them). The route is approved for evaluation data only until Vertex AI's data retention for Gemini is checked (spec §18, technical question 2), before M4; until then the composition root refuses it for conversations and `vertex-eu-opus` stays the only route approved for patient-derived text. A fallback (a more capable Gemini on `eu`, or Claude Haiku once its quota arrives) becomes a route only for a task Flash-Lite fails, on the evidence of the evaluations; Haiku, when its quota arrives, runs the same cases.

## Consequences
- Only `frontdesk.model_gateway` imports Google's model libraries (ruff and import-linter), as for Anthropic's.
- Every live client pins its host, location, project and API version, and no environment variable can switch it to recording requests. Its model requests ignore proxy and certificate variables on every operating system: they verify with certifi's bundle through one explicit context. The credentials' token refresh is another path: google-auth reads `HTTPS_PROXY` and `REQUESTS_CA_BUNDLE`, and `GOOGLE_CLOUD_QUOTA_PROJECT` moves the project a call is billed to; only credentials and billing attribution travel it, never message text. A live run refuses to start while one of those variables is set, and pinning that path is an item of M4, before the Linux deployment.
- A live run reserves each request against the phase budget before sending it and records what it cost (`frontdesk.model_gateway.metering`); the embedder experiment does the same for each text it embeds (`frontdesk.model_gateway.embeddings`).
- Claude on Vertex AI (`vertex-eu-opus`) has no server-side refusal fallback, an Anthropic API feature; a refusal, from any route, becomes the fixed reply plus handoff.
- The retirement date of the pinned version is part of its route; a version change is an engine change with its test.
- The results of the M3 evaluations are recorded in [`docs/evals`](../evals/README.md).
- Measured on the demo packs after the tuning of M3 (2026-10-11, [results](../evals/2026-10-m3-live-demo.md)):
  - right answers refused: 0% in en, it and sq, on 1 to 3 document questions per language (the MCU run measures it on at least 10);
  - p95 time per reply: 0.9 to 2.2 s;
  - cost: about 0.00025 EUR per message;
  - M3 spend so far: 0.09 EUR of 10.
  The output caps (classifier 600, responder 1,200) are kept; the largest seen was 58 and 118 tokens, with no reply cut. The MCU run is recorded in the private configuration repository.

## Exit plan
Routes are data in `frontdesk.model_gateway.routes`; a new route is one entry plus its evaluation results, a Haiku route one entry with provider `anthropic-vertex`. A new Vertex AI location also needs an entry in `VERTEX_HOSTS` (`frontdesk.model_gateway.factory`), which pins the host it calls.

## Amendment, 2026-10-12: Gemini 3.8 Flash, hedged requests

The reception agent runs on Gemini 3.8 Flash (`vertex-eu-gemini-flash`, eu multi-region, GA since 2026-09-02, evaluation data only like Flash-Lite until Vertex's data retention is checked). Its replies take about 2 seconds at the median. Some requests stayed stuck for 15 to 36 seconds in the runs of 2026-10-12. Two remedies:

- **Hedged requests** (Albi, live and in production). A request still away after 4.5 seconds gets an identical twin, and the first answer wins (`model_gateway/hedging.py`). A meter records the losing twin at the winner's input and at most its output cap: an identical request that may have been billed, and no failure of the provider.
- **A request timeout per route.** 15 seconds for 3.8 Flash, so that a stuck request is retried within the turn.
- **Shorter waits for the agent** (2026-10-13). The run of 2026-10-13 had turns of 27 to 36 seconds whose model calls took 7 to 19: the rest went to attempts answered with an error status, which no record showed, and to the waits of 2, 4 and 8 seconds between attempts. The agent now waits 1, 2 and 4 seconds, and every attempt is a call of the turn, an unbilled one with its status ("error 429") or "not sent".

The latency criterion of the live runs becomes 90% of replies within 10 seconds and 99% within 20 (Albi). A turn of the agent that uses a tool takes two model calls, the tool's and the reply.
