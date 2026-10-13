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
- **A request timeout per route.** 15 seconds for 3.8 Flash, so that a stuck request is retried within the turn. The SDK sends the timeout to Vertex (`X-Server-Timeout`), which answers 504 when it runs out: at about 12.5 seconds with 15. On 2026-10-13, 25 seconds was tried, because 34 504s came at 12.1 to 12.9 seconds while the slowest answers took 13.0. With 25, the 504s came at 22: the requests were stuck, not slow, and the turns got longer. It stays at 15.
- **Shorter waits for the agent** (2026-10-13). The run of 2026-10-13 had turns of 27 to 36 seconds whose model calls took 7 to 19: the rest went to attempts answered with an error status, which no record showed, and to the waits of 2, 4 and 8 seconds between attempts. The agent now waits 1, 2 and 4 seconds, and every attempt is a call of the turn, an unbilled one with its status ("error 429") or "not sent".

The latency criterion of the live runs becomes 90% of replies within 10 seconds and 99% within 20 (Albi). A turn of the agent that uses a tool takes two model calls, the tool's and the reply.

## Amendment, 2026-10-13: a fallback model

Vertex's eu multi-region left requests of 3.8 Flash hanging until a 504 for whole windows, twins included: the run of 2026-10-13 was stopped twelve times. Albi asked for a twin in another European region, or a fallback model, before Instagram.

- **No other region.** A probe of 2026-10-13 found 3.8 Flash only on the `eu` multi-region (404 on europe-west1, -west3, -west4, -west8, -west9, -north1, -central2, -southwest1; 501 on -west10; 400 on -west12). 2.5 Flash answers on europe-west1, -west4, -west8 and -west9, but retires on 2026-10-20, with 2.5 Pro and 2.5 Flash-Lite.
- **3.1 Flash-Lite as fallback.** It is on the `eu` multi-region and retires "May 7, 2027 or later". The route of 3.8 Flash names it (`fallback`).
  - The first model makes one attempt, without the meter's retries. When it fails with an error status, a timeout or a broken connection, the request goes to the fallback at once (`model_gateway/hedging.py`).
  - So do the next requests of the same turn, so that the turn does not wait out the stuck window again. The next turn tries 3.8 Flash again. The twin after 4.5 seconds stays on 3.8 Flash.
  - A request still unanswered after 6 seconds, twin included, also goes to the fallback, and the first answer wins (Albi, 2026-10-13: in `runs/2026-10-13-agent-8`, 19 of the 36 turns over 10 seconds had waited for a 504). Slowness is no failure: the turn keeps the first model.
  - A rest of 120 seconds was measured first (`runs/2026-10-13-agent-6`). After 3 failures of 3.8 Flash in an 11-minute run, Flash-Lite answered 204 of 293 requests, and 11 cases failed on its answers, against 1 before. The time criteria passed (95.9% within 10 s, 99.5% within 20). So the rest is the turn's only.
  - The report counts the requests the fallback answered.

## Amendment, 2026-10-13: guards on the model in production

Albi asked for these before Instagram:
- **A daily spend cap.** `FRONTDESK_DAILY_CAP_EUR` is priced with the routes' prices of `FRONTDESK_PRICES_FILE`, in the live runs' budget format.
  - Each business-local day's spend is kept in Postgres (`daily_spend`, migration 0006).
  - Past the cap no request is sent, and the patient gets reception's contact (rule `spend_cap`), until the next day.
  - The logs carry `spend.alert` at 80% of the cap and `spend.cap_reached` at the cap, for an alert policy.
- **A switch.** `FRONTDESK_MODEL_OFF` turns the model off: the rules still answer, and every other message gets reception's contact (rule `model_off`).
- **Ten messages a minute per conversation at most.** Past them, a short reply and no model call (rule `rate_limit`). A conversation holds one time at a time, for 15 minutes at most (`booking/holds.py`).
