# M3 live results on the demo packs (synthetic data only)

- Date: 2026-10-11.
- Route: `vertex-eu-gemini-flash-lite`, Gemini 3.1 Flash-Lite on the Vertex AI `eu` multi-region, minimal thinking.
- Engine: after the tuning of M3 Task 14. The clinic and document runs used 6de6a59; the shop and MCU-case runs used 72b0f98, which adds the business contact to document answers.
- The rate of right answers refused rests on few document questions on the demo packs, 1 to 3 per language. It shows the checks did not refuse those answers. It is not yet a measure: the MCU run measures it on at least 10 questions per language.
- Review: an independent reviewer agent (Opus) labelled every answer the model wrote, each label with its reason. The labels, the runs and the spend ledger are kept in the private configuration repository (`runs/`).
- Live spend of M3 when these runs were made: 0.09 EUR of the 10 EUR cap.

These runs carry synthetic data only. The MCU run, on the clinic's approved information, is recorded in the private repository and summarised here when it is done.

## Demo clinic (56 demo cases, `runs/2026-10-11-demo-clinic-3`)

### Exit criteria (M3 design §1)

| criterion | threshold | value | status |
|---|---|---|---|
| Missed emergencies | 0 | 0 of 10 | PASS |
| Invented facts that passed the checks | 0 | 0 | PASS |
| Right answers refused (en) | at most 20% | 0% (0/1) | PASS |
| Right answers refused (it) | at most 20% | 0% (0/1) | PASS |
| Right answers refused (sq) | at most 20% | 0% (0/1) | PASS |
| Replies within 8 seconds | at least 95% | 100% (p95 1.6 s) | PASS |
| Critical cases failed | 0 | 0 of 32 | PASS |
| Live spend of M3 | at most 10.00 EUR | 0.08 EUR | PASS |

### By language

| language | cases | passed | critical failed | missed emergencies | right answers refused | wrong answers passed | invented facts | to review | handoffs by failure | p50 s | p95 s | within 8 s | EUR per message | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 13 | 13 | 0 | 0 | 0/1 | 0/9 | 0 | 0 | 0 | 0.7 | 1.5 | 100% | 0.00022 | 0.0029 |
| it | 29 | 29 | 0 | 0 | 0/1 | 0/9 | 0 | 0 | 0 | 0.7 | 1.9 | 100% | 0.00025 | 0.0080 |
| sq | 14 | 13 | 0 | 0 | 0/1 | 0/8 | 0 | 0 | 0 | 0.7 | 1.6 | 100% | 0.00025 | 0.0034 |
| all | 56 | 55 | 0 | 0 | 0/3 | 0/26 | 0 | 0 | 0 | 0.7 | 1.6 | 100% | 0.00024 | 0.0143 |

### Model requests

| agent | requests | max output tokens | p95 output tokens | cut at the limit | thinking tokens |
|---|---|---|---|---|---|
| classifier | 43 | 49 | 47 | 0 | 0 |
| responder | 7 | 91 | 91 | 0 | 0 |

## Demo shop (23 demo cases, `runs/2026-10-11-demo-shop-5`)

### Exit criteria (M3 design §1)

| criterion | threshold | value | status |
|---|---|---|---|
| Missed emergencies | 0 | 0 of 1 | PASS |
| Invented facts that passed the checks | 0 | 0 | PASS |
| Right answers refused (en) | at most 20% | 0% (0/1) | PASS |
| Right answers refused (it) | at most 20% | 0% (0/2) | PASS |
| Replies within 8 seconds | at least 95% | 100% (p95 2.2 s) | PASS |
| Critical cases failed | 0 | 0 of 9 | PASS |
| Live spend of M3 | at most 10.00 EUR | 0.09 EUR | PASS |

### By language

| language | cases | passed | critical failed | missed emergencies | right answers refused | wrong answers passed | invented facts | to review | handoffs by failure | p50 s | p95 s | within 8 s | EUR per message | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 5 | 5 | 0 | 0 | 0/1 | 0/5 | 0 | 0 | 0 | 0.8 | 1.7 | 100% | 0.00024 | 0.0012 |
| it | 18 | 18 | 0 | 0 | 0/2 | 0/6 | 0 | 0 | 0 | 0.7 | 2.2 | 100% | 0.00021 | 0.0047 |
| all | 23 | 23 | 0 | 0 | 0/3 | 0/11 | 0 | 0 | 0 | 0.7 | 2.2 | 100% | 0.00022 | 0.0059 |

### Model requests

| agent | requests | max output tokens | p95 output tokens | cut at the limit | thinking tokens |
|---|---|---|---|---|---|
| classifier | 25 | 40 | 38 | 0 | 0 |
| responder | 6 | 176 | 176 | 0 | 0 |

## MCU cases that need no clinic data, on the demo clinic (45 cases, `runs/2026-10-11-mcu-cases-on-demo-clinic-2`)

### Exit criteria (M3 design §1)

| criterion | threshold | value | status |
|---|---|---|---|
| Missed emergencies | 0 | 0 of 21 | PASS |
| Invented facts that passed the checks | 0 | 0 | PENDING |
| Right answers refused (en) | at most 20% | 0% (0/0) | PENDING |
| Right answers refused (it) | at most 20% | 0% (0/0) | PENDING |
| Right answers refused (sq) | at most 20% | 0% (0/0) | PENDING |
| Replies within 8 seconds | at least 95% | 100% (p95 0.9 s) | PASS |
| Critical cases failed | 0 | 0 of 32 | PASS |
| Live spend of M3 | at most 10.00 EUR | 0.09 EUR | PASS |

### By language

| language | cases | passed | critical failed | missed emergencies | right answers refused | wrong answers passed | invented facts | to review | handoffs by failure | p50 s | p95 s | within 8 s | EUR per message | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 15 | 15 | 0 | 0 | 0/0 | 0/2 | 0 | 0 | 0 | 0.0 | 0.7 | 100% | 0.00013 | 0.0019 |
| it | 16 | 16 | 0 | 0 | 0/0 | 0/3 | 0 | 0 | 0 | 0.0 | 1.9 | 100% | 0.00010 | 0.0016 |
| sq | 14 | 14 | 0 | 0 | 0/0 | 0/2 | 0 | 0 | 0 | 0.0 | 0.9 | 100% | 0.00010 | 0.0014 |
| all | 45 | 45 | 0 | 0 | 0/0 | 0/7 | 0 | 0 | 0 | 0.0 | 0.9 | 100% | 0.00011 | 0.0049 |

### Model requests

| agent | requests | max output tokens | p95 output tokens | cut at the limit | thinking tokens |
|---|---|---|---|---|---|
| classifier | 18 | 40 | 40 | 0 | 0 |

## Document questions on the demo clinic (9 cases, 3 per language, `runs/2026-10-11-demo-documents-2`)

### Exit criteria (M3 design §1)

| criterion | threshold | value | status |
|---|---|---|---|
| Missed emergencies | 0 | 0 of 0 | PENDING |
| Invented facts that passed the checks | 0 | 0 | PASS |
| Right answers refused (en) | at most 20% | 0% (0/3) | PASS |
| Right answers refused (it) | at most 20% | 0% (0/3) | PASS |
| Right answers refused (sq) | at most 20% | 0% (0/3) | PASS |
| Replies within 8 seconds | at least 95% | 100% (p95 1.9 s) | PASS |
| Critical cases failed | 0 | 0 of 0 | PENDING |
| Live spend of M3 | at most 10.00 EUR | 0.08 EUR | PASS |

### By language

| language | cases | passed | critical failed | missed emergencies | right answers refused | wrong answers passed | invented facts | to review | handoffs by failure | p50 s | p95 s | within 8 s | EUR per message | EUR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en | 3 | 3 | 0 | 0 | 0/3 | 0/3 | 0 | 0 | 0 | 1.5 | 1.6 | 100% | 0.00061 | 0.0018 |
| it | 3 | 2 | 0 | 0 | 0/3 | 1/3 | 0 | 0 | 0 | 1.7 | 1.8 | 100% | 0.00054 | 0.0016 |
| sq | 3 | 3 | 0 | 0 | 0/3 | 0/3 | 0 | 0 | 0 | 1.7 | 1.9 | 100% | 0.00068 | 0.0021 |
| all | 9 | 8 | 0 | 0 | 0/9 | 1/9 | 0 | 0 | 0 | 1.7 | 1.9 | 100% | 0.00061 | 0.0055 |

### Model requests

| agent | requests | max output tokens | p95 output tokens | cut at the limit | thinking tokens |
|---|---|---|---|---|---|
| classifier | 9 | 58 | 58 | 0 | 0 |
| responder | 8 | 118 | 118 | 0 | 0 |
