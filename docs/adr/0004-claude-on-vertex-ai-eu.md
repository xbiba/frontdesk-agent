# ADR-0004: Claude on Vertex AI, EU region

## Status
Accepted (2026-10-07). Live use from M5; tests and the demo use deterministic models.

## Context
Patient messages can contain health data. The blueprint requires approved provider routes per data class (§12).

## Options considered
- Anthropic API: `inference_geo` offers `us` or `global` only.
- Claude on Vertex AI in the `eu` multi-region, under the Google Cloud contract.

## Decision
Route `vertex-eu-opus`: `claude-opus-5-5` on Vertex AI region `eu`. The model gateway rejects any route not approved for patient-derived data. Effort starts at `medium` and is tuned by evaluation.

## Consequences
- Availability of the model in the region and Vertex's data-retention settings must be verified before M5.
- Server-side refusal fallbacks are not available on Vertex; refusals become a fixed reply plus handoff.

## Exit plan
Routes are data in `frontdesk.model_gateway.routes`; a new approved route is one entry plus its evaluation results.
