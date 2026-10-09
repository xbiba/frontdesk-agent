# ADR-0004: Claude on Vertex AI, EU region

## Status
Accepted (2026-10-07). Live evaluations from M3; first real messages in M4 (shadow mode); tests and the demo use deterministic models.

## Context
Patient messages can contain health data, so provider routes are approved per data class.

## Options considered
- Anthropic API: `inference_geo` offers `us` or `global` only.
- Claude on Vertex AI in the `eu` multi-region, under the Google Cloud contract.

## Decision
Route `vertex-eu-opus`: `claude-opus-5-5` on Vertex AI region `eu`. The model gateway rejects any route not approved for patient-derived data. Effort starts at `medium` and is tuned by evaluation.

## Consequences
- Model availability in the region must be verified before the M3 live evaluations, and Vertex data retention before M4, when the first real messages arrive.
- Server-side refusal fallbacks are not available on Vertex; refusals become a fixed reply plus handoff.

## Exit plan
Routes are data in `frontdesk.model_gateway.routes`; a new approved route is one entry plus its evaluation results. A route in a new Vertex region also needs an entry in `VERTEX_ENDPOINTS` (`frontdesk.model_gateway.factory`), which pins the endpoint it calls.
