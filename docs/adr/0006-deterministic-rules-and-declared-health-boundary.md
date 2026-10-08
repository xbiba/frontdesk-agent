# ADR-0006: Deterministic rules and a declared health-data boundary

## Status
Accepted (2026-10-06, option B of review finding F3).

## Context
Keyword rules cannot prove that an unrecognised message is harmless. Refusing everything they miss makes the assistant useless on real messages.

## Options considered
- A — strict boundary: only rule-recognised administrative messages leave the system.
- B — declared boundary: patient text may contain health data and is processed only on approved EU routes, read by the model only to understand and route it.

## Decision
B. Deterministic rules run first and stop emergencies, explicit requests for medical advice and personal data. A classifier with closed vocabularies handles the rest; code validates its output and builds every query. When the rules miss an emergency, the classifier can answer with the closed category `emergency`, which takes the same fixed path as the rules: the approved emergency text and the reception's contact. The model never writes that text, never judges severity and cannot lift a block set by the rules. The responder sees only validated fields, never the original text, and never discusses health. Orientation (which service handles which problem) comes only from a map approved by the clinic's doctors.

## Consequences
- The privacy notice must declare the processing; the clinic approves it.
- Telemetry is metadata-only by default.
- Fixed texts never promise an action that has not happened: until the reception notification exists (M3), handoff texts give the reception's contact instead of saying that someone will get in touch.

## Exit plan
Switching to A is a configuration of the routing step: unclassified messages get the fixed clarification.
