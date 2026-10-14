# ADR-0001: One Temporal workflow per conversation

## Status
Superseded by ADR-0011 (2026-10-14). Accepted on 2026-10-07; M0 shipped a worker skeleton, removed in M4.

## Context
A conversation waits: for the next message, for reception to confirm a booking, for Meta's 24-hour reply window. Messages can arrive twice or in bursts, and processes crash or redeploy in the middle of a turn.

## Options considered
- A queue plus a database state machine: every wait, retry and timer is code we own.
- Temporal: durable execution with signals, timers and retries.

## Decision
One Temporal workflow per conversation. It deduplicates Meta message IDs, processes messages in order, holds conversation state and waits durably. Payloads are encrypted client-side with a Cloud KMS key.

## Consequences
- A crash resumes the turn instead of losing it.
- Workflow code must stay deterministic; side effects live in activities with idempotency keys.
- One more managed service to run and pay for.

## Exit plan
Workflow logic delegates to plain Python domain functions, so another durable runtime can call the same functions.
