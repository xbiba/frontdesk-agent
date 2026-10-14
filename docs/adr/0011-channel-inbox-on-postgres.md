# ADR-0011: The channel inbox on Postgres; Temporal set aside

## Status
Accepted (2026-10-14). Supersedes ADR-0001.

## Context
ADR-0001 chose one Temporal workflow per conversation and left the choice to the plan of M4, with the cost of Temporal Cloud and the timers Instagram needs (spec §17). M4 brings the first channel, Instagram direct messages.
- Temporal Cloud starts at 100 $ a month (Essentials). The model costs about 0.014 € a message.
- What Instagram waits for are deadlines, not timers: the staff's pause (24 hours without messages), Meta's 24-hour reply window, the 15-minute hold, 7 days of silence. Each one is checked when the next message comes or just before a send; none has to fire by itself.

## Options considered
- Temporal Cloud: durable execution, signals and timers; a managed service and a payload codec with a KMS key to run, and the cost above.
- Self-hosted Temporal: the same model, and a cluster to operate.
- Postgres as the inbox and Cloud Tasks to call the worker: Postgres is already there, and Cloud Tasks is free at this volume.

## Decision
- Postgres keeps a channel's conversations and messages (migration 0007). A message is stored once, by its channel ID. A conversation runs one turn at a time under a lease. Every reply is recorded before it is sent, and sent at most once.
- The webhook stores and commits, then asks Cloud Tasks to call the private worker 3 seconds later, so a burst becomes one turn. Cloud Tasks retries a worker that fails; a sweep every 10 minutes gives back what a lost task left.
- The engine's turn does not change: it takes text and a state and returns text and a new state (ADR-0010).
- The Temporal worker skeleton and the dependency are removed.

## Consequences
- No timer fires by itself. A staff pause ends when the next message finds it over; a conversation's state starts over when its next turn finds it older than 7 days, and a nightly purge removes it anyway.
- Message text lives in Postgres until the turn that reads it, then only in the masked conversation memory (spec §12).
- Delivery is at most once: a send that may have reached Meta is never repeated. It is logged as an error for a person to check.

## Exit plan
The inbox calls the engine through one function (text and state in, result and state out). Temporal or another durable runtime can take its place when timers that fire by themselves arrive (approvals that expire on WhatsApp, reminders), or when many businesses share one deployment pipeline.
