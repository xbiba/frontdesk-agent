# ADR-0008: One deployment per business

## Status
Accepted (2026-10-06).

## Context
A clinic handles health data; a shop does not. The blueprint asks every tenant to declare an isolation level (TEN-002) and to move between levels without forking the code (TEN-003).

## Decision
One versioned codebase; one deployment per business, selected by one env file, with its own business pack, agent profile, tools, credentials and database. The clinic runs in a dedicated environment (T3); the demo businesses declare T1. Row-level security stays on as a second defence inside each deployment. Integrations are optional and never initialised when disabled.

## Consequences
- More deployments to operate; infrastructure code makes each one a module instance.
- The demo proves it with two deployments: a clinic and a shop with different profiles and tools.

## Exit plan
Every table already carries `business_id` with RLS, so a shared multi-business deployment remains possible.
