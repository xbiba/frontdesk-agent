# ADR-0008: One deployment per business

## Status
Accepted (2026-10-06).

## Context
A clinic handles health data; a shop does not. Every business declares an isolation level, and a business can move between levels without forking the code. The levels used here:
- T1: synthetic or public data only. The demo businesses declare it, and `frontdesk knowledge bootstrap-demo` approves a pack by itself only at this level.
- T3: personal or health data of real people, such as a real clinic's. Its knowledge needs a recorded approval, and it runs in a dedicated environment.
- T2 is accepted by the business pack and counts as real data, like T3.

## Decision
One versioned codebase; one deployment per business, selected by one env file, with its own business pack, agent profile, tools, credentials and database. The clinic runs in a dedicated environment (T3); the demo businesses declare T1. Row-level security stays on as a second defence inside each deployment. Integrations are optional and never initialised when disabled.

## Consequences
- More deployments to operate; infrastructure code makes each one a module instance.
- The demo proves it with two deployments: a clinic and a shop with different profiles and tools.

## Exit plan
Every table already carries `business_id` with RLS, so a shared multi-business deployment remains possible.
