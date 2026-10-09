# ADR-0007: Public code, private business configuration

## Status
Accepted (2026-10-06).

## Context
The code is a public portfolio project; real clients' texts, data and credentials must never be public.

## Decision
This repository contains code and synthetic demo businesses only. Real business packs live in a private repository with the same layout and are loaded at deploy time.

## Consequences
- Tests and evaluations in this repository run on synthetic data.
- Any client-specific behaviour must be expressible as configuration.

## Exit plan
Not applicable.
