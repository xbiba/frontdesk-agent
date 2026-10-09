# ADR-0003: Google Calendar for bookings

## Status
Accepted (2026-10-07). Implemented in M6.

## Context
Reception manages appointments from one account and needs several doctors bookable in parallel.

## Options considered
- cal.com: with one account, one booking blocks every doctor at that time; the team plan is paid per doctor.
- The clinic back office: clinic-only and not designed for slot search.
- Google Calendar: one reception account with one calendar per doctor.

## Decision
Google Calendar, shared with a service account. Slots are computed by code; Postgres reservations with an exclusion constraint stop double bookings; events stay "to confirm" until reception approves.

## Consequences
- Google Calendar must be the real agenda for doctors bookable online.
- Calendar is an optional integration: a business without it needs no credentials and never initialises it.

## Exit plan
Booking goes through a calendar interface; another backend implements the same interface.
