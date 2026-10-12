# ADR-0003: Google Calendar for bookings

## Status
Accepted (2026-10-07). Amended on 2026-10-12 for M4, where the slots arrive, and M5, where the booking requests arrive (the order Albi set after M3).

## Context
Reception manages appointments from one account and needs several doctors bookable in parallel.

## Options considered
- cal.com: with one account, one booking blocks every doctor at that time; the team plan is paid per doctor.
- The clinic back office: clinic-only and not designed for slot search.
- Google Calendar: one reception account with one calendar per doctor.

## Decision
Google Calendar, shared with a service account. Slots are computed by code; Postgres reservations with an exclusion constraint stop double bookings; events stay "to confirm" until reception approves.

Amendment of 2026-10-12, from the bookings design approved that day:
- **One calendar per doctor.** It holds the appointments and the doctor's hours. The hours are events titled with the pack's marker word, "Disponibile" by default, and reception draws them every month. Two calendars per doctor were judged confusing for reception.
- **Two reads only:**
  - the events whose text holds the marker (`events.list` with `q` and `fields` limited to start, end, title, transparency and status);
  - the busy intervals (`freeBusy`).
  Other titles are dropped as soon as they arrive, and appointment notes are never requested. The calendar is shared with "See all event details", raised to "Make changes to events" when requests arrive.
- **Confirmation.** Reception confirms by default; the pack may choose immediate booking instead.

## Consequences
- Google Calendar must be the real agenda for doctors bookable online.
- Calendar is an optional integration: a business without it needs no credentials and never initialises it.

## Exit plan
Booking goes through a calendar interface; another backend implements the same interface.
