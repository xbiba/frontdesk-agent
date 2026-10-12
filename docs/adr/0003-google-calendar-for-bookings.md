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
- **Two listings only** (security review of 2026-10-12):
  - **Blocks.** The events whose text holds the marker, with id, start, end, title, status, type and `organizer(self)`. A block is an ordinary event organised by the calendar itself, so an invitation cannot open hours. Any other title is dropped as soon as it arrives.
  - **Busy time.** Every other event, with id, start, end, status and type only. No title and no notes are requested. Every event that is not a block is busy, whatever its "Show as".
  The calendar is shared with "See all event details", raised to "Make changes to events" when requests arrive.
- **The booking reply.** It is written by code: the first free slots, then the pack's next step, which says how to book. When the calendar cannot be read, or nothing is free, the reply says how to book and never invents a time.
- **Confirmation.** Reception confirms by default; the pack may choose immediate booking instead.

## Consequences
- Google Calendar must be the real agenda for doctors bookable online.
- Calendar is an optional integration: a business without it needs no credentials and never initialises it.

## Exit plan
Booking goes through a calendar interface; another backend implements the same interface.
