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

Amendment of 2026-10-12 for M5, booking from the chat (Albi: the clinic agrees to the collection of a name and a phone number, and booking is immediate for now):
- **The steps.** Code guides every step: the patient picks a numbered slot, then gives a name and a phone, sees a summary and answers yes. The model never sees the name or the phone. The calendar is read again before the hold and before the write.
- **Holds.** Holds in Postgres (migration 0005) carry an exclusion constraint, so two patients never get one slot.
- **The write.** The appointment is written with an id derived from the hold, so a repeated request or a lost answer never makes two. A 409 or a lost answer is checked by reading the id back.
- **Retention.** Name and phone live only on the hold while the patient books, at most 15 minutes. Booking or cancelling drops them, and an expired hold is deleted by the next booking and by `frontdesk bookings purge`. The appointment then lives in the clinic's calendar.
- **Later.** Reception confirmation is kept for later, behind the pack's setting.

Amendment of 2026-10-12, cal.com (Albi): the clinic keeps each doctor's availability in cal.com, which checks the clinic's Google Calendar for conflicts and writes the appointments there.
- **A scheduler behind one protocol.**
  - `CalendarScheduler` computes the slots from a calendar's blocks and busy time, for Google Calendar and for the evaluations' fixture.
  - `CalcomScheduler` reads an event type's free slots and books through cal.com's API.
  - The booking steps are the same for both.
- **What cal.com receives.** The patient's name and phone, in international form, and our hold's reference. A lost answer is checked by that reference.
- **Its constraints.**
  - With one cal.com user, every doctor's event type shares one set of conflicts: a second real doctor needs a cal.com user of their own.
  - cal.com keeps the bookings in its own store, in the US for api.cal.com. The clinic's notice and a data-processing agreement cover it.

## Consequences
- Google Calendar must be the real agenda for doctors bookable online.
- Calendar is an optional integration: a business without it needs no credentials and never initialises it.

## Exit plan
Booking goes through a calendar interface; another backend implements the same interface.
