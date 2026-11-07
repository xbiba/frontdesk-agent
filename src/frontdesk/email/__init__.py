"""Reply drafts for a business's mailbox (email drafts, 2026-11-07).

Every few minutes a pass reads the new conversations of the mailbox, sets aside newsletters,
automatic messages and conversations the staff has answered, and writes a draft reply in the
conversation for each real question, from the business's approved information only. A person reads,
corrects and sends it: nothing here ever sends a message. No message text is kept outside the
mailbox: the engine's table holds message IDs, outcomes, dates and costs.
"""
