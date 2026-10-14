# ADR-0010: One engine, channel adapters, a capability policy per channel

## Status
Accepted (2026-10-09).

## Context
The first channel is Instagram direct messages. The owner also wants a chat on the business website later, with fewer capabilities, for example no bookings. The conversation engine (`frontdesk.conversation.turn.handle_message`) already takes plain text, a metadata-only state and a runtime. It knows nothing about Instagram: the local chat command is a second front end on the same engine. Capabilities are declared per business (`Features` in the business pack), not per channel.

## Options considered
- One engine, with channel adapters and a policy per channel that can only switch capabilities off.
- A separate assistant per channel. This duplicates the rules, the facts written by code and the claim checks, and lets them drift apart.

## Decision
- Channels are adapters. An adapter does three things:
  - it verifies and receives inbound messages;
  - it maps them to a business and a conversation;
  - it sends replies.
  It also holds the channel's own texts and limits. It never classifies, retrieves, composes or checks a reply: the shared engine does that.
- Each channel has a policy that narrows the business features. The capabilities of a turn are the business features AND the channel policy. A policy cannot enable what the business disables.
- A website FAQ mode is planned as a later channel:
  - anonymous sessions;
  - bookings and order lookups switched off. A per-conversation attempt limit cannot bound sessions that anyone can recreate.
  - rate limits per session and per network address, with bot protection;
  - its own disclosure and handoff texts;
  - a privacy notice and a session cookie.
- The adapter interface and the channel policy are built together with the Instagram adapter. The website then adds one adapter and one policy, not a fork.

## Consequences
- The turn pipeline already checks features when it validates the classifier output. The composition root will pass it the effective features of the channel.
- Logs and spans gain a channel attribute. Conversation ids stay opaque and bounded: the order-attempt key column holds 64 characters.
- Fixed texts that say how to reach a person become per channel.

## Exit plan
If a channel needs behaviour the engine cannot express through features, add a feature to the engine. Never branch on the channel name inside the engine.

## Amendment (2026-10-14, M4)
- The policy is `ChannelPolicy` in the business pack (`channels.instagram`), with switches for chat bookings, order status and orientation. The deployment builds the turn from a copy of the pack with those features off (`frontdesk.channels.policy.narrow`); a policy cannot turn anything on.
- The adapter is two parts: a webhook reader (signature, events) and a sender (`frontdesk.channels.messages`).
- Fixed texts per channel are postponed: Instagram uses the business's texts. They become per channel with the website.
