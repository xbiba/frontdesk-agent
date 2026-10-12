# ADR-0009: Facts written by code; document answers carry verified quotes

## Status
Accepted (2026-10-08, after an external review of the M0–M2 plan).

## Context
Checking a generated reply with patterns cannot prove it right. A check on numbers accepts "30 ALL" when the record says "30 EUR", a price stated for the wrong service, a price written in words, or "no prior consultation is needed" against a record that requires one.

## Options considered
- The model writes every reply; code checks it afterwards (numbers, citations, patterns).
- Code writes replies about records; the model writes only answers drawn from documents, and code checks each of their sentences.
- Only verbatim passages from documents are shown.

## Decision
The second option.
- Prices, opening hours, address and contacts, services, doctors and orientation are written by code from the approved records, and the status of an order from the shop's order data, with fixed phrasing in each language, including the lead-in of an orientation and its closing question. No model writes them. An answer ends with the pack's next step (its optional `next_step` text, such as how to book) once per conversation, never after the contact itself; there is no generic closing question, which read as a bot in the review of the first MCU run.
- Questions answered from documents go to the responder, which returns at most four claims. Each claim is one sentence in the person's language, the ID of a document passage and a quote copied from that passage. Code checks that the passage was provided, that the quote is in it word for word, that every number in the sentence is in the quote, that the sentence names no price, that it uses no diagnostic language and that it is close in meaning to its quote (cosine similarity of their embeddings at least `min_support_similarity`).
- Clarifying questions are chosen by the model from a closed list and written by code. The list is the one that each profile's prompt offers (the shop's offers none). A handoff always uses the approved fixed texts.

## Consequences
- The replies people read most often are exact by construction and cost no model call.
- Similarity cannot see a negation ("we accept cards" and "we do not accept cards" are close). The live evaluation measures how often a sentence contradicts its quote, with a judge model, before the threshold changes or a verifier model is added.
- Prices belong in records, not in documents: a document answer that names a price is refused.
- Numbers are compared as values, whether written in digits or in words of the three languages ("fourteen", "quattordici" and "14" are one number). Words that are also articles or verbs ("one", "un", "uno", "una", "sei", "një") count only after a tens word or before a unit of time. A thousands group ("1.000") is compared as written.
- A quote must start and end on whole words and numbers, and no negation may stand among the four words before it in its clause; otherwise the claim is refused as `quote_not_in_source`. The negation words are those of the passage's own language, so the Italian "s'intende" is not the Albanian "s'" and a name such as "Jo" does not negate an English sentence; a passage of unknown language is read with the words of all three. "no" before a number, after a word that takes one ("Order no 12"), is the abbreviation of "number", not a negation.
- Contacts are written by code, so a sentence may not bring its own: a web address, an email address or a markdown link (`](`) in a claim must be copied from the claim's quote, or the claim is refused as `link_not_in_quote`. The replies stay plain text.
- A clarifying question that the profile's prompt does not offer is refused like a failed claim check: the person gets the approved "no information" text and the handoff text, and the turn trace records `clarify_not_offered` among its `check_reasons`.
- Still unseen by code: a sentence that negates its own quote, ordinals and fractions, hundreds and thousands written in several words, a price given without a currency; the live evaluation measures them.

## Exit plan
The claim checks sit behind one function; a verifier model can be added after them if the live evaluation calls for one.
