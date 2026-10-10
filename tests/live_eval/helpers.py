"""Recorded turns and a manifest for the judging, report and review tests: no model, no database."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from frontdesk.conversation.evaluation import TurnCase
from frontdesk.live_eval.records import CallRow, ClaimRow, DraftRow, RunManifest, TurnRecord
from frontdesk.tenancy.pack import FixedTextKey

AT = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
CONTACT = "Per parlare con una persona, chiama la reception."
TEXTS = {
    FixedTextKey.HANDOFF: CONTACT,
    FixedTextKey.CLINICAL: "Non posso dare consigli medici via messaggio.",
    FixedTextKey.CLARIFICATION: "Puoi dirmi meglio di cosa hai bisogno?",
    FixedTextKey.EMERGENCY: "Se è un'emergenza, chiama subito il numero di emergenza.",
    FixedTextKey.NO_INFORMATION: "Non ho questa informazione.",
}
FIXED_TEXTS = {"it": TEXTS, "en": TEXTS, "sq": TEXTS}


def case(case_id: str, message: str = "Domanda?", **expect: Any) -> TurnCase:
    return TurnCase.model_validate({"id": case_id, "message": message, "expect": expect})


def call(agent: str = "classifier", **fields: Any) -> CallRow:
    values: dict[str, Any] = {
        "agent": agent,
        "model_version": "gemini-3.1-flash-lite",
        "input_tokens": 1_000,
        "output_tokens": 100,
        "thoughts_tokens": 0,
        "finish_reason": "stop",
        "usd": 0.001,
        "seconds": 0.8,
    }
    return CallRow(**{**values, **fields})


def draft(text: str, quote: str) -> DraftRow:
    claim = ClaimRow(text=text, evidence_id="D-0000abcd", quote=quote)
    return DraftRow(action="answer", clarify=None, claims=(claim,))


def record(case_id: str, reply: str, **fields: Any) -> TurnRecord:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "attempt": "a1",
        "case_id": case_id,
        "turn": 1,
        "turns": 1,
        "message": "Domanda?",
        "reply": reply,
        "language": "it",
        "outcome": "replied",
        "rule": "pass",
        "category": "administrative",
        "intent": "general_question",
        "rejection": None,
        "check_reasons": (),
        "sources": (),
        "draft": None,
        "calls": (call(),),
        "seconds": 1.0,
        "at": AT,
    }
    return TurnRecord(**{**values, **fields})


def manifest(**fields: Any) -> RunManifest:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "business_id": "demo-clinic",
        "route": "vertex-eu-gemini-flash-lite",
        "model_name": "gemini-3.1-flash-lite",
        "engine_commit": "abc1234",
        "cases_file": "evals/turns.yaml",
        "cases_digest": "d" * 64,
        "knowledge_version": "kv-1",
        "classifier_prompt": "c" * 12,
        "responder_prompt": "r" * 12,
        "fixed_texts": FIXED_TEXTS,
        "budget_file": None,
        "min_support_similarity": 0.5,
        "min_retrieval_similarity": 0.0,
        "started_at": AT,
    }
    return RunManifest(**{**values, **fields})
