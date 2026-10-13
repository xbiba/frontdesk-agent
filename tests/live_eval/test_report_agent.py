"""The report of a run the reception agent led counts the replies the check sent back to the model
and the turns that ended in the fixed text (Albi, 2026-10-12)."""

from frontdesk.live_eval.judging import judge
from frontdesk.live_eval.report import render_report
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, call, case, manifest, record

SENT = "sent_back:numbers"


def _report(**fields: object) -> str:
    cases = [case(name, behaviour="answer") for name in ("plain", "mended", "refused", "stuck")]
    records = [
        record("plain", "Apriamo alle 09:00."),
        record("mended", "Costa 40 EUR.", check_reasons=(SENT,)),
        record(
            "refused",
            CONTACT,
            outcome="handoff",
            rejection="reception_error",
            check_reasons=(SENT, SENT),
        ),
        record("stuck", CONTACT, outcome="handoff", rejection="reception_error"),
    ]
    verdicts = judge(cases, records, FIXED_TEXTS, {})
    return render_report(manifest(**fields), verdicts, total_cases=4, budget=None)


def test_an_agent_run_counts_the_replies_sent_back_and_the_fixed_texts() -> None:
    text = _report(reception_prompt="r" * 12)
    assert "## The agent's replies" in text
    assert "- Sent back to the model once, with the reasons: 2 of 4 turns; 1 then passed" in text
    assert "- Ended in the fixed text: 2 turns; 1 after a second refused reply, 1 after a" in text


def test_a_pipeline_run_has_no_such_section() -> None:
    assert "## The agent's replies" not in _report()


def test_an_agent_run_counts_the_requests_the_fallback_answered() -> None:
    # 3.8 Flash falls back to 3.1 Flash-Lite when it fails (Albi, 2026-10-13: stability).
    cases = [case(name, behaviour="answer") for name in ("plain", "spared")]
    records = [
        record(
            "plain", "Apriamo alle 09:00.", calls=(call("agent", model_version="gemini-3.8-flash"),)
        ),
        record(
            "spared",
            "Costa 40 EUR.",
            calls=(
                call("agent", model_version=None, finish_reason="error 504", usd=0.0),
                call("agent", model_version="gemini-3.1-flash-lite"),
            ),
        ),
    ]
    verdicts = judge(cases, records, FIXED_TEXTS, {})
    text = render_report(
        manifest(reception_prompt="r" * 12, model_name="gemini-3.8-flash"),
        verdicts,
        total_cases=2,
        budget=None,
    )
    assert (
        "- Answered by the fallback model (gemini-3.1-flash-lite): 1 of 2 answered requests, "
        "in 1 of 2 turns." in text
    )
