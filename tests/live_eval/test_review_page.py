"""What the review page shows for each kind of case, and how it counts them."""

from __future__ import annotations

import re

from frontdesk.live_eval.judging import Label, judge
from frontdesk.live_eval.records import ClaimRow, DraftRow
from frontdesk.live_eval.review import render_review_page, review_cases
from tests.live_eval.helpers import CONTACT, FIXED_TEXTS, TEXTS, case, draft, manifest, record
from tests.live_eval.report_scenario import CASES, verdicts

REFUSED_REPLY = f"{TEXTS['no_information']}\n\n{CONTACT}"


def sections(page: str) -> dict[str, str]:
    """The text of each case's section, by case id."""
    found = re.findall(r'(<section data-case="([^"]+)">.*?</section>)', page, flags=re.DOTALL)
    return {case_id: body for body, case_id in found}


def test_the_refused_and_the_dubious_cases_of_a_run_are_reviewed_in_order() -> None:
    assert [v.case.id for v in review_cases(verdicts())] == ["hours", "refused"]


def test_a_run_with_nothing_to_review_has_an_empty_page() -> None:
    answered = judge(CASES[:1], [verdicts()[0].turns[0]], FIXED_TEXTS, {})
    assert review_cases(answered) == ()
    page = render_review_page(manifest(), answered, {}, run_dir="runs/run-1")
    assert "0 cases to review" in page
    assert "<section" not in page


def test_the_page_counts_the_cases_and_names_the_run_and_the_command() -> None:
    page = render_review_page(manifest(), verdicts(), {}, run_dir="runs/run-1")
    assert "<title>Review: run-1</title>" in page
    assert "<h1>Review: run-1</h1>" in page
    assert "2 cases to review" in page
    assert "<code>frontdesk eval report --run-dir runs/run-1</code>" in page
    assert page.startswith('<!doctype html>\n<html lang="en">')
    assert page.rstrip().endswith("</html>")


def test_the_run_directory_and_the_run_name_are_escaped() -> None:
    other = manifest(run_id='run <1> & "2"')
    page = render_review_page(other, verdicts(), {}, run_dir='runs/<run> & "1"')
    assert "<title>Review: run &lt;1&gt; &amp; &quot;2&quot;</title>" in page
    assert "<h1>Review: run &lt;1&gt; &amp; &quot;2&quot;</h1>" in page
    assert "--run-dir runs/&lt;run&gt; &amp; &quot;1&quot;</code>" in page
    assert "<run>" not in page


def test_the_page_can_save_the_labels() -> None:
    page = render_review_page(manifest(), verdicts(), {}, run_dir="runs/run-1")
    assert "function labels()" in page
    assert "function save()" in page
    assert 'link.download = "labels.yaml"' in page
    assert 'onclick="save()"' in page


def test_a_refused_case_shows_the_question_the_draft_and_why_it_was_refused() -> None:
    page = render_review_page(manifest(), verdicts(), {}, run_dir="runs/run-1")
    refused = sections(page)["refused"]
    assert "<h2>refused (it)</h2>" in refused
    assert "<h3>Question</h3><pre>Domanda?</pre>" in refused
    assert "Behaviour: answer; must say: sei; must not say: -" in refused
    assert f"<h3>Reply sent</h3><pre>{REFUSED_REPLY}</pre>" in refused
    assert "<h3>Draft refused by the checks</h3>" in refused
    assert (
        "<pre>Ci sono sei posti.</pre>quote (D-0000abcd):<pre>sei posti gratuiti</pre>" in refused
    )
    assert "<p>Why: quote_not_in_source</p>" in refused


def test_a_refused_case_asks_only_whether_the_refusal_was_right() -> None:
    refused = sections(render_review_page(manifest(), verdicts(), {}, run_dir="d"))["refused"]
    assert (
        '<p>Refusal: <label><input type="radio" name="refusal-refused" value="right"> '
        'refusal right</label> <label><input type="radio" name="refusal-refused" value="wrong"> '
        "refusal wrong</label></p>"
    ) in refused
    assert 'name="answer-' not in refused
    assert "invented-" not in refused


def test_a_dubious_case_asks_whether_the_answer_was_right_and_whether_it_invented_a_fact() -> None:
    hours = sections(render_review_page(manifest(), verdicts(), {}, run_dir="d"))["hours"]
    assert "<h3>Reply sent</h3><pre>Apriamo alle 09:00.</pre>" in hours
    assert "Behaviour: answer; must say: 08:00; must not say: -" in hours
    assert (
        '<p>Answer: <label><input type="radio" name="answer-hours" value="right"> answer right'
        '</label> <label><input type="radio" name="answer-hours" value="wrong"> answer wrong'
        "</label></p>"
    ) in hours
    assert '<input type="checkbox" name="invented-hours">' in hours
    assert "a price, time, contact, number or negation nobody approved" in hours
    assert "Draft refused" not in hours
    assert "refusal-" not in hours


def test_the_labels_given_are_shown_as_chosen() -> None:
    labels = {
        "refused": Label(refusal="right"),
        "hours": Label(answer="wrong", invented_fact=True, note="line one\nline <two>"),
    }
    found = sections(render_review_page(manifest(), verdicts(), labels, run_dir="d"))
    assert 'name="refusal-refused" value="right" checked' in found["refused"]
    assert 'name="refusal-refused" value="wrong" checked' not in found["refused"]
    assert 'name="answer-hours" value="wrong" checked' in found["hours"]
    assert 'name="answer-hours" value="right" checked' not in found["hours"]
    assert '<input type="checkbox" name="invented-hours" checked>' in found["hours"]
    assert "<textarea rows=2>line one\nline &lt;two&gt;</textarea>" in found["hours"]


def test_no_label_means_nothing_is_chosen_and_the_note_is_empty() -> None:
    found = sections(render_review_page(manifest(), verdicts(), {}, run_dir="d"))
    for body in found.values():
        assert "checked" not in body
        assert "<p>Note</p><textarea rows=2></textarea>" in body


def test_a_label_without_a_note_gives_an_empty_note() -> None:
    labels = {"hours": Label(answer="right")}
    hours = sections(render_review_page(manifest(), verdicts(), labels, run_dir="d"))["hours"]
    assert "<textarea rows=2></textarea>" in hours


def test_a_refusal_with_no_reason_recorded_says_so() -> None:
    wanted = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
    bare = record(
        "parking",
        REFUSED_REPLY,
        outcome="handoff",
        draft=draft("six spaces", "six free parking spaces"),
    )
    [verdict] = judge([wanted], [bare], FIXED_TEXTS, {})
    page = render_review_page(manifest(), [verdict], {}, run_dir="d")
    assert "<p>Why: -</p>" in page


def test_several_reasons_are_listed_and_every_claim_of_the_draft_is_shown() -> None:
    wanted = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
    claims = (
        ClaimRow(text="six spaces", evidence_id="D-0000abcd", quote="six free parking spaces"),
        ClaimRow(text="and a roof", evidence_id="D-0000ef01", quote="covered"),
    )
    sent = record(
        "parking",
        REFUSED_REPLY,
        outcome="handoff",
        check_reasons=("quote_not_in_source", "unsupported_claim"),
        draft=DraftRow(action="answer", clarify=None, claims=claims),
    )
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    page = render_review_page(manifest(), [verdict], {}, run_dir="d")
    assert "<p>Why: quote_not_in_source, unsupported_claim</p>" in page
    assert page.count('<div class="claim">') == 2
    assert "<pre>and a roof</pre>quote (D-0000ef01):<pre>covered</pre>" in page


def test_every_turn_of_a_conversation_is_shown_as_a_question() -> None:
    conversation = case("two", behaviour="answer", contains=["six"], sources=["parking-en"])
    conversation = conversation.model_copy(update={"message": None, "messages": ("uno", "due")})
    first = record("two", "ok", turn=1, turns=2, message="uno")
    last = record(
        "two",
        "Ci sono sei posti, aperti 24 ore.",
        turn=2,
        turns=2,
        message="due",
        sources=("parking-en",),
    )
    [verdict] = judge([conversation], [first, last], FIXED_TEXTS, {})
    two = sections(render_review_page(manifest(), [verdict], {}, run_dir="d"))["two"]
    assert "<h3>Question</h3><pre>uno</pre><pre>due</pre>" in two
    assert "<h3>Reply sent</h3><pre>Ci sono sei posti, aperti 24 ore.</pre>" in two


def test_what_must_not_be_said_is_shown() -> None:
    wanted = case(
        "careful",
        behaviour="answer",
        contains=["six", "free"],
        not_contains=["EUR", "paid"],
        sources=["parking-en"],
    )
    sent = record("careful", "There are six free spaces, open 24 hours.", sources=("parking-en",))
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    careful = sections(render_review_page(manifest(), [verdict], {}, run_dir="d"))["careful"]
    assert "Behaviour: answer; must say: six; free; must not say: EUR; paid" in careful


def test_an_expectation_with_nothing_to_say_shows_dashes() -> None:
    # Only an answer ahead of a notice reaches the page from a case that names no behaviour.
    open_case = case("open", "Ho la febbre. C'è parcheggio?", sources=["parking-it"])
    reply = f"Ci sono 6 posti.\n\n{TEXTS['clinical']}\n\n{CONTACT}"
    sent = record("open", reply, outcome="handoff", category="clinical", sources=("parking-it",))
    [verdict] = judge([open_case], [sent], FIXED_TEXTS, {})
    assert verdict.dubious
    body = sections(render_review_page(manifest(), [verdict], {}, run_dir="d"))["open"]
    assert "<p>Behaviour: -; must say: -; must not say: -</p>" in body


def test_an_answer_ahead_of_a_notice_is_reviewed_with_the_whole_reply() -> None:
    mixed = case(
        "mixed",
        "Ho la febbre. C'è parcheggio?",
        behaviour="clinical",
        contains=["6 posti", "Non posso dare consigli medici"],
        sources=["parking-it"],
    )
    reply = f"Ci sono 6 posti, aperti 24 ore.\n\n{TEXTS['clinical']}\n\n{CONTACT}"
    sent = record("mixed", reply, outcome="handoff", category="clinical", sources=("parking-it",))
    found = judge([mixed], [sent], FIXED_TEXTS, {})
    page = render_review_page(manifest(), found, {}, run_dir="d")
    body = sections(page)["mixed"]
    assert f"<h3>Reply sent</h3><pre>{reply}</pre>" in body
    assert 'name="answer-mixed" value="wrong"' in body
    assert 'name="invented-mixed"' in body
    assert "refusal-" not in body


def test_the_reply_and_what_the_case_expects_are_escaped_too() -> None:
    wanted = case(
        "markup",
        behaviour="answer",
        contains=["<i>six</i> & more"],
        not_contains=["<script>"],
        sources=["parking-en"],
    )
    sent = record("markup", "<b>six</b> & 24 hours", sources=("parking-en",))
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    body = sections(render_review_page(manifest(), [verdict], {}, run_dir="d"))["markup"]
    assert "<h3>Reply sent</h3><pre>&lt;b&gt;six&lt;/b&gt; &amp; 24 hours</pre>" in body
    assert "must say: &lt;i&gt;six&lt;/i&gt; &amp; more; must not say: &lt;script&gt;" in body
    assert "<b>" not in body
    assert "<i>" not in body
    assert "<script>" not in body


def test_every_part_of_a_refused_draft_is_escaped() -> None:
    wanted = case("parking", behaviour="answer", contains=["six"], sources=["parking-en"])
    markup = DraftRow(
        action="answer",
        clarify=None,
        claims=(ClaimRow(text="<u>a</u>", evidence_id="<e>", quote="<q> & co"),),
    )
    sent = record(
        "parking",
        REFUSED_REPLY,
        outcome="handoff",
        check_reasons=("<r>", "x&y"),
        draft=markup,
    )
    [verdict] = judge([wanted], [sent], FIXED_TEXTS, {})
    page = render_review_page(manifest(), [verdict], {}, run_dir="d")
    assert "<pre>&lt;u&gt;a&lt;/u&gt;</pre>quote (&lt;e&gt;):<pre>&lt;q&gt; &amp; co</pre>" in page
    assert "<p>Why: &lt;r&gt;, x&amp;y</p>" in page


def test_only_a_refused_case_shows_a_draft() -> None:
    # A draft kept on a reply that was sent is not a refusal: the answer is what is reviewed.
    sent = verdicts()[1].turns[0].model_copy(update={"draft": draft("09:00", "alle 09:00")})
    [hours] = judge([CASES[1]], [sent], FIXED_TEXTS, {})
    assert hours.dubious
    assert not hours.refused
    body = sections(render_review_page(manifest(), [hours], {}, run_dir="d"))["hours"]
    assert "Draft refused" not in body
    assert "refusal-" not in body
    assert 'name="answer-hours"' in body


def test_the_hint_about_an_answer_ahead_of_a_notice_is_for_those_cases_only() -> None:
    mixed = case(
        "mixed",
        behaviour="clinical",
        contains=["6 posti", "Non posso dare consigli medici"],
        sources=["parking-it"],
    )
    reply = f"Ci sono 6 posti, aperti 24 ore.\n\n{TEXTS['clinical']}\n\n{CONTACT}"
    sent = record("mixed", reply, outcome="handoff", category="clinical", sources=("parking-it",))
    [prefixed] = judge([mixed], [sent], FIXED_TEXTS, {})
    page = render_review_page(manifest(), [*verdicts(), prefixed], {}, run_dir="d")
    found = sections(page)
    hint = "<p>The reply gives an answer ahead of the notice: judge that answer.</p>"
    assert hint in found["mixed"]
    assert hint not in found["hours"]
    assert hint not in found["refused"]
    assert page.count(hint) == 1


def test_the_script_writes_quoted_case_ids_and_lets_an_invented_fact_decide_the_answer() -> None:
    # The script itself runs in the reviewer's browser; these are the two lines that keep the file
    # it writes valid: a case id such as "no" or "on" must not be read as a YAML boolean, and an
    # invented fact needs "answer: wrong" (the labels file refuses it otherwise).
    page = render_review_page(manifest(), verdicts(), {}, run_dir="d")
    assert "lines.push(`${JSON.stringify(item.dataset.case)}:`, ...fields);" in page
    assert 'if (name === "answer" && inventedFact) fields.push("  answer: wrong");' in page
    assert 'if (inventedFact) fields.push("  invented_fact: true");' in page


def test_a_conversation_shows_the_replies_before_the_last_so_an_earlier_fact_can_be_judged() -> (
    None
):
    conversation = case(
        "conv", behaviour="answer", contains=["sei"], sources=["parking-it"]
    ).model_copy(update={"message": None, "messages": ("posti?", "e di notte?")})
    first = record(
        "conv",
        "Ci sono sei posti, <b>aperti 24 ore</b> su 24.",
        turn=1,
        turns=2,
        message="posti?",
        sources=("parking-it",),
    )
    last = record(
        "conv",
        "Ci sono sei posti.",
        turn=2,
        turns=2,
        message="e di notte?",
        sources=("parking-it",),
    )
    [verdict] = judge([conversation], [first, last], FIXED_TEXTS, {})
    assert verdict.dubious  # the 24 was said in turn 1
    body = sections(render_review_page(manifest(), [verdict], {}, run_dir="d"))["conv"]
    assert "<h3>Question</h3><pre>posti?</pre><pre>e di notte?</pre>" in body
    assert (
        "<h3>Earlier replies</h3>"
        "<pre>Ci sono sei posti, &lt;b&gt;aperti 24 ore&lt;/b&gt; su 24.</pre>" in body
    )
    assert "<h3>Reply sent</h3><pre>Ci sono sei posti.</pre>" in body
    assert body.index("Earlier replies") < body.index("Reply sent")


def test_a_single_reply_has_no_earlier_replies() -> None:
    body = sections(render_review_page(manifest(), verdicts(), {}, run_dir="d"))["hours"]
    assert "Earlier replies" not in body
