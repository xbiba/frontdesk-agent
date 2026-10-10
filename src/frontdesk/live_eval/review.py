"""The review page: the refused cases and, in M3's first round, every answer the model wrote (M3
design §4.4; external review of 2026-10-11: the number, link and negation filter only says where to
look first, since an added sentence without any of them can be invented too).

For each case the page shows the question, the expected answer, the bot's reply, the model's draft
and why a check refused it. The reviewer chooses "refusal right" or "refusal wrong", "answer right"
or "answer wrong" (and whether a wrong answer invents a fact), then saves `labels.yaml` into the
run directory; `frontdesk eval report` then recomputes the report. Everything shown is escaped:
the texts are the model's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape

from frontdesk.live_eval.judging import CaseVerdict, Label
from frontdesk.live_eval.records import RunManifest

_SCRIPT = """
function labels() {
  const lines = [];
  for (const item of document.querySelectorAll("section[data-case]")) {
    const fields = [];
    const invented = item.querySelector("input[name^='invented-']");
    const inventedFact = Boolean(invented && invented.checked);
    for (const name of ["refusal", "answer"]) {
      const chosen = item.querySelector(`input[name="${name}-${item.dataset.case}"]:checked`);
      // An invented fact is a wrong answer, whatever the other choice says.
      if (name === "answer" && inventedFact) fields.push("  answer: wrong");
      else if (chosen) fields.push(`  ${name}: ${chosen.value}`);
    }
    if (inventedFact) fields.push("  invented_fact: true");
    const note = item.querySelector("textarea").value.trim();
    if (note) fields.push(`  note: ${JSON.stringify(note)}`);
    if (fields.length) lines.push(`${JSON.stringify(item.dataset.case)}:`, ...fields);
  }
  return lines.join("\\n") + "\\n";
}
function save() {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([labels()], {type: "text/yaml"}));
  link.download = "labels.yaml";
  link.click();
}
"""

_STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; max-width: 60rem; margin: 2rem auto; padding: 0 1rem; }
section { border-top: 1px solid #ccc; padding: 1rem 0; }
pre { white-space: pre-wrap; background: #f4f4f4; padding: .5rem; }
.claim { margin: .5rem 0; }
textarea { width: 100%; }
"""


def review_cases(verdicts: Sequence[CaseVerdict]) -> tuple[CaseVerdict, ...]:
    return tuple(verdict for verdict in verdicts if verdict.refused or verdict.model_text)


def render_review_page(
    manifest: RunManifest,
    verdicts: Sequence[CaseVerdict],
    labels: Mapping[str, Label],
    *,
    run_dir: str,
) -> str:
    items = review_cases(verdicts)
    sections = "\n".join(_section(verdict, labels.get(verdict.case.id)) for verdict in items)
    title = f"Review: {manifest.run_id}"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{escape(title)}</title><style>{_STYLE}</style></head>
<body>
<h1>{escape(title)}</h1>
<p>{len(items)} cases to review: refused answers and answers to check. Choose, then
<button type="button" onclick="save()">save labels.yaml</button> into the run directory and run
<code>frontdesk eval report --run-dir {escape(run_dir)}</code> again.</p>
{sections}
<script>{_SCRIPT}</script>
</body>
</html>
"""


def _section(verdict: CaseVerdict, label: Label | None) -> str:
    case = verdict.case
    last = verdict.turns[-1]
    questions = "".join(f"<pre>{escape(turn.message)}</pre>" for turn in verdict.turns)
    earlier = "".join(f"<pre>{escape(turn.reply)}</pre>" for turn in verdict.turns[:-1])
    expected = (
        f"<p>Behaviour: {escape(str(case.expect.behaviour or '-'))}; must say: "
        f"{escape('; '.join(case.expect.contains) or '-')}; must not say: "
        f"{escape('; '.join(case.expect.not_contains) or '-')}</p>"
    )
    parts = [
        f'<section data-case="{escape(case.id)}">',
        f"<h2>{escape(case.id)} ({escape(verdict.language)})</h2>",
        f"<h3>Question</h3>{questions}",
        *([f"<h3>Earlier replies</h3>{earlier}"] if earlier else []),
        f"<h3>Expected answer</h3>{expected}",
        f"<h3>Reply sent</h3><pre>{escape(last.reply)}</pre>",
    ]
    if verdict.answered_prefix:
        parts.append("<p>The reply gives an answer ahead of the notice: judge that answer.</p>")
    if verdict.refused and last.draft is not None:
        claims = "".join(
            f'<div class="claim"><pre>{escape(claim.text)}</pre>quote ({escape(claim.evidence_id)}):'
            f"<pre>{escape(claim.quote)}</pre></div>"
            for claim in last.draft.claims
        )
        reasons = escape(", ".join(last.check_reasons) or "-")
        parts += [
            f"<h3>Draft refused by the checks</h3>{claims}<p>Why: {reasons}</p>",
            _choice("refusal", case.id, label.refusal if label else None, "Refusal"),
        ]
    if verdict.model_text:
        invented = " checked" if label is not None and label.invented_fact else ""
        if verdict.dubious:
            parts.append(
                "<p>Look here first: the answer adds a number, a link or a negation the "
                "expected answer lacks.</p>"
            )
        parts += [
            _choice("answer", case.id, label.answer if label else None, "Answer"),
            f'<label><input type="checkbox" name="invented-{escape(case.id)}"{invented}> '
            "a wrong answer that states a price, time, contact, number or negation nobody "
            "approved</label>",
        ]
    note = escape(label.note) if label is not None and label.note else ""
    parts += [f"<p>Note</p><textarea rows=2>{note}</textarea>", "</section>"]
    return "\n".join(parts)


def _choice(name: str, case_id: str, chosen: str | None, title: str) -> str:
    options = [
        f'<label><input type="radio" name="{name}-{escape(case_id)}" value="{value}"'
        f"{' checked' if chosen == value else ''}> {title.lower()} {value}</label>"
        for value in ("right", "wrong")
    ]
    return f"<p>{title}: " + " ".join(options) + "</p>"
