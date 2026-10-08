from hypothesis import given
from hypothesis import strategies as st

from frontdesk.knowledge.chunking import chunk_markdown

BODY = """Intro paragraph before any heading.

## Che cos'è
La fisioterapia comprende esercizi guidati.

## Come prepararsi
Porta abiti comodi.
"""


def test_one_chunk_per_section_with_heading_path() -> None:
    chunks = chunk_markdown("Fisioterapia", BODY)
    assert [c.heading for c in chunks] == [
        "Fisioterapia",
        "Fisioterapia > Che cos'è",
        "Fisioterapia > Come prepararsi",
    ]
    assert chunks[1].content == "La fisioterapia comprende esercizi guidati."
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_long_sections_split_on_paragraphs() -> None:
    body = "## Lungo\n" + "\n\n".join(f"Paragrafo {i} " + "x" * 80 for i in range(6))
    chunks = chunk_markdown("Doc", body, max_chars=200)
    assert len(chunks) > 1
    assert all(len(c.content) <= 200 for c in chunks)
    assert chunks[0].content.startswith("Paragrafo 0")


def test_empty_body_gives_no_chunks() -> None:
    assert chunk_markdown("Doc", "   \n\n") == []


@given(st.text(max_size=3000))
def test_chunks_are_never_empty_or_too_long(body: str) -> None:
    for chunk in chunk_markdown("Doc", body, max_chars=300):
        assert chunk.content.strip()
        assert len(chunk.content) <= 300
