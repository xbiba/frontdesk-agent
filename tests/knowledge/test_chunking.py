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


def test_long_sections_pack_whole_paragraphs() -> None:
    paragraphs = [f"Paragrafo {i} " + "x" * 80 for i in range(6)]
    chunks = chunk_markdown("Doc", "## Lungo\n" + "\n\n".join(paragraphs), max_chars=200)
    assert [c.content for c in chunks] == [
        "\n\n".join(paragraphs[0:2]),
        "\n\n".join(paragraphs[2:4]),
        "\n\n".join(paragraphs[4:6]),
    ]
    assert {c.heading for c in chunks} == {"Doc > Lungo"}
    assert [c.ordinal for c in chunks] == [0, 1, 2]


def test_a_paragraph_longer_than_the_limit_is_cut_at_the_limit() -> None:
    chunks = chunk_markdown("Doc", "x" * 500, max_chars=200)
    assert [len(c.content) for c in chunks] == [200, 200, 100]
    assert "".join(c.content for c in chunks) == "x" * 500


def test_two_paragraphs_that_exactly_fit_share_a_chunk() -> None:
    body = "a" * 99 + "\n\n" + "b" * 99
    assert [len(c.content) for c in chunk_markdown("Doc", body, max_chars=200)] == [200]
    assert [len(c.content) for c in chunk_markdown("Doc", body, max_chars=199)] == [99, 99]


def test_blank_lines_carrying_whitespace_still_separate_paragraphs() -> None:
    first, second = "a" * 150, "b" * 150
    for blank in ("", " ", "\t", "\N{NO-BREAK SPACE}"):
        chunks = chunk_markdown("Doc", f"{first}\n{blank}\n{second}", max_chars=200)
        assert [c.content for c in chunks] == [first, second]


_UNIT = st.text(
    alphabet=st.characters(blacklist_characters="#", blacklist_categories=["Cs"]),
    min_size=1,
    max_size=12,
)


def _repeat(unit: str, times: int) -> str:
    return unit * times


# Repeating a short unit gives paragraphs that often exceed max_chars (st.text alone stays tiny).
_PARAGRAPH = st.builds(_repeat, _UNIT, st.integers(min_value=1, max_value=60))


@given(st.lists(_PARAGRAPH, max_size=8), st.integers(min_value=10, max_value=200))
def test_chunks_are_trimmed_bounded_and_lossless(paragraphs: list[str], max_chars: int) -> None:
    chunks = chunk_markdown("Doc", "\n\n".join(paragraphs), max_chars=max_chars)
    for chunk in chunks:
        assert chunk.content
        assert chunk.content == chunk.content.strip()
        assert len(chunk.content) <= max_chars
    kept = "".join("".join(chunk.content.split()) for chunk in chunks)
    assert kept == "".join("".join(paragraph.split()) for paragraph in paragraphs)
    assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))
