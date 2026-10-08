from frontdesk.safety.normalize import normalize, strip_diacritics, tokens


def test_albanian_and_italian_diacritics_are_removed() -> None:
    assert strip_diacritics("Vizitë çmimi è perché") == "Vizite cmimi e perche"


def test_normalize_lowers_and_drops_punctuation() -> None:
    assert normalize("  Sa kushton një VIZITË?!  ") == "sa kushton nje vizite"


def test_tokens_split_on_spaces() -> None:
    assert tokens("Dr. Arben-Demo, 30€") == ["dr", "arben", "demo", "30"]
