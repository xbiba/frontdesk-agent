import math

import pytest

from frontdesk.db.models import EMBEDDING_DIM
from frontdesk.knowledge.embedders import (
    DEMO_MODEL,
    FastEmbedEmbedder,
    HashEmbedder,
    NoEmbedder,
    build_embedder,
    cosine,
)
from frontdesk.settings import EmbeddingBackend


def test_hash_embedder_is_deterministic_and_unit_length() -> None:
    first, second = HashEmbedder().embed(["Sa kushton fizioterapia?", "Sa kushton fizioterapia?"])
    assert first == second
    assert len(first) == EMBEDDING_DIM
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-9)


def test_hash_embedder_ignores_diacritics() -> None:
    accented, plain = HashEmbedder().embed(["vizitë", "vizite"])
    assert accented == plain


def test_hash_embedder_rewards_shared_tokens() -> None:
    query, related, unrelated = HashEmbedder().embed(
        ["parking at the clinic", "free parking spaces", "dermatology visit price"]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_build_embedder_selects_backend() -> None:
    assert isinstance(build_embedder(EmbeddingBackend.HASH), HashEmbedder)


def test_identity_names_the_model() -> None:
    assert HashEmbedder().identity == "hash-v1"


def test_cosine_handles_unnormalized_and_zero_vectors() -> None:
    assert cosine([2.0, 0.0], [5.0, 0.0]) == 1.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


@pytest.mark.slow
def test_demo_model_relates_albanian_and_italian() -> None:
    pytest.importorskip("fastembed")
    embedder = FastEmbedEmbedder()
    sq, it, other = embedder.embed(
        ["Sa kushton fizioterapia?", "Quanto costa la fisioterapia?", "Dove posso parcheggiare?"]
    )
    assert len(sq) == EMBEDDING_DIM
    assert cosine(sq, it) > cosine(sq, other)
    assert embedder.identity == f"fastembed:{DEMO_MODEL}"


def test_a_deployment_may_have_no_embedder_and_then_searches_nothing() -> None:
    # Albi, 2026-10-16: a small pack's knowledge rides in the prompt, with no model in the image.
    embedder = build_embedder(EmbeddingBackend.NONE)
    assert isinstance(embedder, NoEmbedder)
    assert embedder.identity == "none"
    with pytest.raises(RuntimeError, match="no embedder"):
        embedder.embed(["Sa kushton?"])
