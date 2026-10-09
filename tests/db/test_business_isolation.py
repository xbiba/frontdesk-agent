"""What keeps one business's rows apart, beyond a policy that compares two strings (migration 0004).

The policies read the scope from a setting that a pooled connection leaves behind as '', foreign
key checks bypass row-level security, and a chunk's position is what ranks equal scores: so the
policies fail closed on an empty scope, children are tied to the business and version of their
parent, and a document cannot hold two chunks at one position.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from sqlalchemy import Engine, create_engine, func, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.db.engine import business_session, make_session_factory
from frontdesk.db.models import KnowledgeVersion

db = pytest.mark.db
BUSINESS_TABLES = ("knowledge_versions", "documents", "chunks", "records", "order_lookup_attempts")


class Owner:
    """Writes rows as the database owner, who is not bound by row-level security.

    That leaves the constraints alone to decide, which is what these tests are about.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def run(self, sql: str, **values: object) -> None:
        with self._engine.begin() as connection:
            connection.execute(text(sql), values)

    def version(self, business: str, status: str = "draft") -> uuid.UUID:
        version = uuid.uuid4()
        self.run(
            "INSERT INTO knowledge_versions (id, business_id, status, source_digest) "
            "VALUES (:id, :business, :status, 'digest')",
            id=version,
            business=business,
            status=status,
        )
        return version

    def document(self, business: str, version: uuid.UUID, key: str = "doc") -> uuid.UUID:
        document = uuid.uuid4()
        self.run(
            "INSERT INTO documents (id, business_id, knowledge_version_id, doc_key, title, "
            "language, content_hash) VALUES (:id, :business, :version, :key, 'Title', 'en', 'h')",
            id=document,
            business=business,
            version=version,
            key=key,
        )
        return document

    def chunk(
        self,
        business: str,
        version: uuid.UUID,
        document: uuid.UUID,
        ordinal: int = 0,
    ) -> uuid.UUID:
        chunk = uuid.uuid4()
        self.run(
            "INSERT INTO chunks (id, business_id, knowledge_version_id, document_id, ordinal, "
            "heading, content, search_text) "
            "VALUES (:id, :business, :version, :document, :ordinal, 'H', 'C', 'c')",
            id=chunk,
            business=business,
            version=version,
            document=document,
            ordinal=ordinal,
        )
        return chunk

    def record(self, business: str, version: uuid.UUID, key: str = "main") -> uuid.UUID:
        record = uuid.uuid4()
        self.run(
            "INSERT INTO records (id, business_id, knowledge_version_id, kind, record_key, "
            "payload) VALUES (:id, :business, :version, 'location', :key, '{}'::jsonb)",
            id=record,
            business=business,
            version=version,
            key=key,
        )
        return record

    def count(self, table: str) -> int:
        with self._engine.connect() as connection:
            return int(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


@contextmanager
def owner_of(owner_url: str) -> Iterator[Owner]:
    engine = create_engine(owner_url)
    try:
        yield Owner(engine)
    finally:
        engine.dispose()


# --- an empty scope sees nothing and writes nothing -------------------------------------------


@db
def test_an_unscoped_session_on_a_reused_connection_sees_nothing_and_writes_nothing(
    database: tuple[str, str],
) -> None:
    owner_url, app_url = database
    with owner_of(owner_url) as owner:
        owner.version("alpha")
    # One connection for everything, as a busy pool hands the same one out again.
    engine = create_engine(app_url, pool_size=1, max_overflow=0)
    factory = make_session_factory(engine)
    with business_session(factory, "alpha") as scoped:
        assert scoped.scalar(select(func.count()).select_from(KnowledgeVersion)) == 1
    with factory() as unscoped:
        # The premise: the transaction-local setting now reads '' on this connection, not NULL.
        scope = unscoped.execute(text("SELECT current_setting('app.business_id', true)"))
        assert scope.scalar_one() == ""
        assert unscoped.scalars(select(KnowledgeVersion)).all() == []
        unscoped.add(KnowledgeVersion(business_id="alpha", status="draft", source_digest="d"))
        with pytest.raises(DBAPIError, match="row-level security"):
            unscoped.flush()
    with factory() as unscoped:
        unscoped.add(KnowledgeVersion(business_id="", status="draft", source_digest="d"))
        with pytest.raises(DBAPIError, match=r"row-level security|ck_knowledge_versions"):
            unscoped.flush()
    engine.dispose()


@db
def test_a_scope_still_sees_its_own_rows_and_not_the_others(database: tuple[str, str]) -> None:
    owner_url, app_url = database
    with owner_of(owner_url) as owner:
        owner.version("alpha")
        owner.version("beta")
    engine = create_engine(app_url, pool_size=1, max_overflow=0)
    factory = make_session_factory(engine)
    for scope in ("alpha", "beta", "alpha"):
        with business_session(factory, scope) as session:
            assert set(session.scalars(select(KnowledgeVersion.business_id))) == {scope}
    engine.dispose()


def _enter(factory: sessionmaker[Session], scope: str) -> None:
    with business_session(factory, scope):
        pass


@pytest.mark.parametrize("scope", ["", " ", "\t\n"])
def test_a_blank_business_scope_is_refused_before_any_connection(scope: str) -> None:
    factory = sessionmaker()  # bound to nothing: an attempt to connect would fail differently
    with pytest.raises(ValueError, match="business id"):
        _enter(factory, scope)


# --- a blank business id cannot be stored ---------------------------------------------------------


@db
@pytest.mark.parametrize("table", BUSINESS_TABLES)
def test_no_business_table_stores_a_blank_business_id(
    database: tuple[str, str], table: str
) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        version = owner.version("alpha")
        document = owner.document("alpha", version)
        inserts = {
            "knowledge_versions": lambda: owner.version(""),
            "documents": lambda: owner.document("", version, key="other"),
            "chunks": lambda: owner.chunk("", version, document, ordinal=7),
            "records": lambda: owner.record("", version, key="other"),
            "order_lookup_attempts": lambda: owner.run(
                "INSERT INTO order_lookup_attempts (business_id, conversation_id, used, "
                "updated_at) VALUES ('', 'c1', 1, now())"
            ),
        }
        with pytest.raises(IntegrityError, match=f"ck_{table}_business_id_not_blank"):
            inserts[table]()


# --- a child belongs to the business and the version of its parent --------------------------------


@db
@pytest.mark.parametrize(
    ("child", "constraint"),
    [
        ("documents", "fk_documents_version_business"),
        ("chunks", "fk_chunks_version_business"),
        ("records", "fk_records_version_business"),
    ],
)
def test_a_child_cannot_name_a_version_of_another_business(
    database: tuple[str, str], child: str, constraint: str
) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        alpha_version = owner.version("alpha")
        alpha_document = owner.document("alpha", alpha_version)
        # Foreign key checks bypass row-level security: only the constraint can refuse these.
        inserts = {
            "documents": lambda: owner.document("beta", alpha_version, key="other"),
            "chunks": lambda: owner.chunk("beta", alpha_version, alpha_document),
            "records": lambda: owner.record("beta", alpha_version),
        }
        with pytest.raises(IntegrityError, match=constraint):
            inserts[child]()


@db
def test_a_chunk_cannot_point_at_a_document_of_another_version(database: tuple[str, str]) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        first = owner.version("alpha")
        second = owner.version("alpha")
        document = owner.document("alpha", first)
        with pytest.raises(IntegrityError, match="fk_chunks_document_version"):
            owner.chunk("alpha", second, document)


@db
def test_a_chunk_cannot_point_at_a_document_of_another_business(database: tuple[str, str]) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        alpha_version = owner.version("alpha")
        beta_version = owner.version("beta")
        beta_document = owner.document("beta", beta_version)
        with pytest.raises(IntegrityError, match="fk_chunks_document_version"):
            owner.chunk("alpha", alpha_version, beta_document)


@db
def test_rows_that_agree_on_business_and_version_are_stored(database: tuple[str, str]) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        version = owner.version("alpha")
        document = owner.document("alpha", version)
        owner.chunk("alpha", version, document)
        owner.record("alpha", version)
        other = owner.version("beta")
        owner.chunk("beta", other, owner.document("beta", other))
        assert [owner.count(table) for table in ("documents", "chunks", "records")] == [2, 2, 1]


@db
def test_deleting_a_version_removes_what_hangs_on_it(database: tuple[str, str]) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        version = owner.version("alpha")
        owner.chunk("alpha", version, owner.document("alpha", version))
        owner.record("alpha", version)
        kept = owner.version("beta")
        owner.chunk("beta", kept, owner.document("beta", kept))
        owner.run("DELETE FROM knowledge_versions WHERE id = :id", id=version)
        assert [owner.count(table) for table in ("documents", "chunks", "records")] == [1, 1, 0]


@db
def test_a_document_has_one_chunk_per_position(database: tuple[str, str]) -> None:
    owner_url, _ = database
    with owner_of(owner_url) as owner:
        version = owner.version("alpha")
        document = owner.document("alpha", version)
        owner.chunk("alpha", version, document, ordinal=0)
        owner.chunk("alpha", version, document, ordinal=1)
        with pytest.raises(IntegrityError, match="uq_chunks_document_ordinal"):
            owner.chunk("alpha", version, document, ordinal=1)
        # The same position in another document is fine.
        owner.chunk("alpha", version, owner.document("alpha", version, key="other"), ordinal=1)
