"""Versioned knowledge: versions, documents, chunks and records, with RLS.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

BUSINESS_TABLES = ("knowledge_versions", "documents", "chunks", "records")


def _id() -> sa.Column:
    return sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True)


def _business() -> sa.Column:
    return sa.Column("business_id", sa.String(64), nullable=False)


def _version_fk() -> sa.Column:
    return sa.Column(
        "knowledge_version_id",
        postgresql.UUID(as_uuid=True),
        sa.ForeignKey("knowledge_versions.id", ondelete="CASCADE"),
        nullable=False,
    )


def upgrade() -> None:
    op.create_table(
        "knowledge_versions",
        _id(),
        _business(),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("source_digest", sa.String(64), nullable=False),
        sa.Column("embedder", sa.String(200)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("approved_by", sa.String(200)),
        sa.Column("approval_evidence", sa.Text()),
        sa.Column("approved_at", sa.DateTime(timezone=True)),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.Column("retired_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('draft', 'approved', 'active', 'retired')",
            name="ck_knowledge_versions_status",
        ),
    )
    op.execute(
        "CREATE UNIQUE INDEX uq_knowledge_versions_one_active "
        "ON knowledge_versions (business_id) WHERE status = 'active'"
    )
    op.create_table(
        "documents",
        _id(),
        _business(),
        _version_fk(),
        sa.Column("doc_key", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("language", sa.String(2), nullable=False),
        sa.Column("public_url", sa.Text()),
        sa.Column("valid_from", sa.Date()),
        sa.Column("valid_until", sa.Date()),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.UniqueConstraint("knowledge_version_id", "doc_key", name="uq_documents_version_key"),
    )
    op.create_table(
        "chunks",
        _id(),
        _business(),
        _version_fk(),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("search_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384)),
    )
    op.execute(
        "ALTER TABLE chunks ADD COLUMN tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED"
    )
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
    op.execute(
        "CREATE INDEX ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_table(
        "records",
        _id(),
        _business(),
        _version_fk(),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("record_key", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "knowledge_version_id", "kind", "record_key", name="uq_records_version_kind_key"
        ),
    )
    for table in BUSINESS_TABLES:
        op.execute(f"CREATE INDEX ix_{table}_business ON {table} (business_id)")
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_business_isolation ON {table} "
            "USING (business_id = current_setting('app.business_id', true)) "
            "WITH CHECK (business_id = current_setting('app.business_id', true))"
        )


def downgrade() -> None:
    for table in reversed(BUSINESS_TABLES):
        op.drop_table(table)
