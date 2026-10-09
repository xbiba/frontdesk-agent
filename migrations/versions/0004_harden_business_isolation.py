"""Harden business isolation: fail closed without a scope, tie children to their parents.

Revision ID: 0004
Revises: 0003

Three weaknesses of the policies in 0002 and 0003 and of the schema under them:

- A policy compares `business_id` with `current_setting('app.business_id', true)`. On a pooled
  connection that has left a scoped transaction the setting reads '' rather than NULL, so a session
  with no scope could read and write rows whose `business_id` is ''. The policies now compare with
  NULLIF(setting, ''), which is NULL for an empty scope and matches no row; and no table can hold
  a blank `business_id`.
- Foreign key checks bypass row-level security, so a row of one business could name a version of
  another, and a chunk could name a document of another version. Children now reference the
  business and the version of their parent together.
- A document could hold two chunks at one position, and equal retrieval scores are ranked by that
  position. It is unique now.

This is a revision of its own, not an edit of 0002 and 0003: a database already at 0003 would
otherwise keep the weak policies. The single-column foreign keys of 0002 stay in place (they cascade
too), so the downgrade is exactly the removal of what is added here. The upgrade fails on a database
whose rows break a new rule (a blank `business_id`, a child that names another business's version),
and because DDL is transactional it then changes nothing.
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

BUSINESS_TABLES = (
    "knowledge_versions",
    "documents",
    "chunks",
    "records",
    "order_lookup_attempts",
)
CHILD_TABLES = ("documents", "chunks", "records")
SCOPE = "current_setting('app.business_id', true)"


def _set_policy(table: str, scope: str) -> None:
    op.execute(
        f"ALTER POLICY {table}_business_isolation ON {table} "
        f"USING (business_id = {scope}) WITH CHECK (business_id = {scope})"
    )


def upgrade() -> None:
    for table in BUSINESS_TABLES:
        _set_policy(table, f"NULLIF({SCOPE}, '')")
        op.create_check_constraint(f"ck_{table}_business_id_not_blank", table, "business_id <> ''")
    op.create_unique_constraint(
        "uq_knowledge_versions_id_business", "knowledge_versions", ["id", "business_id"]
    )
    op.create_unique_constraint(
        "uq_documents_id_version", "documents", ["id", "knowledge_version_id"]
    )
    for table in CHILD_TABLES:
        op.create_foreign_key(
            f"fk_{table}_version_business",
            table,
            "knowledge_versions",
            ["knowledge_version_id", "business_id"],
            ["id", "business_id"],
            ondelete="CASCADE",
        )
    op.create_foreign_key(
        "fk_chunks_document_version",
        "chunks",
        "documents",
        ["document_id", "knowledge_version_id"],
        ["id", "knowledge_version_id"],
        ondelete="CASCADE",
    )
    op.create_unique_constraint("uq_chunks_document_ordinal", "chunks", ["document_id", "ordinal"])


def downgrade() -> None:
    op.drop_constraint("uq_chunks_document_ordinal", "chunks", type_="unique")
    op.drop_constraint("fk_chunks_document_version", "chunks", type_="foreignkey")
    for table in reversed(CHILD_TABLES):
        op.drop_constraint(f"fk_{table}_version_business", table, type_="foreignkey")
    op.drop_constraint("uq_documents_id_version", "documents", type_="unique")
    op.drop_constraint("uq_knowledge_versions_id_business", "knowledge_versions", type_="unique")
    for table in reversed(BUSINESS_TABLES):
        op.drop_constraint(f"ck_{table}_business_id_not_blank", table, type_="check")
        _set_policy(table, SCOPE)
