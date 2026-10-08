"""Baseline: pgvector and default privileges for the application role.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

GRANT_APP_ROLE = """
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'frontdesk_app') THEN
    GRANT USAGE ON SCHEMA public TO frontdesk_app;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public
      GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO frontdesk_app;
  END IF;
END
$$;
"""


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(GRANT_APP_ROLE)


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")
