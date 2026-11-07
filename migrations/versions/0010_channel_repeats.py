"""The site's chat (2026-11-07): a conversation remembers whether its last messages were the same.

Revision ID: 0010
Revises: 0009

A keyed digest of the last turn's text and how many turns in a row had it: the third identical
message gets the handoff text without the model (design of the site's chat, §3). The text itself
is never kept; the digest goes with the state after a week of silence. Row-level security is the
table's own (0007).
"""

import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("channel_conversations", sa.Column("last_digest", sa.String(64)))
    op.add_column(
        "channel_conversations",
        sa.Column("repeats", sa.SmallInteger(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("channel_conversations", "repeats")
    op.drop_column("channel_conversations", "last_digest")
