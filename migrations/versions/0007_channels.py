"""Channels (M4): the conversations, messages and testers of a channel such as Instagram.

Revision ID: 0007
Revises: 0006

Postgres is the channel's durable inbox (ADR-0011): a message is stored once by its channel ID, a
conversation runs one turn at a time under a lease, and every reply is recorded before it is sent.
A message's text stays only until the turn that reads it. Row-level security as on every business
table (0004).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

SCOPE = "NULLIF(current_setting('app.business_id', true), '')"
TABLES = ("channel_conversations", "channel_messages", "channel_testers")


def _not_blank(table: str) -> sa.CheckConstraint:
    return sa.CheckConstraint("business_id <> ''", name=f"ck_{table}_business_id_not_blank")


def upgrade() -> None:
    op.create_table(
        "channel_conversations",
        sa.Column("business_id", sa.String(64), primary_key=True),
        sa.Column("conversation_id", sa.String(64), primary_key=True),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),  # the person on the channel
        sa.Column("state", JSONB(), nullable=True),  # the engine's ConversationState
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("active_at", sa.DateTime(timezone=True), nullable=True),  # the last turn
        sa.Column("paused_until", sa.DateTime(timezone=True), nullable=True),  # the staff's
        sa.Column("slowed_at", sa.DateTime(timezone=True), nullable=True),  # the last slow-down
        sa.Column("lease_until", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "business_id", "channel", "user_id", name="uq_channel_conversations_user"
        ),
        _not_blank("channel_conversations"),
    )
    op.create_table(
        "channel_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", sa.String(64), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("channel", sa.String(16), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("external_id", sa.String(512), nullable=True),  # the channel's message ID
        sa.Column("text", sa.Text(), nullable=True),  # gone once read (in) or after a day (out)
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("attempts", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("reason", sa.String(64), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),  # the channel's time
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["business_id", "conversation_id"],
            ["channel_conversations.business_id", "channel_conversations.conversation_id"],
            ondelete="CASCADE",
            name="fk_channel_messages_conversation",
        ),
        sa.UniqueConstraint(
            "business_id", "channel", "external_id", name="uq_channel_messages_external_id"
        ),
        sa.CheckConstraint("direction IN ('in', 'out')", name="ck_channel_messages_direction"),
        _not_blank("channel_messages"),
    )
    op.create_index(
        "ix_channel_messages_work",
        "channel_messages",
        ["business_id", "conversation_id", "direction", "status"],
    )
    op.create_index(
        "ix_channel_messages_age", "channel_messages", ["business_id", "status", "created_at"]
    )
    op.create_table(
        "channel_testers",
        sa.Column("business_id", sa.String(64), primary_key=True),
        sa.Column("channel", sa.String(16), primary_key=True),
        sa.Column("user_id", sa.String(64), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        _not_blank("channel_testers"),
    )
    for table in TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_business_isolation ON {table} "
            f"USING (business_id = {SCOPE}) WITH CHECK (business_id = {SCOPE})"
        )


def downgrade() -> None:
    for table in reversed(TABLES):  # the messages before their conversations
        op.drop_table(table)
