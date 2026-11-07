"""Email drafts (2026-11-07): the messages of the mailbox the email pass has handled.

Revision ID: 0009
Revises: 0008

One row per handled message: the conversation's and the message's IDs, the outcome, a keyed hash
of the sender, the business-local day, when, and what the model cost. No text and no address: the
text stays in the mailbox. The pass deletes the rows older than 30 days. Row-level security as on
every business table (0004).
"""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

SCOPE = "NULLIF(current_setting('app.business_id', true), '')"


def upgrade() -> None:
    op.create_table(
        "email_handled",
        sa.Column("business_id", sa.String(64), primary_key=True),
        sa.Column("thread_id", sa.String(128), primary_key=True),
        sa.Column("message_id", sa.String(128), primary_key=True),
        sa.Column("outcome", sa.String(24), nullable=False),
        sa.Column("sender_key", sa.String(64), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("handled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        sa.CheckConstraint("business_id <> ''", name="ck_email_handled_business_id_not_blank"),
        sa.CheckConstraint("usd >= 0", name="ck_email_handled_usd_not_negative"),
    )
    op.create_index("ix_email_handled_day", "email_handled", ["business_id", "day", "sender_key"])
    op.create_index(
        "ix_email_handled_thread", "email_handled", ["business_id", "thread_id", "handled_at"]
    )
    op.execute("ALTER TABLE email_handled ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE email_handled FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY email_handled_business_isolation ON email_handled "
        f"USING (business_id = {SCOPE}) WITH CHECK (business_id = {SCOPE})"
    )


def downgrade() -> None:
    op.drop_table("email_handled")
