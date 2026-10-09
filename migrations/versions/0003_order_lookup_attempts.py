"""Order lookup attempts per conversation, with RLS.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "order_lookup_attempts",
        sa.Column("business_id", sa.String(64), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("used", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("business_id", "conversation_id"),
    )
    op.execute("ALTER TABLE order_lookup_attempts ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE order_lookup_attempts FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY order_lookup_attempts_business_isolation ON order_lookup_attempts "
        "USING (business_id = current_setting('app.business_id', true)) "
        "WITH CHECK (business_id = current_setting('app.business_id', true))"
    )


def downgrade() -> None:
    op.drop_table("order_lookup_attempts")
