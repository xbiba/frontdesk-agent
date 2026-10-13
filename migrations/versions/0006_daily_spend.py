"""Daily spend of the models in production (Albi, 2026-10-13: a cap per day, with an alert and a
switch, before Instagram).

Revision ID: 0006
Revises: 0005

One row per business and business-local day: what its model requests cost, in USD. Row-level
security as on every business table (0004).
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

SCOPE = "NULLIF(current_setting('app.business_id', true), '')"


def upgrade() -> None:
    op.create_table(
        "daily_spend",
        sa.Column("business_id", sa.String(64), primary_key=True),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("usd", sa.Numeric(12, 6), nullable=False, server_default="0"),
        # The highest alert logged that day: 0 none, 1 the alert share, 2 the cap.
        sa.Column("alerted", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.CheckConstraint("business_id <> ''", name="ck_daily_spend_business_id_not_blank"),
        sa.CheckConstraint("usd >= 0", name="ck_daily_spend_usd_not_negative"),
    )
    op.execute("ALTER TABLE daily_spend ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE daily_spend FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY daily_spend_business_isolation ON daily_spend "
        f"USING (business_id = {SCOPE}) WITH CHECK (business_id = {SCOPE})"
    )


def downgrade() -> None:
    op.drop_table("daily_spend")
