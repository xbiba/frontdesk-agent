"""Booking holds (M5): a slot taken while a patient books it from the chat, then booked.

Revision ID: 0005
Revises: 0004

An exclusion constraint (btree_gist) refuses two holds of one doctor whose times overlap, so two
patients never get the same slot, even in parallel. Name and phone live on a hold only while it is
held (the patient is still typing them, at most 15 minutes); a booked hold keeps none, by
constraint. Row-level security as on every business table (0004).
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

SCOPE = "NULLIF(current_setting('app.business_id', true), '')"


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")
    op.create_table(
        "booking_holds",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", sa.String(64), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("doctor_id", sa.String(64), nullable=False),
        sa.Column("service_id", sa.String(64), nullable=False),
        sa.Column("during", postgresql.TSTZRANGE(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("name", sa.String(80)),
        sa.Column("phone", sa.String(32)),
        sa.Column("event_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("business_id <> ''", name="ck_booking_holds_business_id_not_blank"),
        sa.CheckConstraint("status IN ('held', 'booked')", name="ck_booking_holds_status"),
        sa.CheckConstraint(
            "status = 'held' OR (name IS NULL AND phone IS NULL)",
            name="ck_booking_holds_no_personal_data_once_booked",
        ),
    )
    op.execute(
        "ALTER TABLE booking_holds ADD CONSTRAINT ex_booking_holds_overlap "
        "EXCLUDE USING gist (business_id WITH =, doctor_id WITH =, during WITH &&)"
    )
    op.execute("ALTER TABLE booking_holds ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE booking_holds FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY booking_holds_business_isolation ON booking_holds "
        f"USING (business_id = {SCOPE}) WITH CHECK (business_id = {SCOPE})"
    )


def downgrade() -> None:
    op.drop_table("booking_holds")
