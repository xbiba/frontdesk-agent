"""Booking requests the clinic decides (WhatsApp W1): a booking that waits for the clinic keeps its
request state on its hold.

Revision ID: 0008
Revises: 0007

The request's status, the patient's language for the answer, who decided and when, and when the
staff was told. A pending request keeps the patient's name for the staff's message until it is
decided; the phone is never kept once booked. Row-level security is the table's own (0005).
"""

import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

_OLD_CHECK = "status = 'held' OR (name IS NULL AND phone IS NULL)"
# NULL IN (...) is NULL, which a check lets pass: the status is tested for NULL first.
_NEW_CHECK = (
    "status = 'held' OR (phone IS NULL AND (name IS NULL OR (request_status IS NOT NULL AND "
    "request_status IN ('pending', 'deciding'))))"
)


def upgrade() -> None:
    op.add_column("booking_holds", sa.Column("request_status", sa.String(16)))
    op.add_column("booking_holds", sa.Column("language", sa.String(2)))
    op.add_column("booking_holds", sa.Column("decided_by", sa.String(80)))
    op.add_column("booking_holds", sa.Column("decided_at", sa.DateTime(timezone=True)))
    op.add_column("booking_holds", sa.Column("notified_at", sa.DateTime(timezone=True)))
    op.create_check_constraint(
        "ck_booking_holds_request_status",
        "booking_holds",
        "request_status IN ('pending', 'deciding', 'confirmed', 'declined', 'gone')",
    )
    op.drop_constraint("ck_booking_holds_no_personal_data_once_booked", "booking_holds")
    op.create_check_constraint(
        "ck_booking_holds_no_personal_data_once_booked", "booking_holds", _NEW_CHECK
    )
    op.create_index(
        "ix_booking_holds_requests",
        "booking_holds",
        ["business_id", "request_status", "notified_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_booking_holds_requests", "booking_holds")
    op.execute("UPDATE booking_holds SET name = NULL WHERE status = 'booked'")
    op.drop_constraint("ck_booking_holds_no_personal_data_once_booked", "booking_holds")
    op.create_check_constraint(
        "ck_booking_holds_no_personal_data_once_booked", "booking_holds", _OLD_CHECK
    )
    op.drop_constraint("ck_booking_holds_request_status", "booking_holds")
    for column in ("notified_at", "decided_at", "decided_by", "language", "request_status"):
        op.drop_column("booking_holds", column)
