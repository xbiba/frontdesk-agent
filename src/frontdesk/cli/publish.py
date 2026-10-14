"""`frontdesk knowledge publish` (M4): a deployment whose knowledge was approved on another
database (the Mac's, with the clinic's written confirmation in approvals/) drafts, approves and
activates the same content in one step. The pack's digest decides what the content is."""

from __future__ import annotations

from datetime import datetime

import typer

from frontdesk.composition import open_deployment
from frontdesk.db.engine import business_session
from frontdesk.knowledge.versions import (
    VersionStatus,
    activate,
    approve,
    create_draft,
    version_status,
)
from frontdesk.settings import Settings


def publish_knowledge(settings: Settings, *, by: str, evidence: str, now: datetime) -> int:
    deployment = open_deployment(settings)
    with business_session(deployment.session_factory, deployment.pack.id) as session:
        draft = create_draft(session, deployment.pack)
        status = version_status(session, draft.version_id)
        if status is VersionStatus.ACTIVE:
            typer.echo(f"already active: {draft.version_id}")
            return 0
        if status is VersionStatus.DRAFT:
            approve(session, draft.version_id, approved_by=by, evidence=evidence, now=now)
        activate(session, draft.version_id, deployment.embedder, now=now)
    typer.echo(f"active: {draft.version_id}")
    return 0
