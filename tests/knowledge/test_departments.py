"""The clinic's departments, by name only (Albi, 2026-10-13): the agent may say the clinic has one and
give reception for its prices, times and doctors. Knowing only the services it prices, the agent said
the clinic offers nothing else (run of 2026-10-13)."""

from collections.abc import Callable
from datetime import date
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from frontdesk.conversation.reception_facts import render_facts
from frontdesk.db.engine import business_session
from frontdesk.knowledge.facts import department_names, load_catalog
from frontdesk.knowledge.versions import pack_digest
from frontdesk.tenancy.pack import BusinessPack

Activate = Callable[..., sessionmaker[Session]]
DEPARTMENTS = [
    {
        "id": "ultrasound",
        "names": {"sq": "Diagnostikë ekografike", "it": "Ecografie", "en": "Ultrasound"},
    },
    {"id": "dentistry", "names": {"sq": "Stomatologji", "it": "Odontoiatria", "en": "Dentistry"}},
]


def _with(pack: BusinessPack, departments: list[dict[str, Any]]) -> BusinessPack:
    return BusinessPack.model_validate({**pack.model_dump(), "departments": departments})


def test_a_department_needs_a_name_in_every_language(clinic_pack: BusinessPack) -> None:
    unnamed = [{"id": "ultrasound", "names": {"sq": "Diagnostikë ekografike", "it": "Ecografie"}}]
    with pytest.raises(ValidationError, match="department 'ultrasound' has no name in 'en'"):
        _with(clinic_pack, unnamed)
    with pytest.raises(ValidationError, match="duplicate department id 'ultrasound'"):
        _with(clinic_pack, [DEPARTMENTS[0], DEPARTMENTS[0]])


def test_the_departments_are_knowledge(clinic_pack: BusinessPack) -> None:
    # A new department is a new version to approve, like a new price.
    assert pack_digest(_with(clinic_pack, DEPARTMENTS)) != pack_digest(clinic_pack)


@pytest.mark.db
def test_the_agent_reads_the_departments_of_the_active_version(
    activate_pack: Activate, clinic_pack: BusinessPack
) -> None:
    pack = _with(clinic_pack, DEPARTMENTS)
    factory = activate_pack(pack)
    with business_session(factory, pack.id) as session:
        names = department_names(session, pack.id, "it")
        catalog = load_catalog(session, pack.id)
        assert catalog is not None
        facts = render_facts(session, pack, catalog, "it", date(2026, 10, 20), chat_booking=False)
    assert names == ["Odontoiatria", "Ecografie"]  # by id
    assert "Departments of the clinic" in facts
    assert "- Ecografie" in facts
