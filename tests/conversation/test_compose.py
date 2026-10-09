import uuid

from frontdesk.agent.classification import ValidatedFields
from frontdesk.agent.vocab import Category, ClarifyField, Intent
from frontdesk.conversation.compose import clarify_question, closing, compose_facts, unanswered
from frontdesk.knowledge.evidence import Evidence, EvidenceKind
from frontdesk.knowledge.facts import Catalog

LOCATION = Evidence.create(
    kind=EvidenceKind.RECORD,
    source_ref="location:main",
    version="v",
    title="Indirizzo e contatti",
    content="...",
    values={
        "address": "Rruga e Shembullit 1",
        "phone": "+355 00 000 0000",
        "email": "info@klinika-demo.example",
        "hours": "lun-ven 08:00-19:00; sab 09:00-14:00",
    },
)
PRICE = Evidence.create(
    kind=EvidenceKind.RECORD,
    source_ref="price:physiotherapy",
    version="v",
    title="Prezzo",
    content="Fisioterapia: 30 EUR (a seduta)",
    values={"price": "30 EUR"},
)
BACK = Evidence.create(
    kind=EvidenceKind.RECORD,
    source_ref="orientation:back",
    version="v",
    title="Servizi",
    content="schiena: Visita ortopedica, Osteopatia",
    values={"services": "Visita ortopedica, Osteopatia"},
)
CATALOG = Catalog(
    business_id="demo-clinic",
    version_id=uuid.uuid4(),
    services={"osteopathy": {"it": "Osteopatia"}, "physiotherapy": {"it": "Fisioterapia"}},
    doctors={},
    problem_categories={},
)


def _fields(intent: Intent, category: Category = Category.ADMINISTRATIVE) -> ValidatedFields:
    return ValidatedFields(category=category, intent=intent, language="it")


def test_hours_and_contacts_show_only_what_was_asked() -> None:
    hours = compose_facts(_fields(Intent.HOURS), [LOCATION])
    assert hours == f"Orari: lun-ven 08:00-19:00; sab 09:00-14:00\n\n{closing('it')}"
    contact = compose_facts(_fields(Intent.CONTACT), [LOCATION])
    assert contact is not None
    assert contact.startswith("Telefono: +355 00 000 0000\nEmail: info@klinika-demo.example")


def test_prices_are_the_record_text() -> None:
    assert (
        compose_facts(_fields(Intent.PRICE), [PRICE])
        == f"Fisioterapia: 30 EUR (a seduta)\n\n{closing('it')}"
    )
    assert compose_facts(_fields(Intent.PRICE), []) is None


def test_orientation_names_services_without_repeating_the_problem() -> None:
    text = compose_facts(_fields(Intent.SERVICE_INFO, Category.ORIENTATION), [BACK])
    assert text == "Se ne occupano: Visita ortopedica, Osteopatia.\n\nVuoi sapere prezzi o orari?"


def test_clarifying_questions_are_written_by_code() -> None:
    assert clarify_question(ClarifyField.SERVICE, "it", CATALOG) == (
        "Per quale servizio? Per esempio: Osteopatia, Fisioterapia."
    )
    assert "ORD-" in clarify_question(ClarifyField.ORDER_DETAILS, "en", CATALOG)


def test_unanswered_names_the_asked_items_without_a_record() -> None:
    fields = ValidatedFields(
        category=Category.ADMINISTRATIVE,
        intent=Intent.PRICE,
        language="it",
        service_ids=("physiotherapy", "osteopathy"),
    )
    assert unanswered(fields, [PRICE]) == ("osteopathy",)
    assert unanswered(fields, []) == ("physiotherapy", "osteopathy")
    assert unanswered(_fields(Intent.HOURS), []) == ()
    assert unanswered(_fields(Intent.SERVICE_INFO, Category.ORIENTATION), [BACK]) == ()


def test_unanswered_reads_doctors_for_doctor_questions_and_services_for_service_questions() -> None:
    doctors = ValidatedFields(
        category=Category.ADMINISTRATIVE,
        intent=Intent.DOCTOR_INFO,
        language="it",
        doctor_ids=("dr-arben-demo", "ft-elira-shembull"),
        service_ids=("physiotherapy",),  # not what a doctor question asks about
    )
    arben = Evidence.create(
        kind=EvidenceKind.RECORD,
        source_ref="doctor:dr-arben-demo",
        version="v",
        title="Medico",
        content="Dr. Arben Demo: Visita ortopedica",
    )
    assert unanswered(doctors, [arben]) == ("ft-elira-shembull",)
    services = ValidatedFields(
        category=Category.ADMINISTRATIVE,
        intent=Intent.SERVICE_INFO,
        language="it",
        service_ids=("physiotherapy", "osteopathy"),
    )
    physiotherapy = Evidence.create(
        kind=EvidenceKind.RECORD,
        source_ref="service:physiotherapy",
        version="v",
        title="Servizio",
        content="Fisioterapia\nDurata: 45 minuti",
    )
    assert unanswered(services, [physiotherapy]) == ("osteopathy",)


def test_missing_items_follow_the_answered_ones_and_are_never_alone() -> None:
    missing = ["Osteopatia: Non ho questa informazione."]
    answered = compose_facts(_fields(Intent.PRICE), [PRICE], with_closing=False, missing=missing)
    assert answered == "Fisioterapia: 30 EUR (a seduta)\nOsteopatia: Non ho questa informazione."
    assert compose_facts(_fields(Intent.PRICE), [], missing=missing) is None


def test_unanswered_ignores_a_service_named_alongside_an_orientation_question() -> None:
    fields = ValidatedFields(
        category=Category.ORIENTATION,
        intent=Intent.SERVICE_INFO,
        language="it",
        problem_category_id="back",
        service_ids=("physiotherapy",),
    )
    assert unanswered(fields, [BACK]) == ()
