"""The turn pipeline a command runs on, and the one line a failure to build it gives.

Every command that builds the pipeline (`chat`, `eval turns` and their live forms) turns the same
failures into the same `error:` lines, so a person reads one wording whichever command stopped. The
module imports nothing of the commands, so that each of them can import it.
"""

from __future__ import annotations

from pathlib import Path

import typer
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from frontdesk.composition import (
    Deployment,
    ModelWrapper,
    NoActiveKnowledgeError,
    build_turn_runtime,
)
from frontdesk.conversation.turn import TurnRuntime
from frontdesk.knowledge.versions import EmbedderMismatchError
from frontdesk.model_gateway.routes import DataClass, UnapprovedRouteError


def fail(message: str) -> typer.Exit:
    """Print `error: <message>` and give back the exit to raise; no traceback follows."""
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(1)


def without_knowledge(error: NoActiveKnowledgeError, env_file: Path) -> typer.Exit:
    """Say what to run when the business has no active knowledge; the caller raises the result."""
    return fail(
        f"{error}. Draft, approve and activate a version first "
        f"(demo businesses: frontdesk knowledge bootstrap-demo --env-file {env_file})."
    )


def database_location(url: str) -> str:
    """`host:port` of a database URL: all of it that is safe to print (no user, no password)."""
    parsed = make_url(url)
    host = parsed.host or "localhost"
    return f"{host}:{parsed.port}" if parsed.port else host


def unreachable(database_url: str) -> typer.Exit:
    """The one line for a database that does not answer, and the exit to raise."""
    return fail(
        f"cannot reach the database ({database_location(database_url)}); "
        "start it with: docker compose up -d --wait postgres"
    )


def build_runtime(
    deployment: Deployment,
    env_file: Path,
    *,
    data_class: DataClass = DataClass.PATIENT_DERIVED,
    wrap_model: ModelWrapper | None = None,
) -> TurnRuntime:
    """The deployment's pipeline for messages of `data_class`; each way it can fail to build is an
    `error:` line and exit code 1, never a traceback.

    A conversation carries patient-derived text: a route approved for evaluation data only never
    receives one (M3 design §4.2), and the line says so. `wrap_model` is the live evaluation's meter.
    """
    try:
        return build_turn_runtime(deployment, data_class=data_class, wrap_model=wrap_model)
    except NoActiveKnowledgeError as error:
        raise without_knowledge(error, env_file) from error
    except EmbedderMismatchError as error:
        raise fail(f"{error} --env-file {env_file}") from error
    except UnapprovedRouteError as error:
        raise fail(f"{error} (--env-file {env_file})") from error
    except OperationalError as error:
        # The driver's own message (its host, its port, SQLAlchemy's help link) is left out.
        raise unreachable(deployment.settings.database_url) from error
    except ValueError as error:
        # What is left is a setting the route needs, such as the Google Cloud project, or a prompt
        # file that is missing or blank.
        raise fail(str(error)) from error
