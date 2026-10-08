"""Temporal worker entrypoint: `uv run python -m frontdesk.worker.main`."""

from __future__ import annotations

import asyncio

from temporalio.client import Client
from temporalio.worker import Worker

from frontdesk.observability.logs import configure_logging
from frontdesk.settings import Settings
from frontdesk.worker.workflows import HealthWorkflow, report_version

TASK_QUEUE = "frontdesk"


async def run_worker(settings: Settings) -> None:
    client = await Client.connect(settings.temporal_address, namespace=settings.temporal_namespace)
    worker = Worker(
        client, task_queue=TASK_QUEUE, workflows=[HealthWorkflow], activities=[report_version]
    )
    await worker.run()


def main() -> None:
    settings = Settings()
    configure_logging(settings.log_level)
    asyncio.run(run_worker(settings))


if __name__ == "__main__":
    main()
