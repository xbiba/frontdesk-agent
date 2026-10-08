import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

import frontdesk
from frontdesk.worker.workflows import HealthWorkflow, report_version

pytestmark = pytest.mark.temporal


async def test_health_workflow_returns_version() -> None:
    async with (
        await WorkflowEnvironment.start_time_skipping() as env,
        Worker(
            env.client,
            task_queue="health-test",
            workflows=[HealthWorkflow],
            activities=[report_version],
        ),
    ):
        result = await env.client.execute_workflow(
            HealthWorkflow.run, id="health-1", task_queue="health-test"
        )
    assert result == frontdesk.__version__
