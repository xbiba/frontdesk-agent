"""M0 health workflow: proves client, worker and activity wiring."""

from __future__ import annotations

from datetime import timedelta

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
    import frontdesk


@activity.defn
async def report_version() -> str:
    return frontdesk.__version__


@workflow.defn
class HealthWorkflow:
    @workflow.run
    async def run(self) -> str:
        return await workflow.execute_activity(
            report_version, start_to_close_timeout=timedelta(seconds=10)
        )
