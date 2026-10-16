"""The second try (Albi, 2026-10-16): a reply a check sends back is written again by a stronger model,
and so is the rest of the turn; a tool's own retry stays with the first."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent, ModelResponse, ModelRetry, RunContext, TextPart, ToolCallPart
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

from frontdesk.model_gateway.second_try import SecondTryModel


class Reply(BaseModel):
    text: str


def _model(
    name: str, steps: list[ModelResponse | dict[str, str]], seen: list[str]
) -> FunctionModel:
    """Answers with its steps in order: a response as it is, a dict as the final reply."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(name)
        step = steps.pop(0)
        if isinstance(step, dict):
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, step)])
        return step

    return FunctionModel(respond, model_name=name)


def _agent(first: FunctionModel, second: FunctionModel) -> Agent[None, Reply]:
    agent = Agent(SecondTryModel(first, second=second), output_type=Reply, retries=1)

    @agent.output_validator
    def checked(ctx: RunContext[None], output: Reply) -> Reply:
        if output.text == "bad":
            raise ModelRetry("Rewrite the reply")
        return output

    @agent.tool_plain
    def lookup(day: str) -> str:
        if day == "never":
            raise ModelRetry("no such day")
        return f"free on {day}"

    return agent


async def test_a_reply_without_problems_is_the_first_model_s() -> None:
    seen: list[str] = []
    first = _model("first", [{"text": "good"}], seen)
    result = await _agent(first, _model("second", [], seen)).run("hi")
    assert (result.output.text, seen) == ("good", ["first"])


async def test_a_reply_sent_back_is_written_again_by_the_second_model() -> None:
    seen: list[str] = []
    first = _model("first", [{"text": "bad"}], seen)
    second = _model("second", [{"text": "good"}], seen)
    result = await _agent(first, second).run("hi")
    assert (result.output.text, seen) == ("good", ["first", "second"])


async def test_after_a_reply_sent_back_the_second_model_ends_the_turn() -> None:
    seen: list[str] = []
    call = ModelResponse(parts=[ToolCallPart("lookup", {"day": "friday"})])
    first = _model("first", [{"text": "bad"}], seen)
    second = _model("second", [call, {"text": "good"}], seen)
    result = await _agent(first, second).run("hi")
    assert (result.output.text, seen) == ("good", ["first", "second", "second"])


async def test_a_tool_s_own_retry_stays_with_the_first_model() -> None:
    seen: list[str] = []
    call = ModelResponse(parts=[ToolCallPart("lookup", {"day": "never"})])
    first = _model("first", [call, {"text": "good"}], seen)
    result = await _agent(first, _model("second", [], seen)).run("hi")
    assert (result.output.text, seen) == ("good", ["first", "first"])


async def test_a_reply_with_no_tool_call_is_written_again_by_the_second_model() -> None:
    # Final review of #48, M4: the model answered with text where the reply's tool was due.
    seen: list[str] = []
    first = _model("first", [ModelResponse(parts=[TextPart("ciao")])], seen)
    second = _model("second", [{"text": "good"}], seen)
    result = await _agent(first, second).run("hi")
    assert (result.output.text, seen) == ("good", ["first", "second"])
