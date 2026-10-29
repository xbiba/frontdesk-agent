"""The second try (Albi, 2026-10-16): a reply a check sent back is written again by a stronger model,
and so is the rest of the turn.

Gemini 3.5 Flash-Lite leads the reception agent: about five times cheaper than 3.8 Flash and more
than twice as fast on the first-clinic cases (run agent-11-lite35). When the agent's reply is refused (by its
output validator, frontdesk.conversation.reception, or by the reply's own schema), the agent sends
the model a retry of its output; from that request on, the turn's requests go to the second model.
A tool's own retry (arguments it refused) stays with the first.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, RetryPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


class SecondTryModel(WrapperModel):
    def __init__(self, wrapped: Model, *, second: Model) -> None:
        super().__init__(wrapped)
        self._second = second

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        refused = sent_back(messages, model_request_parameters)
        model = self._second if refused else self.wrapped
        return await model.request(messages, model_settings, model_request_parameters)


def sent_back(messages: Sequence[ModelMessage], parameters: ModelRequestParameters) -> bool:
    """Whether a reply of this run was refused: a retry of the output among the messages. The
    earlier turns of the conversation come as text alone (reception.history_messages)."""
    outputs = {tool.name for tool in parameters.output_tools}
    return any(
        isinstance(part, RetryPromptPart) and (part.tool_name is None or part.tool_name in outputs)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )
