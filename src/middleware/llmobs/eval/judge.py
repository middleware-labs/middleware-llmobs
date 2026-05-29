"""``LLMJudge`` (sync) + ``AsyncLLMJudge`` — provider-agnostic LLM-judge evaluators.

Both drive a user-supplied client (:class:`LLMClient` / :class:`AsyncLLMClient`). The SDK never
imports a provider library; the rendering / parsing / assessment helpers are pure dict
manipulation and are shared between the two judges.
"""

import json
import re
from dataclasses import asdict
from typing import Any, Optional

from .base import AsyncBaseEvaluator, BaseEvaluator
from .structured_output import (
    BooleanStructuredOutput,
    CategoricalStructuredOutput,
    ScoreStructuredOutput,
    StructuredOutput,
    compute_assessment,
)
from .types import (
    AsyncLLMClient,
    EvaluatorContext,
    EvaluatorResult,
    LLMClient,
    MetricType,
)

_TEMPLATE_PATTERN = re.compile(r"\{\{(.+?)\}\}")

_METRIC_TYPE_BY_OUTPUT: dict[type, MetricType] = {
    BooleanStructuredOutput: "boolean",
    ScoreStructuredOutput: "score",
    CategoricalStructuredOutput: "categorical",
}


# --- shared helpers (pure functions — no I/O, used by both sync and async judges) ----------


def _build_messages(
    system_prompt: Optional[str], user_prompt: str, context: EvaluatorContext
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": _render(user_prompt, context)})
    return messages


def _json_schema_for(
    structured_output: Optional[StructuredOutput],
) -> Optional[dict[str, Any]]:
    if structured_output is None:
        return None
    if isinstance(structured_output, dict):
        return structured_output
    return structured_output.to_json_schema()


def _render(template: str, context: EvaluatorContext) -> str:
    """Substitute ``{{field.path}}`` placeholders with values from the context."""
    ctx = asdict(context)

    def resolve(path: str) -> Any:
        parts = path.split(".")
        value: Any = ctx.get(parts[0])
        for part in parts[1:]:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def replace(match: "re.Match[str]") -> str:
        value = resolve(match.group(1).strip())
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        return str(value)

    return _TEMPLATE_PATTERN.sub(replace, template)


def _parse_response(
    response: str, structured_output: Optional[StructuredOutput]
) -> EvaluatorResult:
    """Parse the judge's JSON response into an ``EvaluatorResult``."""
    if not response or not isinstance(response, str):
        raise ValueError("Invalid response: expected non-empty string")
    try:
        data = json.loads(response)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid JSON response: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Invalid JSON response: expected object, got {type(data).__name__}")

    # Custom dict schema or no schema: opaque to the SDK. No metric_type, no assessment.
    if isinstance(structured_output, dict) or structured_output is None:
        return EvaluatorResult(value=data, reasoning=data.get("reasoning"))

    label = structured_output.label
    value = data.get(label)

    if isinstance(structured_output, BooleanStructuredOutput):
        if not isinstance(value, bool):
            raise ValueError(f"Expected boolean, got {type(value).__name__}")
    elif isinstance(structured_output, ScoreStructuredOutput):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Expected number, got {type(value).__name__}")
    elif isinstance(structured_output, CategoricalStructuredOutput):
        if not isinstance(value, str):
            raise ValueError(f"Expected string, got {type(value).__name__}")

    metric_type = _METRIC_TYPE_BY_OUTPUT[type(structured_output)]
    reasoning = data.get("reasoning") if structured_output.reasoning else None

    return EvaluatorResult(
        value=value,
        metric_type=metric_type,
        reasoning=reasoning,
        assessment=compute_assessment(structured_output, value),
        metadata={"raw_response": data},
    )


class LLMJudge(BaseEvaluator):
    """Evaluate an output by calling a user-supplied LLM as the judge.

    The user provides ``client`` (any callable matching :class:`LLMClient`), a prompt template
    using ``{{field.path}}`` placeholders, and an optional :class:`StructuredOutput` describing the
    expected response. ``LLMJudge`` renders the prompt, calls the client with the JSON schema,
    parses the response, and returns an :class:`EvaluatorResult`.
    """

    def __init__(
        self,
        *,
        client: LLMClient,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[StructuredOutput] = None,
        model_params: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        if client is None:
            raise ValueError("LLMJudge requires a user-supplied 'client' (LLMClient).")
        super().__init__(name=name or "llm_judge")
        self._client = client
        self._model = model
        self._user_prompt = user_prompt
        self._system_prompt = system_prompt
        self._structured_output = structured_output
        self._model_params = model_params

    def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        messages = _build_messages(self._system_prompt, self._user_prompt, context)
        json_schema = _json_schema_for(self._structured_output)

        # Call convention matches the public LLMClient signature: first two args positional,
        # optionals always by keyword.
        raw = self._client(
            messages,
            self._model,
            json_schema=json_schema,
            model_params=self._model_params,
        )
        if self._structured_output is None:
            # No schema: treat the raw response as a free-form categorical value.
            return EvaluatorResult(value=raw, metric_type="categorical")
        return _parse_response(raw, self._structured_output)


class AsyncLLMJudge(AsyncBaseEvaluator):
    """Async sibling of :class:`LLMJudge` driving a user-supplied :class:`AsyncLLMClient`.

    Behaviour is identical to :class:`LLMJudge` (prompt rendering, JSON-schema generation,
    response parsing, pass/fail assessment) except the judge LLM call is ``await``\\ ed instead
    of blocking. Use this in async apps so the event loop can serve other work while the judge
    model generates.
    """

    def __init__(
        self,
        *,
        client: AsyncLLMClient,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[StructuredOutput] = None,
        model_params: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        if client is None:
            raise ValueError("AsyncLLMJudge requires a user-supplied 'client' (AsyncLLMClient).")
        super().__init__(name=name or "llm_judge")
        self._client = client
        self._model = model
        self._user_prompt = user_prompt
        self._system_prompt = system_prompt
        self._structured_output = structured_output
        self._model_params = model_params

    async def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        messages = _build_messages(self._system_prompt, self._user_prompt, context)
        json_schema = _json_schema_for(self._structured_output)

        raw = await self._client(
            messages,
            self._model,
            json_schema=json_schema,
            model_params=self._model_params,
        )
        if self._structured_output is None:
            return EvaluatorResult(value=raw, metric_type="categorical")
        return _parse_response(raw, self._structured_output)
