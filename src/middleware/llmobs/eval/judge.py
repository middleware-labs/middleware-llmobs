"""``LLMJudge`` — a provider-agnostic LLM-judge evaluator driving a user-supplied ``LLMClient``.

The SDK imports no LLM provider library. ``LLMJudge`` only accepts a ``client``; there is no
``provider=`` shortcut and no client factory.
"""

import json
import re
from dataclasses import asdict
from typing import Any, Optional

from .base import BaseEvaluator
from .structured_output import (
    BooleanStructuredOutput,
    CategoricalStructuredOutput,
    ScoreStructuredOutput,
    StructuredOutput,
    compute_assessment,
)
from .types import EvaluatorContext, EvaluatorResult, LLMClient, MetricType

_TEMPLATE_PATTERN = re.compile(r"\{\{(.+?)\}\}")

_METRIC_TYPE_BY_OUTPUT: dict[type, MetricType] = {
    BooleanStructuredOutput: "boolean",
    ScoreStructuredOutput: "score",
    CategoricalStructuredOutput: "categorical",
}


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
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": self._render(self._user_prompt, context)})

        json_schema: Optional[dict[str, Any]] = None
        if self._structured_output is not None:
            json_schema = (
                self._structured_output
                if isinstance(self._structured_output, dict)
                else self._structured_output.to_json_schema()
            )

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
        return self._parse_response(raw)

    def _render(self, template: str, context: EvaluatorContext) -> str:
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

    def _parse_response(self, response: str) -> EvaluatorResult:
        """Parse the judge's JSON response into an ``EvaluatorResult``."""
        if not response or not isinstance(response, str):
            raise ValueError("Invalid response: expected non-empty string")
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON response: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"Invalid JSON response: expected object, got {type(data).__name__}")

        structured_output = self._structured_output

        # Custom dict schema: opaque to the SDK. No metric_type, no assessment.
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
