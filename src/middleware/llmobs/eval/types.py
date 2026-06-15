"""Shared data contract for evaluations: contexts, results, and the LLM-judge client protocol."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Union

MetricType = Literal["score", "boolean", "categorical"]
Assessment = Literal["pass", "fail"]
ScoreValue = Union[float, int, bool, str, dict[str, Any]]


@dataclass
class EvaluatorContext:
    """Input handed to any evaluator. Built by the user or by ``evaluate_and_submit`` helpers."""

    input: Any
    output: Any
    expected_output: Optional[Any] = None
    retrieved_contexts: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluatorResult:
    """Output of any evaluator. ``metric_type`` is inferred from ``value`` when omitted.

    ``value`` is ``None`` only when ``error`` is set (the evaluator raised, producing no score).
    """

    value: Optional[ScoreValue]
    metric_type: Optional[MetricType] = None
    assessment: Optional[Assessment] = None
    reasoning: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    evaluator_name: Optional[str] = None


class LLMClient(Protocol):
    """User-supplied LLM caller for judge evaluators. The SDK never imports a provider library.

    Any callable matching this signature satisfies the contract.
    """

    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str,
        json_schema: Optional[dict[str, Any]] = None,
        model_params: Optional[dict[str, Any]] = None,
    ) -> str: ...


class AsyncLLMClient(Protocol):
    """Async sibling of :class:`LLMClient`. Used by :class:`AsyncLLMJudge`.

    The user provides any awaitable callable matching this signature (e.g. an adapter over
    ``openai.AsyncOpenAI`` or ``anthropic.AsyncAnthropic``); the SDK never imports a provider.
    """

    async def __call__(
        self,
        messages: list[dict[str, str]],
        model: str,
        json_schema: Optional[dict[str, Any]] = None,
        model_params: Optional[dict[str, Any]] = None,
    ) -> str: ...


def infer_metric_type(value: ScoreValue) -> MetricType:
    """Infer the metric type from a value.

    ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int``. A ``dict`` value
    has no inferable type (there is intentionally no ``"json"`` metric type) and raises.
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "score"
    if isinstance(value, str):
        return "categorical"
    raise TypeError("metric_type cannot be inferred from dict; set it explicitly")
