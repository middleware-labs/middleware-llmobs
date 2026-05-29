"""Evaluation surfaces for middleware-llmobs: submit scores and run local evaluators."""

from .base import AsyncBaseEvaluator, BaseEvaluator
from .client import (
    EvalClient,
    aevaluate_and_submit,
    evaluate_and_submit,
    export_current_span,
    flush_evaluations,
    submit_evaluation,
    submit_evaluation_error,
)
from .decorator import async_evaluator, evaluator
from .judge import AsyncLLMJudge, LLMJudge
from .schema import format_schema_for_provider
from .structured_output import (
    BaseStructuredOutput,
    BooleanStructuredOutput,
    CategoricalStructuredOutput,
    ScoreStructuredOutput,
    StructuredOutput,
)
from .types import (
    Assessment,
    AsyncLLMClient,
    EvaluatorContext,
    EvaluatorResult,
    LLMClient,
    MetricType,
    ScoreValue,
)

__all__ = [
    # types
    "EvaluatorContext",
    "EvaluatorResult",
    "LLMClient",
    "AsyncLLMClient",
    "MetricType",
    "Assessment",
    "ScoreValue",
    # evaluator authoring (sync + async)
    "evaluator",
    "async_evaluator",
    "BaseEvaluator",
    "AsyncBaseEvaluator",
    # LLM-judge authoring (sync + async)
    "LLMJudge",
    "AsyncLLMJudge",
    "BaseStructuredOutput",
    "BooleanStructuredOutput",
    "ScoreStructuredOutput",
    "CategoricalStructuredOutput",
    "StructuredOutput",
    "format_schema_for_provider",
    # submission
    "submit_evaluation",
    "submit_evaluation_error",
    "evaluate_and_submit",
    "aevaluate_and_submit",
    "flush_evaluations",
    "export_current_span",
    "EvalClient",
]
