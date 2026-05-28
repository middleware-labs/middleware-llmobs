"""Evaluation surfaces for middleware-llmobs: submit scores and run local evaluators."""

from .base import BaseEvaluator
from .client import (
    EvalClient,
    evaluate_and_submit,
    export_current_span,
    flush_evaluations,
    submit_evaluation,
    submit_evaluation_error,
)
from .decorator import evaluator
from .judge import LLMJudge
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
    "MetricType",
    "Assessment",
    "ScoreValue",
    # evaluator authoring
    "evaluator",
    "BaseEvaluator",
    # LLM-judge authoring
    "LLMJudge",
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
    "flush_evaluations",
    "export_current_span",
    "EvalClient",
]
