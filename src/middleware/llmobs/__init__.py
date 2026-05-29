from openinference.instrumentation import (
    suppress_tracing,
    using_attributes,
    using_metadata,
    using_prompt_template,
    using_session,
    using_tags,
    using_user,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry.sdk.resources import Resource

from .eval import (
    Assessment,
    AsyncBaseEvaluator,
    AsyncLLMClient,
    AsyncLLMJudge,
    BaseEvaluator,
    BaseStructuredOutput,
    BooleanStructuredOutput,
    CategoricalStructuredOutput,
    EvalClient,
    EvaluatorContext,
    EvaluatorResult,
    LLMClient,
    LLMJudge,
    MetricType,
    ScoreStructuredOutput,
    ScoreValue,
    StructuredOutput,
    aevaluate_and_submit,
    async_evaluator,
    evaluate_and_submit,
    evaluator,
    export_current_span,
    flush_evaluations,
    format_schema_for_provider,
    submit_evaluation,
    submit_evaluation_error,
)
from .otel import (
    PROJECT_NAME,
    BatchSpanProcessor,
    GRPCSpanExporter,
    HTTPSpanExporter,
    Providers,
    SimpleSpanProcessor,
    TracerProvider,
    register,
)

# Import version from package metadata
try:
    from importlib.metadata import version

    __version__ = version("middleware-llmobs")
except Exception:
    __version__ = "unknown"

# TraceConfig is re-exported for convenience when wiring instrumentors manually. It may be absent
# on older openinference-instrumentation releases, so the import is optional.
try:
    from openinference.instrumentation import TraceConfig
except ImportError:
    TraceConfig = None  # type: ignore[assignment,misc]

__all__ = [
    "TracerProvider",
    "SimpleSpanProcessor",
    "BatchSpanProcessor",
    "HTTPSpanExporter",
    "GRPCSpanExporter",
    "Resource",
    "PROJECT_NAME",
    "Providers",
    "register",
    "__version__",
    "suppress_tracing",
    "using_attributes",
    "using_metadata",
    "using_prompt_template",
    "using_session",
    "using_tags",
    "using_user",
    "SpanAttributes",
    "OpenInferenceSpanKindValues",
    "OpenInferenceMimeTypeValues",
    "TraceConfig",
    # Evaluations
    "EvaluatorContext",
    "EvaluatorResult",
    "LLMClient",
    "AsyncLLMClient",
    "MetricType",
    "Assessment",
    "ScoreValue",
    "evaluator",
    "async_evaluator",
    "BaseEvaluator",
    "AsyncBaseEvaluator",
    "LLMJudge",
    "AsyncLLMJudge",
    "BaseStructuredOutput",
    "BooleanStructuredOutput",
    "ScoreStructuredOutput",
    "CategoricalStructuredOutput",
    "StructuredOutput",
    "format_schema_for_provider",
    "submit_evaluation",
    "submit_evaluation_error",
    "evaluate_and_submit",
    "aevaluate_and_submit",
    "flush_evaluations",
    "export_current_span",
    "EvalClient",
]
