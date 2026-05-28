"""``EvalClient`` and module-level helpers: validate, auto-bind to the active span, emit via OTLP.

Evaluations are shipped as OpenTelemetry log records plus gauge metrics through the **global**
logger/meter providers that ``register()`` wires up in ``otel.py``. This module does not create
providers or exporters — it only emits. Span/trace correlation uses the log record's native
``trace_id``/``span_id`` fields
"""

import json
import re
import time
from typing import Any, Callable, Optional

from opentelemetry import trace as trace_api
from opentelemetry._logs import SeverityNumber, get_logger, get_logger_provider
from opentelemetry.metrics import get_meter, get_meter_provider
from opentelemetry.sdk._logs._internal import LogRecord  # type: ignore[attr-defined]

from ..settings import get_env_service_name
from .types import (
    Assessment,
    EvaluatorContext,
    EvaluatorResult,
    MetricType,
    ScoreValue,
    infer_metric_type,
)

_LABEL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def _resolve_ml_app(ml_app: Optional[str]) -> str:
    # get_env_service_name() always returns a non-empty str (defaults to "default").
    return ml_app or get_env_service_name()


def _coerce_score(
    metric_type: MetricType, value: ScoreValue, assessment: Optional[Assessment]
) -> float:
    """Numeric value for the score gauge/attribute.
    - ``score`` → ``float(value)``.
    - ``boolean`` → raw model output (``True`` → ``1.0``, else ``0.0``).
    - ``categorical`` → pass/fail (``1.0`` when ``assessment == "pass"``, else ``0.0``).
    """
    if metric_type == "score":
        return float(value)  # type: ignore[arg-type]
    if metric_type == "boolean":
        return 1.0 if value is True else 0.0
    if metric_type == "categorical":
        return 1.0 if assessment == "pass" else 0.0
    return 0.0


def _log_context(
    trace_id: Optional[str], span_id: Optional[str]
) -> Any:
    """Build the ``context=`` payload for ``LogRecord`` from raw hex IDs.

    Returns ``None`` when neither ID is given (unattached eval). Otherwise wraps a synthetic
    ``SpanContext`` (with a placeholder span_id if span scope wasn't set) so OTel can populate the
    record's ``trace_id``/``span_id`` fields without the deprecated direct kwargs.
    """
    if not trace_id and not span_id:
        return None
    return trace_api.set_span_in_context(
        trace_api.NonRecordingSpan(
            trace_api.SpanContext(
                trace_id=int(trace_id, 16) if trace_id else 0,
                span_id=int(span_id, 16) if span_id else 0,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED),
            )
        )
    )


def _resolve_target(
    span_id: Optional[str],
    trace_id: Optional[str],
    join_on_tag: Optional[tuple[str, str]],
) -> tuple[Optional[str], Optional[str]]:
    """Resolve (trace_id, span_id): explicit ref > tag > active span > unattached."""
    resolved_trace_id, resolved_span_id = trace_id, span_id
    if join_on_tag is None and span_id is None and trace_id is None:
        ctx = trace_api.get_current_span().get_span_context()
        if ctx.is_valid:
            resolved_trace_id = trace_api.format_trace_id(ctx.trace_id)
            resolved_span_id = trace_api.format_span_id(ctx.span_id)
    return resolved_trace_id, resolved_span_id


class EvalClient:
    """Submits evaluations to Middleware over the global OTel logger/meter providers."""

    def __init__(self, *, ml_app: Optional[str] = None):
        self._ml_app = ml_app
        self._logger: Any = None
        self._gauges: dict[str, Any] = {}

    def _ensure_instruments(self) -> None:
        if self._logger is not None:
            return
        self._logger = get_logger("middleware.llmobs.eval")
        meter = get_meter("middleware.llmobs.eval")
        self._gauges = {
            "count": meter.create_gauge("gen_ai.evaluations.count", unit="1"),
            "score": meter.create_gauge("gen_ai.evaluations.score", unit="1"),
            "outcome": meter.create_gauge("gen_ai.evaluations.outcome", unit="1"),
            "cost": meter.create_gauge("gen_ai.evaluations.cost.usd", unit="USD"),
        }

    # -- public API -------------------------------------------------------

    def submit_evaluation(
        self,
        *,
        label: str,
        value: ScoreValue,
        metric_type: Optional[MetricType] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        join_on_tag: Optional[tuple[str, str]] = None,
        assessment: Optional[Assessment] = None,
        reasoning: Optional[str] = None,
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None,
        cost_usd: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
        timestamp_ms: Optional[int] = None,
        ml_app: Optional[str] = None,
    ) -> None:
        metric_type = self._validate(
            label=label,
            value=value,
            metric_type=metric_type,
            span_id=span_id,
            trace_id=trace_id,
            join_on_tag=join_on_tag,
            assessment=assessment,
            metadata=metadata,
            tags=tags,
        )

        resolved_trace_id, resolved_span_id = _resolve_target(
            span_id, trace_id, join_on_tag
        )

        self._emit(
            label=label,
            value=value,
            metric_type=metric_type,
            trace_id=resolved_trace_id,
            span_id=resolved_span_id,
            join_on_tag=join_on_tag,
            assessment=assessment,
            reasoning=reasoning,
            score_label=label,  # server contract slot; always equal to the eval label
            judge_provider=judge_provider or "",
            judge_model=judge_model or "",
            cost_usd=cost_usd,
            metadata=metadata or {},
            tags=tags or {},
            timestamp_ms=timestamp_ms,
            ml_app=_resolve_ml_app(ml_app or self._ml_app),
        )

    def submit_evaluation_error(
        self,
        *,
        label: str,
        error: str,
        error_type: Optional[str] = None,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        join_on_tag: Optional[tuple[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, str]] = None,
        timestamp_ms: Optional[int] = None,
        ml_app: Optional[str] = None,
    ) -> None:
        """Record a *failed* evaluation: an error log + an ``outcome="error"`` metric, no score.

        Mirrors the server-side eval-error payload so SDK and online eval failures look the same in
        dashboards. Target resolution matches :meth:`submit_evaluation` (auto-binds to the active
        span when no IDs/tag are given).
        """
        if not label or not _LABEL_RE.match(label) or len(label) > 200:
            raise ValueError(
                "label must match ^[a-zA-Z][a-zA-Z0-9_]*$ and be at most 200 characters."
            )
        if join_on_tag is not None and (span_id is not None or trace_id is not None):
            raise ValueError("Pass either join_on_tag or span_id/trace_id, not both.")
        if (span_id is None) != (trace_id is None):
            raise ValueError("span_id and trace_id must be provided together.")
        for name, obj in (("metadata", metadata), ("tags", tags)):
            if obj is not None:
                try:
                    json.dumps(obj)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"{name} must be JSON-serializable: {e}") from e

        resolved_trace_id, resolved_span_id = _resolve_target(
            span_id, trace_id, join_on_tag
        )
        self._emit_error(
            label=label,
            error=error,
            error_type=error_type or "EvaluationError",
            trace_id=resolved_trace_id,
            span_id=resolved_span_id,
            join_on_tag=join_on_tag,
            metadata=metadata or {},
            tags=tags or {},
            timestamp_ms=timestamp_ms,
            ml_app=_resolve_ml_app(ml_app or self._ml_app),
        )

    def evaluate_and_submit(
        self,
        evaluator_fn: Callable[[EvaluatorContext], Optional[EvaluatorResult]],
        context: EvaluatorContext,
    ) -> Optional[EvaluatorResult]:
        result = evaluator_fn(context)
        if result is None:
            return result
        # An errored evaluator is reported as a failed eval, not silently dropped.
        if result.error:
            self.submit_evaluation_error(
                span_id=context.span_id,
                trace_id=context.trace_id,
                label=result.evaluator_name or "evaluation",
                error=result.error,
                error_type=result.error_type,
                metadata=result.metadata,
                tags=result.tags,
            )
            return result
        if result.value is None:
            return result
        self.submit_evaluation(
            span_id=context.span_id,
            trace_id=context.trace_id,
            label=result.evaluator_name or "evaluation",
            value=result.value,
            metric_type=result.metric_type,
            assessment=result.assessment,
            reasoning=result.reasoning,
            metadata=result.metadata,
            tags=result.tags,
        )
        return result

    def flush(self, timeout_millis: int = 30_000) -> bool:
        """Force-flush the global logger + meter providers.

        Returns ``True`` only if both report success within the timeout. Providers that don't
        expose ``force_flush`` (e.g. proxy providers when ``register()`` was never called) are
        treated as a no-op success.
        """
        ok = True
        for provider in (get_logger_provider(), get_meter_provider()):
            flush = getattr(provider, "force_flush", None)
            if callable(flush):
                ok = bool(flush(timeout_millis)) and ok
        return ok

    # -- validation -------------------------------------------------------

    @staticmethod
    def _validate(
        *,
        label: str,
        value: ScoreValue,
        metric_type: Optional[MetricType],
        span_id: Optional[str],
        trace_id: Optional[str],
        join_on_tag: Optional[tuple[str, str]],
        assessment: Optional[Assessment],
        metadata: Optional[dict[str, Any]],
        tags: Optional[dict[str, str]],
    ) -> MetricType:
        # Targets are optional (auto-bind / unattached). Reject only contradictory/incomplete ones.
        if join_on_tag is not None and (span_id is not None or trace_id is not None):
            raise ValueError("Pass either join_on_tag or span_id/trace_id, not both.")
        if (span_id is None) != (trace_id is None):
            raise ValueError("span_id and trace_id must be provided together.")
        if join_on_tag is not None:
            if (
                not isinstance(join_on_tag, tuple)
                or len(join_on_tag) != 2
                or not all(isinstance(x, str) and x for x in join_on_tag)
            ):
                raise ValueError("join_on_tag must be a 2-tuple of non-empty strings.")

        if not label or not _LABEL_RE.match(label) or len(label) > 200:
            raise ValueError(
                "label must match ^[a-zA-Z][a-zA-Z0-9_]*$ and be at most 200 characters."
            )

        resolved = metric_type or infer_metric_type(value)
        if resolved == "score":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "value must be int or float (not bool) for a score metric."
                )
        elif resolved == "boolean":
            if not isinstance(value, bool):
                raise TypeError("value must be a bool for a boolean metric.")
        elif resolved == "categorical":
            if not isinstance(value, str):
                raise TypeError("value must be a str for a categorical metric.")

        if assessment is not None and assessment not in ("pass", "fail"):
            raise ValueError("assessment must be 'pass' or 'fail'.")

        for name, obj in (("metadata", metadata), ("tags", tags)):
            if obj is not None:
                try:
                    json.dumps(obj)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"{name} must be JSON-serializable: {e}") from e

        return resolved

    # -- emit -------------------------------------------------------------

    def _emit(
        self,
        *,
        label: str,
        value: ScoreValue,
        metric_type: MetricType,
        trace_id: Optional[str],
        span_id: Optional[str],
        join_on_tag: Optional[tuple[str, str]],
        assessment: Optional[Assessment],
        reasoning: Optional[str],
        score_label: str,
        judge_provider: str,
        judge_model: str,
        cost_usd: Optional[float],
        metadata: dict[str, Any],
        tags: dict[str, str],
        timestamp_ms: Optional[int],
        ml_app: str,
    ) -> None:
        self._ensure_instruments()

        score = _coerce_score(metric_type, value, assessment)
        now_ns = (timestamp_ms * 1_000_000) if timestamp_ms else time.time_ns()
        # Server-side body uses ``verdict``/``explanation`` for the same data the SDK exposes as
        # ``assessment``/``reasoning``. Map them here so callers don't have to pass both.
        verdict = assessment or ""
        explanation = reasoning or ""

        body: dict[str, Any] = {
            "eval_name": label,
            "ml_app": ml_app,
            "score_value": value,
            "metric_type": metric_type,
            "score_label": score_label,
            "judge_provider": judge_provider,
            "judge_model": judge_model,
        }
        if assessment is not None:
            body["verdict"] = verdict
        if reasoning is not None:
            body["explanation"] = explanation
        if cost_usd is not None:
            body["cost_usd"] = float(cost_usd)
        if metadata:
            body["metadata"] = metadata
        if tags:
            body["tags"] = tags

        attributes: dict[str, Any] = {
            "gen_ai.evaluation.name": label,
            "gen_ai.evaluation.score.label": score_label,
            "gen_ai.evaluation.explanation": explanation[:20000],
            "gen_ai.evaluation.verdict": verdict,
            "gen_ai.evaluation.score.value": score,
        }

        if judge_provider:
            attributes["eval.model.provider"] = judge_provider
        if judge_model:
            attributes["eval.model.name"] = judge_model
        if cost_usd is not None:
            attributes["gen_ai.evaluation.cost.usd"] = float(cost_usd)
        if trace_id is not None:
            attributes["eval.target.trace_id"] = trace_id
        if span_id is not None:
            attributes["eval.target.span_id"] = span_id
        if join_on_tag is not None:
            attributes["gen_ai.evaluation.tag_key"] = join_on_tag[0]
            attributes["gen_ai.evaluation.tag_value"] = join_on_tag[1]
        for k, v in tags.items():
            attributes[k] = v

        record = LogRecord(
            timestamp=now_ns,
            observed_timestamp=now_ns,
            context=_log_context(trace_id, span_id),
            severity_text="INFO",
            severity_number=SeverityNumber.INFO,
            body=json.dumps(body, ensure_ascii=False),
            attributes=attributes,
        )
        self._logger.emit(record)

        outcome = assessment or "submitted"
        common = {
            "eval_name": label,
            "score_label": score_label,
            "verdict": verdict,
            "model": judge_model,
            "provider": judge_provider,
        }
        self._gauges["count"].set(1, common)
        self._gauges["score"].set(score, common)
        self._gauges["outcome"].set(1, {**common, "outcome": outcome})
        if cost_usd is not None:
            self._gauges["cost"].set(float(cost_usd), common)

    def _emit_error(
        self,
        *,
        label: str,
        error: str,
        error_type: str,
        trace_id: Optional[str],
        span_id: Optional[str],
        join_on_tag: Optional[tuple[str, str]],
        metadata: dict[str, Any],
        tags: dict[str, str],
        timestamp_ms: Optional[int],
        ml_app: str,
    ) -> None:
        self._ensure_instruments()

        now_ns = (timestamp_ms * 1_000_000) if timestamp_ms else time.time_ns()
        msg = (error or "")[:8000]

        body: dict[str, Any] = {
            "outcome": "error",
            "eval_name": label,
            "ml_app": ml_app,
            "error.type": error_type,
            "error.message": msg,
            "source": "sdk",
        }
        if metadata:
            body["metadata"] = metadata
        if tags:
            body["tags"] = tags

        attributes: dict[str, Any] = {
            "gen_ai.evaluation.outcome": "error",
            "gen_ai.evaluation.name": label,
            # OpenTelemetry exception semconv + an explicit error.type alias for dashboards.
            "exception.type": error_type,
            "exception.message": msg,
            "error.type": error_type,
        }
        if trace_id is not None:
            attributes["eval.target.trace_id"] = trace_id
        if span_id is not None:
            attributes["eval.target.span_id"] = span_id
        if join_on_tag is not None:
            attributes["gen_ai.evaluation.tag_key"] = join_on_tag[0]
            attributes["gen_ai.evaluation.tag_value"] = join_on_tag[1]
        for k, v in tags.items():
            attributes[k] = v

        record = LogRecord(
            timestamp=now_ns,
            observed_timestamp=now_ns,
            context=_log_context(trace_id, span_id),
            severity_text="WARN",
            severity_number=SeverityNumber.WARN,
            body=json.dumps(body, ensure_ascii=False),
            attributes=attributes,
        )
        self._logger.emit(record)

        common = {
            "eval_name": label,
            "score_label": "",
            "verdict": "",
            "error.type": error_type,
            "gen_ai.evaluation.outcome": "error",
        }
        self._gauges["count"].set(1, common)
        self._gauges["outcome"].set(1, {**common, "outcome": "error"})


# -- module-level convenience over a process-wide default client ----------

_default_client: Optional[EvalClient] = None


def _get_default_client() -> EvalClient:
    global _default_client
    if _default_client is None:
        _default_client = EvalClient()
    return _default_client


def submit_evaluation(**kwargs: Any) -> None:
    """Module-level :meth:`EvalClient.submit_evaluation` over a default client."""
    _get_default_client().submit_evaluation(**kwargs)


def submit_evaluation_error(**kwargs: Any) -> None:
    """Module-level :meth:`EvalClient.submit_evaluation_error` over a default client."""
    _get_default_client().submit_evaluation_error(**kwargs)


def evaluate_and_submit(
    evaluator_fn: Callable[[EvaluatorContext], Optional[EvaluatorResult]],
    context: EvaluatorContext,
) -> Optional[EvaluatorResult]:
    """Module-level :meth:`EvalClient.evaluate_and_submit` over a default client."""
    return _get_default_client().evaluate_and_submit(evaluator_fn, context)


def flush_evaluations(timeout_millis: int = 30_000) -> bool:
    """Force-flush the global logger + meter providers used for evaluations."""
    return _get_default_client().flush(timeout_millis)


def export_current_span() -> Optional[dict[str, str]]:
    """Return ``{"span_id", "trace_id"}`` for the active OTel span, or ``None`` outside one."""
    ctx = trace_api.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return {
        "span_id": trace_api.format_span_id(ctx.span_id),
        "trace_id": trace_api.format_trace_id(ctx.trace_id),
    }
