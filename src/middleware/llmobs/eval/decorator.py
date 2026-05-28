"""The ``@evaluator`` decorator and the shared result-normalization logic."""

import functools
from dataclasses import replace
from typing import Any, Callable, Optional

from .types import EvaluatorContext, EvaluatorResult, infer_metric_type


def _normalize_result(raw: Any, eval_name: str) -> Optional[EvaluatorResult]:
    """Coerce an evaluator's raw return into an ``EvaluatorResult`` (or ``None`` to skip).

    Shared by ``@evaluator`` and ``BaseEvaluator.__call__`` so the two cannot drift.
    """
    if raw is None:
        return None
    if isinstance(raw, EvaluatorResult):
        # Copy first — never mutate the object the caller handed back.
        result = replace(raw)
        if result.evaluator_name is None:
            result.evaluator_name = eval_name
        if result.metric_type is None and result.error is None and result.value is not None:
            try:
                result.metric_type = infer_metric_type(result.value)
            except TypeError:
                pass  # dict value with no metric_type — leave for caller/server
        return result
    if isinstance(raw, bool):
        return EvaluatorResult(value=raw, metric_type="boolean", evaluator_name=eval_name)
    if isinstance(raw, (int, float)):
        return EvaluatorResult(value=raw, metric_type="score", evaluator_name=eval_name)
    if isinstance(raw, str):
        return EvaluatorResult(value=raw, metric_type="categorical", evaluator_name=eval_name)
    if isinstance(raw, dict):
        # setdefault on a copy: avoids "got multiple values for keyword argument" if the user's
        # dict already carries evaluator_name, and lets their value win.
        fields = dict(raw)
        fields.setdefault("evaluator_name", eval_name)
        return EvaluatorResult(**fields)
    raise TypeError(
        f"Evaluator {eval_name!r} returned {type(raw).__name__}; "
        f"expected bool, int, float, str, dict, EvaluatorResult, or None"
    )


def evaluator(
    fn: Optional[Callable[[EvaluatorContext], Any]] = None,
    *,
    name: Optional[str] = None,
) -> Callable[..., Any]:
    """Mark a function as an evaluator.

    The wrapped function always returns an ``EvaluatorResult`` (or ``None`` to skip), and captures
    any exception into ``error``/``error_type`` rather than propagating.

    Usage::

        @evaluator
        def my_eval(ctx: EvaluatorContext) -> bool: ...

        @evaluator(name="custom_name")
        def my_eval(ctx: EvaluatorContext) -> float: ...
    """

    def decorate(f: Callable[[EvaluatorContext], Any]) -> Callable[..., Any]:
        eval_name = name or f.__name__

        @functools.wraps(f)
        def wrapped(ctx: EvaluatorContext) -> Optional[EvaluatorResult]:
            try:
                raw = f(ctx)
            except Exception as e:  # noqa: BLE001 — capture, never propagate
                return EvaluatorResult(
                    value=None,
                    error=str(e),
                    error_type=type(e).__name__,
                    evaluator_name=eval_name,
                )
            return _normalize_result(raw, eval_name)

        wrapped._is_evaluator = True  # type: ignore[attr-defined]
        wrapped._evaluator_name = eval_name  # type: ignore[attr-defined]
        return wrapped

    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
