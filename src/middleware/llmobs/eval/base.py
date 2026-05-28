"""``BaseEvaluator`` for stateful or configurable evaluators (e.g. LLM judges)."""

from abc import ABC, abstractmethod
from typing import Optional

from .decorator import _normalize_result
from .types import EvaluatorContext, EvaluatorResult


class BaseEvaluator(ABC):
    """Base class for evaluators that hold state (a threshold, an injected LLM client, …).

    Subclasses implement ``evaluate``; ``__call__`` applies the same normalization and error
    capture as the ``@evaluator`` decorator.

    Example::

        class SimilarityEvaluator(BaseEvaluator):
            def __init__(self, threshold=0.8):
                super().__init__(name="similarity")
                self.threshold = threshold

            def evaluate(self, context: EvaluatorContext):
                score = cosine_sim(context.output, context.expected_output)
                return EvaluatorResult(
                    value=score,
                    reasoning=f"Score: {score:.2f}",
                    assessment="pass" if score >= self.threshold else "fail",
                )

    Note: ``evaluate`` may be called concurrently from multiple threads.
    Don't mutate instance attributes inside it; use local variables.

    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(self, context: EvaluatorContext) -> EvaluatorResult:
        """Return an ``EvaluatorResult``; ``__call__`` fills in defaults and captures errors."""
        ...

    def __call__(self, context: EvaluatorContext) -> Optional[EvaluatorResult]:
        try:
            raw = self.evaluate(context)
        except Exception as e:  # noqa: BLE001 — capture, never propagate
            return EvaluatorResult(
                value=None,
                error=str(e),
                error_type=type(e).__name__,
                evaluator_name=self.name,
            )
        return _normalize_result(raw, self.name)

    # Parity with decorated functions so callers can treat both uniformly.
    @property
    def _is_evaluator(self) -> bool:
        return True

    @property
    def _evaluator_name(self) -> str:
        return self.name
