"""``BaseEvaluator`` (sync) + ``AsyncBaseEvaluator`` for stateful or configurable evaluators."""

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
    def evaluate(self, context: EvaluatorContext) -> Optional[EvaluatorResult]:
        """Return an ``EvaluatorResult`` (or ``None`` to skip); ``__call__`` wraps with capture."""
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


class AsyncBaseEvaluator(ABC):
    """Async sibling of :class:`BaseEvaluator`.

    Subclasses implement ``async def evaluate``; ``__call__`` is awaitable, applies the same
    error capture as the sync base, and runs the shared ``_normalize_result`` post-processor.

    Use this when your evaluator needs to ``await`` an :class:`AsyncLLMClient`, an async
    embedding service, or any I/O that's natural to express as a coroutine. The non-I/O parts
    of evaluation (template rendering, JSON parsing, threshold computation) stay sync because
    they're pure dict manipulation.

    Example::

        class AsyncSimilarityEvaluator(AsyncBaseEvaluator):
            def __init__(self, embed_client, threshold=0.8):
                super().__init__(name="similarity")
                self.embed_client = embed_client
                self.threshold = threshold

            async def evaluate(self, context):
                a, b = await self.embed_client.embed_pair(context.output, context.expected_output)
                score = cosine(a, b)
                return EvaluatorResult(
                    value=score,
                    assessment="pass" if score >= self.threshold else "fail",
                )

    Note: ``evaluate`` may be awaited concurrently from many tasks. Don't mutate instance
    attributes inside it; use local variables.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def evaluate(self, context: EvaluatorContext) -> Optional[EvaluatorResult]:
        """Return an ``EvaluatorResult`` (or ``None`` to skip); ``__call__`` wraps with capture."""
        ...

    async def __call__(self, context: EvaluatorContext) -> Optional[EvaluatorResult]:
        try:
            raw = await self.evaluate(context)
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
    def _is_async_evaluator(self) -> bool:
        return True

    @property
    def _evaluator_name(self) -> str:
        return self.name
