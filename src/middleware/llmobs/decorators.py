"""Span decorators for LLM operations.

Each decorator opens a span around a function and tags it with a GenAI ``gen_ai.operation.name`` so
the operation shows up in Middleware:

* ``retriever`` — a retrieval (RAG) step. Call ``annotate_rag`` inside to record the query and the
  documents you got back.
* ``task`` — a generic step in your pipeline.
* ``embedding`` — an embedding call. Pass ``model_name`` (and optionally ``model_provider``).

``annotate_rag`` sets the retrieval query and/or documents on the current span. Use it inside a
``retriever`` function, or in your own ``start_as_current_span`` block. The documents become a JSON
string (OTel attributes can't hold a list of objects). You can pass plain dicts, or any object with
``id`` / ``score`` / ``text`` (or ``content``) / ``name`` / ``metadata`` attributes (e.g. a
vector-store hit) — both are turned into dicts for you::

    @retriever
    def get_relevant_docs(question):
        docs = [{"id": d.id, "score": d.score, "text": d.text} for d in vector_db.search(question)]
        annotate_rag(query=question, documents=docs)
        return docs

All decorators work on sync and async functions, and can be used bare (``@task``) or with arguments
(``@task(name="...")``). Session/user/tag context (from ``using_session`` etc.) is already on the
span when it starts, so these helpers don't touch it.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import wrapt
from opentelemetry import trace as trace_api
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.trace import Span
from typing_extensions import ParamSpec

ParametersType = ParamSpec("ParametersType")
ReturnType = TypeVar("ReturnType")

# GenAI attribute names (the keys we write on the span).
GEN_AI_OPERATION_NAME = gen_ai_attributes.GEN_AI_OPERATION_NAME
GEN_AI_REQUEST_MODEL = gen_ai_attributes.GEN_AI_REQUEST_MODEL
GEN_AI_PROVIDER_NAME = gen_ai_attributes.GEN_AI_PROVIDER_NAME
GEN_AI_RETRIEVAL_QUERY_TEXT = gen_ai_attributes.GEN_AI_RETRIEVAL_QUERY_TEXT
GEN_AI_RETRIEVAL_DOCUMENTS = gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS

# gen_ai.operation.name values. "embeddings"/"retrieval" come from the semconv enum; "task" has no
# enum value, so we use the literal.
_RETRIEVAL_OPERATION = gen_ai_attributes.GenAiOperationNameValues.RETRIEVAL.value
_EMBEDDINGS_OPERATION = gen_ai_attributes.GenAiOperationNameValues.EMBEDDINGS.value
_TASK_OPERATION = "task"

# Default model provider when the caller doesn't pass one (matches the LLMObs convention).
_DEFAULT_MODEL_PROVIDER = "custom"

# A document can be a dict or any object we read the known fields off of.
DocumentLike = Union[Mapping[str, Any], Any]

# Fields we read off a document object, in the order we want them.
_DOCUMENT_FIELDS = ("id", "score", "text", "content", "name", "metadata")


def _normalize_document(document: DocumentLike) -> Dict[str, Any]:
    """Turn one document into a plain dict.

    A dict is returned as-is. For other objects, we copy whatever known fields are set. If none
    are, we keep the document as text so it isn't dropped.
    """
    if isinstance(document, Mapping):
        return dict(document)
    normalized: Dict[str, Any] = {}
    for field in _DOCUMENT_FIELDS:
        value = getattr(document, field, None)
        if value is not None:
            normalized[field] = value
    if not normalized:
        return {"text": str(document)}
    return normalized


def _serialize_documents(documents: Sequence[DocumentLike]) -> str:
    """Turn the documents into the JSON string we store on the span."""
    normalized: List[Dict[str, Any]] = [_normalize_document(doc) for doc in documents]
    return json.dumps(normalized, default=str, ensure_ascii=False)


def _set_retrieval_attributes(
    span: Span,
    *,
    query: Optional[str] = None,
    documents: Optional[Sequence[DocumentLike]] = None,
) -> None:
    """Write the retrieval attributes on ``span``.

    Marks it as a retrieval operation and sets the query and/or documents. You can pass either one
    on its own.
    """
    if not span.is_recording():
        return
    span.set_attribute(GEN_AI_OPERATION_NAME, _RETRIEVAL_OPERATION)
    if query is not None:
        span.set_attribute(GEN_AI_RETRIEVAL_QUERY_TEXT, query)
    if documents is not None:
        span.set_attribute(GEN_AI_RETRIEVAL_DOCUMENTS, _serialize_documents(documents))


def annotate_rag(
    *,
    query: Optional[str] = None,
    documents: Optional[Sequence[DocumentLike]] = None,
    span: Optional[Span] = None,
) -> None:
    """Record the retrieval ``query`` and/or ``documents`` on the span.

    Call it inside a ``retriever`` function or your own span. It uses the current span by default;
    pass ``span=`` to target another one. See the module docstring for the document shapes you can
    pass::

        with tracer.start_as_current_span("get_relevant_docs"):
            docs = vector_db.search(question)
            annotate_rag(
                query=question,
                documents=[{"id": d.id, "score": d.score, "text": d.text} for d in docs],
            )
    """
    target = span if span is not None else trace_api.get_current_span()
    _set_retrieval_attributes(target, query=query, documents=documents)


def retriever(
    wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
    /,
    *,
    name: Optional[str] = None,
) -> Union[
    Callable[ParametersType, ReturnType],
    Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
]:
    """Open a retrieval span around the decorated function.

    The decorator only opens the span; call ``annotate_rag`` inside to record the query and
    documents. Use it as ``@retriever`` or ``@retriever(name="...")``.

    Example::

        @retriever
        def get_relevant_docs(question):
            hits = vector_db.search(question)
            docs = [{"id": h.id, "score": h.score, "text": h.text} for h in hits]
            annotate_rag(query=question, documents=docs)
            return docs
    """

    @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
    def sync_wrapper(
        wrapped: Callable[ParametersType, ReturnType],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> ReturnType:
        span_name = name or _span_name(instance, wrapped)
        with trace_api.get_tracer(__name__).start_as_current_span(span_name) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, _RETRIEVAL_OPERATION)
            return wrapped(*args, **kwargs)

    @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
    async def async_wrapper(
        wrapped: Callable[ParametersType, Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        span_name = name or _span_name(instance, wrapped)
        with trace_api.get_tracer(__name__).start_as_current_span(span_name) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, _RETRIEVAL_OPERATION)
            return await wrapped(*args, **kwargs)

    def _select(fn: Callable[ParametersType, ReturnType]) -> Callable[ParametersType, ReturnType]:
        if asyncio.iscoroutinefunction(fn):
            return cast("Callable[ParametersType, ReturnType]", async_wrapper(fn))
        return cast("Callable[ParametersType, ReturnType]", sync_wrapper(fn))

    if wrapped_function is not None:
        return _select(wrapped_function)
    return _select


def task(
    wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
    /,
    *,
    name: Optional[str] = None,
) -> Union[
    Callable[ParametersType, ReturnType],
    Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
]:
    """Open a task span around the decorated function.

    For a generic step in your pipeline. Use it as ``@task`` or ``@task(name="...")``.

    Example::

        @task
        def preprocess(text):
            return text.strip().lower()
    """

    @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
    def sync_wrapper(
        wrapped: Callable[ParametersType, ReturnType],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> ReturnType:
        span_name = name or _span_name(instance, wrapped)
        with trace_api.get_tracer(__name__).start_as_current_span(span_name) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, _TASK_OPERATION)
            return wrapped(*args, **kwargs)

    @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
    async def async_wrapper(
        wrapped: Callable[ParametersType, Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        span_name = name or _span_name(instance, wrapped)
        with trace_api.get_tracer(__name__).start_as_current_span(span_name) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, _TASK_OPERATION)
            return await wrapped(*args, **kwargs)

    def _select(fn: Callable[ParametersType, ReturnType]) -> Callable[ParametersType, ReturnType]:
        if asyncio.iscoroutinefunction(fn):
            return cast("Callable[ParametersType, ReturnType]", async_wrapper(fn))
        return cast("Callable[ParametersType, ReturnType]", sync_wrapper(fn))

    if wrapped_function is not None:
        return _select(wrapped_function)
    return _select


def embedding(
    wrapped_function: Optional[Callable[ParametersType, ReturnType]] = None,
    /,
    *,
    model_name: str,
    name: Optional[str] = None,
    model_provider: str = _DEFAULT_MODEL_PROVIDER,
) -> Union[
    Callable[ParametersType, ReturnType],
    Callable[[Callable[ParametersType, ReturnType]], Callable[ParametersType, ReturnType]],
]:
    """Open an embedding span around the decorated function.

    Pass the ``model_name`` of the embedding model (required). ``name`` overrides the span name
    (defaults to the function name); ``model_provider`` defaults to ``"custom"``.

    Example::

        @embedding(model_name="text-embedding-3-small", model_provider="openai")
        def embed(text):
            return client.embeddings.create(model="text-embedding-3-small", input=text)
    """

    @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
    def sync_wrapper(
        wrapped: Callable[ParametersType, ReturnType],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> ReturnType:
        span_name = name or _span_name(instance, wrapped)
        with trace_api.get_tracer(__name__).start_as_current_span(span_name) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, _EMBEDDINGS_OPERATION)
            span.set_attribute(GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(GEN_AI_PROVIDER_NAME, model_provider)
            return wrapped(*args, **kwargs)

    @wrapt.decorator  # type: ignore[misc,attr-defined,unused-ignore]
    async def async_wrapper(
        wrapped: Callable[ParametersType, Any],
        instance: Any,
        args: Any,
        kwargs: Any,
    ) -> Any:
        span_name = name or _span_name(instance, wrapped)
        with trace_api.get_tracer(__name__).start_as_current_span(span_name) as span:
            span.set_attribute(GEN_AI_OPERATION_NAME, _EMBEDDINGS_OPERATION)
            span.set_attribute(GEN_AI_REQUEST_MODEL, model_name)
            span.set_attribute(GEN_AI_PROVIDER_NAME, model_provider)
            return await wrapped(*args, **kwargs)

    def _select(fn: Callable[ParametersType, ReturnType]) -> Callable[ParametersType, ReturnType]:
        if asyncio.iscoroutinefunction(fn):
            return cast("Callable[ParametersType, ReturnType]", async_wrapper(fn))
        return cast("Callable[ParametersType, ReturnType]", sync_wrapper(fn))

    if wrapped_function is not None:
        return _select(wrapped_function)
    return _select


def _span_name(instance: Any, wrapped: Callable[..., Any]) -> str:
    """Build the span name: ``ClassName.method`` for methods, else the function name."""
    if inspect.ismethod(wrapped):
        owner = instance if isinstance(instance, type) else type(instance)
        return f"{owner.__name__}.{wrapped.__name__}"
    if instance is not None:
        return f"{type(instance).__name__}.{wrapped.__name__}"
    return wrapped.__name__
