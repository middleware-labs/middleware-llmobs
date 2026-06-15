import asyncio
import json
from typing import Any, Dict, Generator, List

import pytest
from openinference.instrumentation import TracerProvider as OITracerProvider
from openinference.instrumentation import using_session
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from middleware.llmobs import annotate_rag, embedding, retriever, task

# We use the OpenInference TracerProvider (which yields an OITracer) so that span-kind handling
# and context attributes behave exactly as they do for real SDK users.


# OTel only allows the global TracerProvider to be set once per process and ignores later calls
# (with a warning). Other test modules may have claimed the global first, so we force our provider
# into place by clearing OTel's "already set" guard before setting it. A single provider serves the
# whole module; each test gets a fresh exporter via ``clear()``.
_EXPORTER = InMemorySpanExporter()
_PROVIDER = OITracerProvider()
_PROVIDER.add_span_processor(SimpleSpanProcessor(_EXPORTER))


@pytest.fixture
def exporter() -> Generator[InMemorySpanExporter, None, None]:
    trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    trace_api.set_tracer_provider(_PROVIDER)
    _EXPORTER.clear()
    yield _EXPORTER


def _attrs(exporter: InMemorySpanExporter, name: str) -> Dict[str, Any]:
    spans = {s.name: s for s in exporter.get_finished_spans()}
    return dict(spans[name].attributes or {})


# --- decorator: opens a retrieval span, captures nothing on its own -----


def test_retriever_opens_retriever_span_without_auto_capture(
    exporter: InMemorySpanExporter,
) -> None:
    @retriever
    def get_relevant_docs(question: str) -> List[Dict[str, Any]]:
        return [{"id": "doc_123", "score": 0.95, "text": "Paris."}]

    get_relevant_docs("What is the capital of France?")

    a = _attrs(exporter, "get_relevant_docs")
    assert a["gen_ai.operation.name"] == "retrieval"
    # Nothing is auto-captured: no query inferred from args, no documents from the return value.
    assert "gen_ai.retrieval.query.text" not in a
    assert "gen_ai.retrieval.documents" not in a


def test_retriever_with_annotate_rag_inside(exporter: InMemorySpanExporter) -> None:
    @retriever
    def get_relevant_docs(question: str) -> List[Dict[str, Any]]:
        docs = [{"id": "doc_123", "score": 0.95, "text": "Paris."}]
        annotate_rag(query=question, documents=docs)
        return docs

    get_relevant_docs("What is the capital of France?")

    a = _attrs(exporter, "get_relevant_docs")
    assert a["gen_ai.operation.name"] == "retrieval"
    assert a["gen_ai.retrieval.query.text"] == "What is the capital of France?"
    assert json.loads(a["gen_ai.retrieval.documents"]) == [
        {"id": "doc_123", "score": 0.95, "text": "Paris."}
    ]


def test_retriever_custom_name(exporter: InMemorySpanExporter) -> None:
    @retriever(name="vector_search")
    def search(q: str) -> List[Dict[str, Any]]:
        return [{"id": "x"}]

    search("q")
    assert "vector_search" in {s.name for s in exporter.get_finished_spans()}


# --- decorator: async ---------------------------------------------------


def test_retriever_async(exporter: InMemorySpanExporter) -> None:
    @retriever
    async def aget(question: str) -> List[Dict[str, Any]]:
        docs = [{"id": "a1", "score": 0.5}]
        annotate_rag(query=question, documents=docs)
        return docs

    asyncio.run(aget("async query"))

    a = _attrs(exporter, "aget")
    assert a["gen_ai.operation.name"] == "retrieval"
    assert a["gen_ai.retrieval.query.text"] == "async query"
    assert json.loads(a["gen_ai.retrieval.documents"]) == [{"id": "a1", "score": 0.5}]


# --- context attributes preserved --------------------------------------


def test_retriever_preserves_context_attributes(exporter: InMemorySpanExporter) -> None:
    @retriever
    def search(q: str) -> List[Dict[str, Any]]:
        return [{"id": "x"}]

    with using_session("sess-42"):
        search("q")

    assert _attrs(exporter, "search")["session.id"] == "sess-42"


# --- explicit annotate_rag (no auto-capture) ---------------------------


def test_annotate_rag_sets_query_and_documents(exporter: InMemorySpanExporter) -> None:
    tracer = trace_api.get_tracer(__name__)
    with tracer.start_as_current_span("manual"):
        annotate_rag(
            query="manual query",
            documents=[{"id": "m1", "score": 0.1, "text": "x"}],
        )

    a = _attrs(exporter, "manual")
    assert a["gen_ai.operation.name"] == "retrieval"
    assert a["gen_ai.retrieval.query.text"] == "manual query"
    assert json.loads(a["gen_ai.retrieval.documents"]) == [{"id": "m1", "score": 0.1, "text": "x"}]


def test_annotate_rag_normalizes_objects(exporter: InMemorySpanExporter) -> None:
    class Doc:
        def __init__(self, id: str, score: float, text: str) -> None:
            self.id, self.score, self.text = id, score, text

    tracer = trace_api.get_tracer(__name__)
    with tracer.start_as_current_span("obj"):
        annotate_rag(query="q", documents=[Doc("d1", 0.5, "hello")])

    docs = json.loads(_attrs(exporter, "obj")["gen_ai.retrieval.documents"])
    assert docs == [{"id": "d1", "score": 0.5, "text": "hello"}]


def test_annotate_rag_query_only(exporter: InMemorySpanExporter) -> None:
    tracer = trace_api.get_tracer(__name__)
    with tracer.start_as_current_span("qonly"):
        annotate_rag(query="just the query")

    a = _attrs(exporter, "qonly")
    assert a["gen_ai.retrieval.query.text"] == "just the query"
    assert "gen_ai.retrieval.documents" not in a


# --- task ---------------------------------------------------------------


def test_task_sync(exporter: InMemorySpanExporter) -> None:
    @task
    def preprocess(text: str) -> str:
        return text.strip()

    assert preprocess("  hi  ") == "hi"
    assert _attrs(exporter, "preprocess")["gen_ai.operation.name"] == "task"


def test_task_async_with_name(exporter: InMemorySpanExporter) -> None:
    @task(name="step")
    async def run() -> int:
        return 1

    assert asyncio.run(run()) == 1
    assert _attrs(exporter, "step")["gen_ai.operation.name"] == "task"


# --- embedding ----------------------------------------------------------


def test_embedding_sets_model_and_provider(exporter: InMemorySpanExporter) -> None:
    @embedding(model_name="text-embedding-3-small", model_provider="openai")
    def embed(text: str) -> List[float]:
        return [0.1, 0.2]

    embed("hello")

    a = _attrs(exporter, "embed")
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["gen_ai.request.model"] == "text-embedding-3-small"
    assert a["gen_ai.provider.name"] == "openai"


def test_embedding_default_provider(exporter: InMemorySpanExporter) -> None:
    @embedding(model_name="my-model")
    def embed(text: str) -> List[float]:
        return [0.0]

    embed("x")

    a = _attrs(exporter, "embed")
    assert a["gen_ai.request.model"] == "my-model"
    assert a["gen_ai.provider.name"] == "custom"


def test_embedding_async(exporter: InMemorySpanExporter) -> None:
    @embedding(model_name="m")
    async def aembed(text: str) -> List[float]:
        return [1.0]

    asyncio.run(aembed("x"))

    a = _attrs(exporter, "aembed")
    assert a["gen_ai.operation.name"] == "embeddings"
    assert a["gen_ai.request.model"] == "m"
