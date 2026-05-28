import json
import sys
from typing import Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry._logs import set_logger_provider
from opentelemetry.metrics import set_meter_provider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

from middleware.llmobs.eval import (
    BooleanStructuredOutput,
    CategoricalStructuredOutput,
    EvalClient,
    EvaluatorContext,
    EvaluatorResult,
    LLMJudge,
    ScoreStructuredOutput,
    evaluator,
    export_current_span,
    format_schema_for_provider,
    submit_evaluation_error,
)
from middleware.llmobs.eval.types import infer_metric_type


@pytest.fixture
def ctx() -> EvaluatorContext:
    return EvaluatorContext(input="q", output="a")


# --- infer_metric_type --------------------------------------------------


def test_infer_bool_before_int() -> None:
    assert infer_metric_type(True) == "boolean"
    assert infer_metric_type(1) == "score"
    assert infer_metric_type(0.5) == "score"
    assert infer_metric_type("x") == "categorical"


def test_infer_dict_raises() -> None:
    with pytest.raises(TypeError):
        infer_metric_type({"a": 1})


# --- @evaluator decorator -----------------------------------------------


def test_decorator_return_types(ctx: EvaluatorContext) -> None:
    @evaluator
    def b(c: EvaluatorContext) -> bool:
        return True

    @evaluator
    def s(c: EvaluatorContext) -> float:
        return 0.5

    @evaluator
    def cat(c: EvaluatorContext) -> str:
        return "good"

    assert b(ctx).metric_type == "boolean"
    assert s(ctx).metric_type == "score"
    assert cat(ctx).metric_type == "categorical"
    assert b._is_evaluator is True  # type: ignore[attr-defined]
    assert b._evaluator_name == "b"  # type: ignore[attr-defined]


def test_decorator_none_skips(ctx: EvaluatorContext) -> None:
    @evaluator
    def skip(c: EvaluatorContext) -> None:
        return None

    assert skip(ctx) is None


def test_decorator_captures_error(ctx: EvaluatorContext) -> None:
    @evaluator
    def boom(c: EvaluatorContext) -> bool:
        raise RuntimeError("nope")

    r = boom(ctx)
    assert r is not None
    assert r.error == "nope"
    assert r.error_type == "RuntimeError"
    assert r.value is None


def test_decorator_bad_return_type_raises(ctx: EvaluatorContext) -> None:
    @evaluator
    def bad(c: EvaluatorContext):  # type: ignore[no-untyped-def]
        return object()

    with pytest.raises(TypeError):
        bad(ctx)


def test_decorator_dict_no_kwarg_collision(ctx: EvaluatorContext) -> None:
    @evaluator
    def d(c: EvaluatorContext) -> dict:
        return {"value": 0.9, "metric_type": "score", "evaluator_name": "mine"}

    assert d(ctx).evaluator_name == "mine"  # user value wins, no collision


def test_decorator_does_not_mutate_returned_result(ctx: EvaluatorContext) -> None:
    held = EvaluatorResult(value=1.0)

    @evaluator
    def passthru(c: EvaluatorContext) -> EvaluatorResult:
        return held

    out = passthru(ctx)
    assert held.metric_type is None  # original untouched
    assert out.metric_type == "score"  # copy back-inferred
    assert out is not held


# --- LLMJudge -----------------------------------------------------------


def test_llmjudge_requires_client() -> None:
    with pytest.raises(ValueError):
        LLMJudge(client=None, model="m", user_prompt="x")  # type: ignore[arg-type]


def test_llmjudge_score_with_assessment(ctx: EvaluatorContext) -> None:
    def fake(messages, model, json_schema=None, model_params=None):  # type: ignore[no-untyped-def]
        return json.dumps({"score_eval": 0.8, "reasoning": "ok"})

    judge = LLMJudge(
        client=fake,
        model="m",
        user_prompt="{{output}}",
        structured_output=ScoreStructuredOutput(
            description="d", min_score=0, max_score=1, reasoning=True, min_threshold=0.7
        ),
    )
    r = judge(ctx)
    assert r is not None
    assert r.value == 0.8
    assert r.metric_type == "score"
    assert r.assessment == "pass"
    assert r.reasoning == "ok"


def test_llmjudge_bad_json_captured(ctx: EvaluatorContext) -> None:
    def fake(messages, model, json_schema=None, model_params=None):  # type: ignore[no-untyped-def]
        return "not json"

    judge = LLMJudge(
        client=fake,
        model="m",
        user_prompt="{{output}}",
        structured_output=BooleanStructuredOutput(description="d"),
    )
    r = judge(ctx)
    assert r is not None
    assert r.error is not None  # captured, not raised


def test_llmjudge_renders_template() -> None:
    captured: dict = {}

    def fake(messages, model, json_schema=None, model_params=None):  # type: ignore[no-untyped-def]
        captured["prompt"] = messages[-1]["content"]
        return json.dumps({"categorical_eval": "positive"})

    judge = LLMJudge(
        client=fake,
        model="m",
        user_prompt="answer: {{output}} meta: {{metadata.k}}",
        structured_output=CategoricalStructuredOutput(
            categories={"positive": "p", "negative": "n"}, pass_values=["positive"]
        ),
    )
    c = EvaluatorContext(input="q", output="hello", metadata={"k": "v"})
    r = judge(c)
    assert captured["prompt"] == "answer: hello meta: v"
    assert r is not None
    assert r.value == "positive"
    assert r.assessment == "pass"


# --- score exclusive-range assessment edge case -------------------------


def test_score_exclusive_range() -> None:
    out = ScoreStructuredOutput(
        description="d", min_score=0, max_score=10, min_threshold=8, max_threshold=2
    )
    from middleware.llmobs.eval.structured_output import compute_assessment

    assert compute_assessment(out, 1) == "pass"  # below max_threshold
    assert compute_assessment(out, 9) == "pass"  # above min_threshold
    assert compute_assessment(out, 5) == "fail"  # inside the excluded band


# --- emission + auto-binding (in-memory exporters) ----------------------


# OTel global providers can only be set once per process, so configure them once at module scope
# and clear the in-memory exporter between tests.
_LOG_EXPORTER = InMemoryLogRecordExporter()
_METRIC_READER = InMemoryMetricReader()
_PROVIDERS_SET = False


@pytest.fixture
def emit_env() -> Generator[tuple[InMemoryLogRecordExporter, InMemoryMetricReader], None, None]:
    global _PROVIDERS_SET
    if not _PROVIDERS_SET:
        trace_api.set_tracer_provider(TracerProvider())
        lp = LoggerProvider()
        lp.add_log_record_processor(SimpleLogRecordProcessor(_LOG_EXPORTER))
        set_logger_provider(lp)
        set_meter_provider(MeterProvider(metric_readers=[_METRIC_READER]))
        _PROVIDERS_SET = True
    _LOG_EXPORTER.clear()
    yield _LOG_EXPORTER, _METRIC_READER


def test_submit_validation_errors() -> None:
    client = EvalClient()
    with pytest.raises(ValueError):  # contradictory targets
        client.submit_evaluation(
            label="x", value=1.0, span_id="a", trace_id="b", join_on_tag=("k", "v")
        )
    with pytest.raises(ValueError):  # incomplete direct ref
        client.submit_evaluation(label="x", value=1.0, span_id="a")
    with pytest.raises(ValueError):  # bad label
        client.submit_evaluation(label="bad.label", value=1.0)
    with pytest.raises(TypeError):  # type mismatch
        client.submit_evaluation(label="x", value="str", metric_type="score")


def test_emit_autobinds_to_active_span(
    emit_env: tuple[InMemoryLogRecordExporter, InMemoryMetricReader],
) -> None:
    log_exp, reader = emit_env
    tracer = trace_api.get_tracer("t")
    client = EvalClient()
    with tracer.start_as_current_span("chat"):
        ids = export_current_span()
        client.submit_evaluation(label="toxicity", value=0.1, metric_type="score")

    logs = log_exp.get_finished_logs()
    assert len(logs) == 1
    assert ids is not None
    assert logs[0].log_record.trace_id == int(ids["trace_id"], 16)
    assert logs[0].log_record.span_id == int(ids["span_id"], 16)

    names = {
        m.name
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    assert names == {
        "gen_ai.evaluations.count",
        "gen_ai.evaluations.score",
        "gen_ai.evaluations.outcome",
    }


def test_emit_unattached_when_no_span(
    emit_env: tuple[InMemoryLogRecordExporter, InMemoryMetricReader],
) -> None:
    log_exp, _ = emit_env
    EvalClient().submit_evaluation(label="offline", value=True, metric_type="boolean")
    logs = log_exp.get_finished_logs()
    assert len(logs) == 1
    assert logs[0].log_record.trace_id == 0
    assert logs[0].log_record.span_id == 0


def test_export_current_span_none_outside() -> None:
    trace_api.set_tracer_provider(TracerProvider())
    assert export_current_span() is None


# --- error submission ---------------------------------------------------


def test_submit_evaluation_error_emits_error_log(
    emit_env: tuple[InMemoryLogRecordExporter, InMemoryMetricReader],
) -> None:
    log_exp, _ = emit_env
    EvalClient().submit_evaluation_error(
        label="faithfulness", error="judge timed out", error_type="TimeoutError"
    )
    logs = log_exp.get_finished_logs()
    assert len(logs) == 1
    rec = logs[0].log_record
    attrs = dict(rec.attributes or {})
    assert attrs["gen_ai.evaluation.outcome"] == "error"
    assert attrs["error.type"] == "TimeoutError"
    assert attrs["exception.message"] == "judge timed out"
    assert rec.severity_text == "WARN"
    body = json.loads(rec.body)
    assert body["outcome"] == "error"
    assert "value" not in body  # no score on an error


def test_evaluate_and_submit_emits_error_for_failed_evaluator(
    emit_env: tuple[InMemoryLogRecordExporter, InMemoryMetricReader],
) -> None:
    log_exp, _ = emit_env

    @evaluator
    def boom(c: EvaluatorContext) -> bool:
        raise RuntimeError("kaboom")

    ctx = EvaluatorContext(input="q", output="a")
    result = EvalClient().evaluate_and_submit(boom, ctx)

    # The errored result is returned AND surfaced as a failed eval (not dropped).
    assert result is not None and result.error == "kaboom"
    logs = log_exp.get_finished_logs()
    assert len(logs) == 1
    attrs = dict(logs[0].log_record.attributes or {})
    assert attrs["gen_ai.evaluation.outcome"] == "error"
    assert attrs["error.type"] == "RuntimeError"


def test_module_level_submit_evaluation_error(
    emit_env: tuple[InMemoryLogRecordExporter, InMemoryMetricReader],
) -> None:
    log_exp, _ = emit_env
    submit_evaluation_error(label="x", error="boom")
    assert len(log_exp.get_finished_logs()) == 1


def test_submit_evaluation_with_judge_metadata_and_cost(
    emit_env: tuple[InMemoryLogRecordExporter, InMemoryMetricReader],
) -> None:
    log_exp, reader = emit_env
    EvalClient().submit_evaluation(
        label="faithfulness",
        value=0.8,
        metric_type="score",
        assessment="pass",
        reasoning="grounded",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        cost_usd=0.0042,
    )
    rec = log_exp.get_finished_logs()[-1].log_record
    attrs = dict(rec.attributes or {})
    assert attrs["eval.model.provider"] == "openai"
    assert attrs["eval.model.name"] == "gpt-4o-mini"
    assert attrs["gen_ai.evaluation.cost.usd"] == 0.0042
    body = json.loads(rec.body)
    assert body["verdict"] == "pass"
    assert body["explanation"] == "grounded"
    assert body["judge_provider"] == "openai"
    assert body["cost_usd"] == 0.0042

    names = {
        m.name
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    assert "gen_ai.evaluations.cost.usd" in names


# --- format_schema_for_provider -----------------------------------------


def test_format_openai_wraps_response_format() -> None:
    schema = ScoreStructuredOutput(description="d", min_score=0, max_score=1).to_json_schema()
    out = format_schema_for_provider(schema, "openai")
    rf = out["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == schema  # openai keeps minimum/maximum


def test_format_anthropic_strips_number_bounds() -> None:
    schema = ScoreStructuredOutput(description="d", min_score=0, max_score=10).to_json_schema()
    out = format_schema_for_provider(schema, "anthropic")
    assert out["extra_headers"]["anthropic-beta"]
    prop = out["extra_body"]["output_format"]["schema"]["properties"]["score_eval"]
    assert "minimum" not in prop and "maximum" not in prop
    assert "range: 0 to 10" in prop["description"]
    # original is untouched (deep-copied)
    assert "minimum" in schema["properties"]["score_eval"]


def test_format_vertexai_const_to_enum() -> None:
    schema = CategoricalStructuredOutput(categories={"a": "A", "b": "B"}).to_json_schema()
    out = format_schema_for_provider(schema, "vertexai")
    prop = out["generation_config"]["response_schema"]["properties"]["categorical_eval"]
    assert prop["enum"] == ["a", "b"]
    assert "anyOf" not in prop


def test_format_bedrock_stringifies_schema() -> None:
    schema = BooleanStructuredOutput(description="d").to_json_schema()
    out = format_schema_for_provider(schema, "bedrock")
    js = out["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
    assert isinstance(js["schema"], str)
    assert json.loads(js["schema"])["properties"]["boolean_eval"]["type"] == "boolean"


def test_format_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        format_schema_for_provider({}, "cohere")  # type: ignore[arg-type]


# --- no LLM provider import ---------------------------------------------


def test_no_llm_provider_imported() -> None:
    import middleware.llmobs.eval  # noqa: F401

    banned = ["openai", "anthropic", "boto3", "vertexai", "cohere", "mistralai"]
    assert [b for b in banned if b in sys.modules] == []
