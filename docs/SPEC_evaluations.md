# Spec: SDK Evaluation Surfaces for `middleware-llmobs`

**Status:** Draft · **Target package:** `middleware-llmobs` (`src/middleware/llmobs/`) · **Date:** 2026-05-26

This is the Middleware-native adaptation of the generic "SDK Evaluation Surfaces" spec. It
reconciles that spec with how this SDK actually works: there is **no REST client and no
`/api/v1/evaluations` endpoint**. The SDK is an OpenTelemetry/OpenInference tracing layer
(`register()` → `TracerProvider`, HTTP/protobuf only). Evaluations are therefore shipped the same
way the platform already ingests online evals — as **OTLP signals over HTTP**, emitted through the
**OpenTelemetry logger and meter providers**.

---

## 0. Scope

In scope for this spec:

1. **Shared types** — `EvaluatorContext`, `EvaluatorResult`, `MetricType`, `Assessment`,
   `ScoreValue`, `LLMClient` (Protocol).
2. **Surface 1 — `submit_evaluation`** — push an externally-computed score, attach it to a
   span/trace, emit it as an OTel log record + metrics.
3. **Surface 2 — Local evaluators** — `@evaluator` decorator, `BaseEvaluator`, and
   `evaluate_and_submit` (call an evaluator and submit its result in one step).
4. **Surface 3 — Structured outputs + `LLMJudge`** — `BaseStructuredOutput` and its
   `Boolean`/`Score`/`Categorical` subclasses (JSON-schema generation + pass/fail assessment), and an
   `LLMJudge(BaseEvaluator)` that drives a **user-supplied** `LLMClient`. **No provider libraries are
   imported** — `LLMJudge` only accepts a `client`; it has no `_create_*_client` factories.

**Deferred (not in this spec):** the `Experiment` / `DatasetRecord` / `ExperimentReport` runner.
Online + local span/trace evaluations land first; the experiment runner is a follow-up because it
needs a different ingestion surface (experiment records, not spans) that does not exist yet.

Hard constraints carried over unchanged from the source spec:

- The SDK **MUST NOT** import any LLM provider library (`openai`, `anthropic`, `boto3`,
  `google-cloud-aiplatform`, `cohere`, `mistralai`, …) — not at module load, not lazily, not in a
  factory. Users bring their own clients via the `LLMClient` Protocol.
  - **Deviation from the ddtrace `LLMJudge` reference:** that reference includes
    `_create_openai_client` / `_create_anthropic_client` / `_create_bedrock_client` /
    `_create_vertexai_client` / `_create_azure_openai_client` factories that lazily import provider
    libraries. **We drop all of these.** Middleware's `LLMJudge` accepts only a user-supplied
    `client`; there is no `provider=` shortcut and no factory that imports a provider SDK. This is
    the single biggest divergence from the pasted source and is intentional.
- The SDK **MUST NOT** ship built-in LLM-judge implementations (no `FaithfulnessMetric` etc.). A
  generic, provider-agnostic `LLMJudge` driver that the *user* configures (prompt + structured
  output + their own client) is allowed and specified in §6 — it is not a pre-baked metric and
  contains no LLM logic of its own.
- Pure-function evaluators and `submit_evaluation` **MUST** work with zero LLM deps installed.
- User evaluator errors **MUST** be captured into `EvaluatorResult.error`, never silently dropped.
- `submit_evaluation` **MUST NOT** block the request path (queue + background flush).

Engineering conventions (from `pyproject.toml`): Python ≥ 3.10, `mypy --strict`,
`ruff` (line length 100, isort), no new hard third-party deps beyond what OTel already pulls in.

---

## 1. Why OTLP and not a REST endpoint

The generic spec's `POST /api/v1/evaluations` does not exist in the Middleware ingest path. The
server-side reference (`eval_service`) proves how online evals are written: as **OTLP JSON** to
`{base}/v1/logs` and `{base}/v1/metrics` with `gen_ai.evaluation.*` attributes. We mirror that
contract so SDK-submitted evals are indistinguishable from online evals in dashboards.

Rather than hand-rolling httpx + gzip like the server reference, the SDK reuses the **OpenTelemetry
logger and meter providers** already shipped in our dependency tree. This gives us batching,
retries, env-var endpoint/header resolution, and `Authorization` handling for free, and keeps the
transport identical to the trace path (`register()` already wires `OTEL_EXPORTER_OTLP_ENDPOINT` /
`OTEL_EXPORTER_OTLP_HEADERS`).

Confirmed available in the venv (`opentelemetry-exporter-otlp`, `opentelemetry-sdk`):

| Concern | API |
|---|---|
| Logs SDK | `opentelemetry.sdk._logs.LoggerProvider`, `Logger.emit(LogRecord)` |
| Log export | `opentelemetry.sdk._logs.export.BatchLogRecordProcessor` + `opentelemetry.exporter.otlp.proto.http._log_exporter.OTLPLogExporter` |
| Log record | `opentelemetry.sdk._logs._internal.LogRecord` (fields: `timestamp`, `observed_timestamp`, `trace_id`, `span_id`, `severity_text`, `severity_number`, `body`, `attributes`, …) |
| Severity | `opentelemetry._logs.SeverityNumber.INFO` (== 9), matching the server's `severity_number: 9` |
| Metrics SDK | `opentelemetry.sdk.metrics.MeterProvider` + `PeriodicExportingMetricReader` |
| Metric export | `opentelemetry.exporter.otlp.proto.http.metric_exporter.OTLPMetricExporter` |
| Instruments | `Meter.create_gauge`, `Meter.create_counter`, `Meter.create_histogram` (all present) |

Both HTTP exporters accept the same `endpoint` + `headers` kwargs as the existing
`HTTPSpanExporter`, so endpoint normalization is the only new wiring.

---

## 2. Module layout

```
src/middleware/llmobs/
  eval/
    __init__.py          # re-exports the public eval surface
    types.py             # EvaluatorContext, EvaluatorResult, MetricType, Assessment, ScoreValue, LLMClient
    decorator.py         # @evaluator, _normalize_result (shared wrapping logic)
    base.py              # BaseEvaluator (reuses _normalize_result)
    structured_output.py # BaseStructuredOutput + Boolean/Score/Categorical, StructuredOutput union
    judge.py             # LLMJudge(BaseEvaluator) — BYO LLMClient only, no provider imports
    client.py            # EvalClient + module-level submit_evaluation / evaluate_and_submit / flush
    _emit.py             # OTLP logger/meter wiring + payload builders (gen_ai.evaluation.* attrs)
```

`structured_output.py` and `judge.py` import **only** stdlib + `eval/types.py`. A test asserts no
provider library lands in `sys.modules` after importing the whole subpackage (§8).

`eval/` is a new subpackage; it does not touch `otel.py`/`settings.py` except to reuse endpoint and
header helpers (`get_env_collector_endpoint`, `get_env_client_headers`, `_normalized_endpoint`).

### Public API (added to `middleware.llmobs.__init__`)

```python
from middleware.llmobs import (
    # types
    EvaluatorContext,
    EvaluatorResult,
    LLMClient,            # Protocol
    MetricType,           # Literal["score","boolean","categorical"]
    Assessment,           # Literal["pass","fail"]
    ScoreValue,           # Union[float,int,bool,str,dict]
    # evaluator authoring
    evaluator,            # decorator
    BaseEvaluator,        # ABC
    # LLM-judge authoring (provider-agnostic, BYO client)
    LLMJudge,             # BaseEvaluator subclass driving a user LLMClient
    BaseStructuredOutput,
    BooleanStructuredOutput,
    ScoreStructuredOutput,
    CategoricalStructuredOutput,
    StructuredOutput,     # Union[Boolean|Score|Categorical|dict]
    # submission
    submit_evaluation,    # module-level convenience
    evaluate_and_submit,  # module-level convenience
    flush_evaluations,    # module-level convenience
    export_current_span,  # {"span_id","trace_id"} from active OTel span, or None
    EvalClient,           # explicit client for advanced/multi-target use
)
```

We expose **both** module-level functions (idiomatic for a `register()`-style SDK with no existing
client object) **and** an explicit `EvalClient` class (for users who need a non-default endpoint or
want to manage lifecycle in tests). The module-level functions delegate to a lazily-created default
`EvalClient` built from the same OTLP env vars `register()` uses.

`DatasetRecord`, `Experiment`, `experiment(...)` are intentionally **omitted** (deferred).

---

## 3. Shared types (`eval/types.py`)

```python
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol, Union

MetricType = Literal["score", "boolean", "categorical"]
Assessment = Literal["pass", "fail"]
ScoreValue = Union[float, int, bool, str, dict]


@dataclass
class EvaluatorContext:
    input: Any
    output: Any
    expected_output: Optional[Any] = None
    retrieved_contexts: Optional[list[str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluatorResult:
    value: ScoreValue
    metric_type: Optional[MetricType] = None
    assessment: Optional[Assessment] = None
    reasoning: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    evaluator_name: Optional[str] = None


class LLMClient(Protocol):
    def __call__(
        self,
        messages: list[dict[str, str]],
        model: str,
        json_schema: Optional[dict[str, Any]] = None,
        model_params: Optional[dict[str, Any]] = None,
    ) -> str: ...
```

`LLMClient` is a `Protocol`, never an ABC — this is the contract that keeps the SDK free of provider
deps. Any callable matching the signature satisfies it.

> **Signature choice (deviation from ddtrace reference):** the ddtrace `LLMJudge` reference uses a
> 5-arg form with a leading `provider` (`__call__(provider, messages, json_schema, model,
> model_params)`). We use the **4-arg form above** (no `provider`). Rationale: because we dropped the
> provider factories, the SDK never branches on `provider` internally, so passing it to the user's
> adapter is dead weight. `LLMJudge` therefore calls `self._client(messages, model,
> json_schema=..., model_params=...)` — keyword args for the optionals so a user adapter written
> against the documented public signature is call-compatible.

`JSONType` from the reference is **not** introduced as a separate alias; a custom schema is just a
`dict[str, Any]`, and the `StructuredOutput` union (§6.1) uses `dict[str, Any]` for the custom case.

### `metric_type` inference (single source of truth)

A helper used by both `submit_evaluation` and the evaluator wrapper:

```python
def infer_metric_type(value: ScoreValue) -> MetricType:
    if isinstance(value, bool):   return "boolean"   # bool BEFORE int — bool is a subclass of int
    if isinstance(value, (int, float)): return "score"
    if isinstance(value, str):    return "categorical"
    raise TypeError("metric_type cannot be inferred from dict; set it explicitly")
```

`bool` **must** be checked before `int`. This applies everywhere a metric type is inferred.

---

## 4. Surface 1 — `submit_evaluation`

### 4.1 Signature

Method on `EvalClient`, mirrored by a module-level function with identical kwargs:

```python
def submit_evaluation(
    self,
    *,
    label: str,
    value: ScoreValue,
    metric_type: Optional[MetricType] = None,
    span_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    join_on_tag: Optional[tuple[str, str]] = None,
    eval_scope: Literal["span", "trace"] = "span",
    assessment: Optional[Assessment] = None,
    reasoning: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    tags: Optional[dict[str, str]] = None,
    timestamp_ms: Optional[int] = None,
    ml_app: Optional[str] = None,
) -> None: ...
```

`ml_app` defaults to the configured project name (`MW_PROJECT_NAME`, falling back to
`OTEL_SERVICE_NAME`), matching `register()`'s identity resolution.

`eval_scope` (ported from ddtrace's `submit_evaluation`) controls whether the eval is associated with
a single span (`"span"`, default) or the whole trace (`"trace"`). For `"trace"` scope the
`span_id` is dropped from the join and only the `trace_id` is used (the provided/auto-bound span
should be the root span); it is carried in the emitted payload as `eval_scope`.

### 4.1a Span/trace target resolution (auto-binding)

**The span and trace IDs may or may not be present** — `submit_evaluation` resolves the target in
this order:

1. **Explicit `join_on_tag`** → correlate by tag; `span_id`/`trace_id` are ignored (and must not also
   be passed — see validation). The backend matches the eval to whichever span carries that tag.
2. **Explicit `span_id` + `trace_id`** → use them directly.
3. **Neither given → auto-bind to the current OTel span.** Read
   `opentelemetry.trace.get_current_span().get_span_context()`. If `ctx.is_valid` is true, use
   `format_trace_id(ctx.trace_id)` / `format_span_id(ctx.span_id)` as the hex IDs. This is the
   common path: an eval computed inside a traced block needs **zero** ID arguments.
4. **Neither given and no valid active span → emit unattached.** The log record carries no
   `trace_id`/`span_id`; the eval is still emitted as a log + metric (it simply isn't correlated to
   any span). **No exception, no skip** — nothing is lost; the platform handles unattached evals.

This is the key divergence from ddtrace, whose `submit_evaluation` *requires* an explicit
`span`/`span_with_tag_value` and raises if neither is given. Because this SDK owns the OTel context
directly (it *is* the tracing layer), it can recover the active span automatically. The OTel API
returns an invalid/zero context (`is_valid == False`) when called outside any span, giving a clean
signal for case 4.

> Helper: `export_current_span() -> Optional[dict[str, str]]` returns
> `{"span_id": ..., "trace_id": ...}` for the current span (or `None` outside one), mirroring
> ddtrace's `export_span()`. Use it to capture IDs now and submit later / off-thread, where the OTel
> context would no longer be active. Exported in the public API.

### 4.2 Validation (raise before emitting)

1. **Join target — mutual exclusion, not presence.** The join target is *optional* (see §4.1a: it
   may auto-bind or stay unattached). The SDK only rejects *contradictory* targets:
   - `join_on_tag` together with `span_id` and/or `trace_id` → `ValueError` (ambiguous).
   - `span_id` without `trace_id`, or `trace_id` without `span_id` → `ValueError` (an incomplete
     direct reference is a mistake; don't silently fall back to auto-binding).
   - **Neither given is allowed** → triggers auto-bind / unattached emission. No error.
   - `join_on_tag` must be a 2-tuple of non-empty strings → else `ValueError`.
2. **`eval_scope`:** one of `"span"` / `"trace"` → else `ValueError`.
3. **Label:** matches `^[a-zA-Z][a-zA-Z0-9_]*$`, length ≤ 200 → else `ValueError`. (Per the
   platform contract a `.` in the label is also rejected; the regex already forbids it.)
4. **Type consistency** (`bool` must be checked **before** `int`/`float`, since `bool` is a subclass
   of `int` — `isinstance(True, int)` is `True`):
   - `metric_type == "score"` → `value` is `int`/`float` **and not** `bool`. Concretely:
     `isinstance(value, (int, float)) and not isinstance(value, bool)`. A bare `isinstance(value,
     (int, float))` would wrongly accept `True`/`False` as a score.
   - `metric_type == "boolean"` → `value` is `bool` (`isinstance(value, bool)`).
   - `metric_type == "categorical"` → `value` is `str`.
   - If `metric_type` is omitted → infer via `infer_metric_type`, which applies the same
     bool-before-int ordering (§3).
5. **JSON-serializable:** `metadata` and `tags` must be JSON-serializable (validated with a trial
   `json.dumps`) → else `ValueError`.

Note: the `ScoreValue` union still allows `dict`, but a `dict` value has no inferable `metric_type`
(there is intentionally **no** `"json"` metric type). A `dict` value with `metric_type` omitted
therefore raises `TypeError` from `infer_metric_type`; the caller must reshape it into a
`score`/`boolean`/`categorical` value before submitting.

### 4.3 Behavior — non-blocking

`submit_evaluation` enqueues the evaluation and returns immediately. A background worker (or the OTel
batch processors themselves) performs the actual export. Concretely:

- Build the OTel **log record** and **metric data points** (§4.4) on the calling thread (cheap).
- Hand the log record to the `Logger` (via `BatchLogRecordProcessor`) and record the metric
  instruments — both buffer and export off-thread.

`flush()` (method) / `flush_evaluations()` (module-level) force-flushes both the logger and meter
providers; required for tests and clean shutdown.

**`flush()` return contract:** `flush(timeout_millis: int = 30_000) -> bool`. It calls
`force_flush(timeout_millis)` on **both** the `LoggerProvider` and the `MeterProvider` and returns
`True` only if **both** report success within the timeout, `False` otherwise (e.g. a partial flush or
timeout). It does not raise on flush failure — callers that need hard guarantees check the boolean.
The timeout is split/applied per provider (each gets up to `timeout_millis`); document that the
worst-case wall time is up to `2 × timeout_millis`. `flush_evaluations()` mirrors the same signature
and return.

### 4.4 Wire format — OTLP via logger + meter providers

Emit **two** signals per submission, matching the server-side `gen_ai.evaluation.*` contract.

**(a) Log record** (`Logger.emit(LogRecord(...))`):

Span/trace correlation uses OTel's **native log-record fields**, not a Datadog-style `join_on`
envelope. OTel correlates a log to a trace by setting `trace_id` and `span_id` directly on the
`LogRecord` — that is the standard mechanism and what the Middleware backend already understands from
the trace path. The tag-based case is simply an attribute the backend matches on. There is **no**
`join_on` object anywhere in the payload.

- `severity_text="INFO"`, `severity_number=SeverityNumber.INFO` (9).
- `timestamp` / `observed_timestamp` = `timestamp_ms * 1e6` (ns) or now.
- **Correlation by the record's own `trace_id` / `span_id` fields:**
  - Span target resolved (explicit **or** auto-bound), `eval_scope="span"` → set both `trace_id` and
    `span_id`.
  - `eval_scope="trace"` → set `trace_id` only, leave `span_id` unset.
  - `join_on_tag` → leave both unset; the tag travels in attributes (below).
  - Unattached (no IDs, no active span) → leave both unset.
- `body` = JSON string of the structured eval object (mirrors the server-side `body_obj` shape, minus
  the Datadog join envelope):
  ```json
  {
    "eval_name": "<label>",
    "ml_app": "<ml_app>",
    "value": 0.87,
    "metric_type": "score",
    "eval_scope": "span",
    "assessment": "pass",
    "reasoning": "...",
    "metadata": {...},
    "tags": {...},
    "source": "sdk"
  }
  ```
  The trace/span correlation is carried by the OTel log-record fields above (and mirrored into the
  `gen_ai.evaluation.*` attributes below), so the body does not repeat target IDs.
- `attributes` (flat, for dashboard filtering — reuse the server-side `gen_ai.evaluation.*` keys):
  - `gen_ai.evaluation.name` = label
  - `gen_ai.evaluation.score.label` = label
  - `gen_ai.evaluation.score.value` = numeric value per the coercion rule below (omit for
    `categorical`)
  - `gen_ai.evaluation.verdict` / `gen_ai.evaluation.assessment` = assessment (if set)
  - `gen_ai.evaluation.explanation` = reasoning (if set)
  - `gen_ai.evaluation.scope` = eval_scope (`"span"` / `"trace"`)
  - `eval.target.trace_id` / `eval.target.span_id` — redundant flat mirror of the record's
    correlation fields, for dashboards that filter on attributes (set per the same scope rules)
  - `gen_ai.evaluation.tag_key` / `gen_ai.evaluation.tag_value` — the tag-correlation key/value
    (when `join_on_tag` is used)
  - `mw.llm.source` = `"sdk-eval"`  *(distinguishes SDK-submitted from online evals)*
  - flattened `tags.<k>` for each tag

**(b) Metrics** (gauge instruments on the meter, one data point per submission), mirroring the
server's `gen_ai.evaluations.*`:

| Instrument | Type | Value | Notes |
|---|---|---|---|
| `gen_ai.evaluations.count` | gauge `asInt` | `1` | common attrs |
| `gen_ai.evaluations.score` | gauge `asDouble` | coerced numeric value (below) | emit for `score`/`boolean` only; **omit** for `categorical` |
| `gen_ai.evaluations.outcome` | gauge `asInt` | `1` | + `outcome` attr = assessment or `"submitted"` |

Common metric attributes: `eval_name`, `score_label`, `verdict`, `service`, `model` (empty for
externally-computed), `provider` (empty), and `mw.llm.source="sdk-eval"`.

**Value coercion (committed rule).** The numeric value used for both
`gen_ai.evaluation.score.value` (log attribute) and the `gen_ai.evaluations.score` gauge is:

- `metric_type == "score"` → `float(value)`.
- `metric_type == "boolean"` → `1.0` if `value is True` else `0.0`. This matches the server
  reference's `score_value` handling and lets boolean evals chart on the same numeric axis as scores.
- `metric_type == "categorical"` → **no numeric value.** Do **not** emit
  `gen_ai.evaluation.score.value` and do **not** emit the `gen_ai.evaluations.score` gauge. The
  category is conveyed by `gen_ai.evaluation.score.label`/`value` *string* attributes and the
  `outcome` metric only. (Categoricals have no meaningful float; coercing them would pollute the
  score gauge with zeros.)

This is the single source of truth for coercion — §6.2 `LLMJudge` results flow through
`submit_evaluation` and inherit it; no other coercion happens elsewhere.

`metric_type` consistency across submissions sharing a `label` for an `ml_app` is enforced
**server-side** (the SDK cannot see prior submissions); the SDK only guarantees per-call type
correctness.

### 4.5 Usage

```python
from middleware.llmobs import submit_evaluation

# (1) Auto-bind: inside a traced block, no IDs needed — the eval correlates to the active span.
with tracer.start_as_current_span("chat"):
    answer = run_llm(question)
    submit_evaluation(label="toxicity", value=my_classifier(answer), metric_type="score")

# (2) Explicit span reference (e.g. submitting from outside the span's context).
submit_evaluation(
    span_id=ids["span_id"],
    trace_id=ids["trace_id"],
    label="toxicity",
    value=score,
    metric_type="score",
)

# (3) Tag-based correlation, backdated user feedback (no active span needed).
submit_evaluation(
    join_on_tag=("session_id", session.id),
    label="user_thumbs_up",
    value=True,
    metric_type="boolean",
    timestamp_ms=feedback.created_at_ms,
)

# (4) Trace-scoped eval against the whole trace (auto-binds to the current/root span's trace_id).
with tracer.start_as_current_span("agent_run"):
    submit_evaluation(label="task_success", value=True, metric_type="boolean", eval_scope="trace")

# (5) Unattached: no active span and no IDs — still emitted, just not correlated to a span.
submit_evaluation(label="offline_batch_score", value=0.42, metric_type="score")

# Capture IDs now, submit later / off-thread:
from middleware.llmobs import export_current_span
with tracer.start_as_current_span("chat"):
    ids = export_current_span()   # {"span_id": "...", "trace_id": "..."} or None outside a span
# ... later, where the OTel context is no longer active ...
submit_evaluation(label="human_review", value="approved", metric_type="categorical", **ids)
```

---

## 5. Surface 2 — Local evaluators

### 5.1 `@evaluator` decorator (`eval/decorator.py`)

Behaves exactly as the source spec: always returns `EvaluatorResult` (or `None` for skip), catches
user exceptions into `error`/`error_type`, infers `metric_type`, sets `evaluator_name`, and tags the
wrapper with `_is_evaluator = True` / `_evaluator_name`.

The normalization logic is factored into a shared `_normalize_result(raw, eval_name)` so the
decorator and `BaseEvaluator.__call__` cannot drift:

```python
import functools
from dataclasses import replace
from typing import Any, Callable, Optional

from .types import EvaluatorResult, infer_metric_type


def _normalize_result(raw: Any, eval_name: str) -> Optional[EvaluatorResult]:
    if raw is None:
        return None
    if isinstance(raw, EvaluatorResult):
        # Do NOT mutate the caller's object — copy, then fill in defaults on the copy.
        result = replace(raw)  # dataclasses.replace -> shallow copy
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
        # Don't collide if the user's dict already carries evaluator_name; their value wins,
        # and we only default it when absent. Passing evaluator_name= as a kwarg alongside
        # **raw would raise "got multiple values for keyword argument".
        fields = dict(raw)
        fields.setdefault("evaluator_name", eval_name)
        return EvaluatorResult(**fields)
    raise TypeError(
        f"Evaluator {eval_name!r} returned {type(raw).__name__}; "
        f"expected bool, int, float, str, dict, EvaluatorResult, or None"
    )


def evaluator(
    fn: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
) -> Callable[..., Any]:
    def decorate(f: Callable[..., Any]) -> Callable[..., Any]:
        eval_name = name or f.__name__

        @functools.wraps(f)
        def wrapped(ctx: "EvaluatorContext") -> Optional[EvaluatorResult]:
            try:
                raw = f(ctx)
            except Exception as e:  # noqa: BLE001 — capture, never propagate
                return EvaluatorResult(
                    value=None, error=str(e), error_type=type(e).__name__,
                    evaluator_name=eval_name,
                )
            return _normalize_result(raw, eval_name)

        wrapped._is_evaluator = True       # type: ignore[attr-defined]
        wrapped._evaluator_name = eval_name  # type: ignore[attr-defined]
        return wrapped

    if fn is not None and callable(fn):
        return decorate(fn)
    return decorate
```

Allowed return types (`bool`/`int`/`float`/`str`/`dict`/`EvaluatorResult`/`None`) and the `TypeError`
on anything else are unchanged. Note the spec's reference impl checks `isinstance(raw, bool)` after
`EvaluatorResult` but before `int` — preserved here, since `bool` is an `int` subclass.

Two implementation details that protect the caller:

- **No in-place mutation.** When the user returns an `EvaluatorResult`, we `dataclasses.replace()` it
  before filling in `evaluator_name` / `metric_type`. A user who holds a reference to the object they
  returned (e.g. for their own logging) must not see the SDK silently rewrite its fields.
- **No dict kwarg collision.** For a `dict` return, `evaluator_name` is applied via
  `setdefault` on a copy of the dict, not as a separate `evaluator_name=` kwarg. Otherwise a user
  dict that already includes `"evaluator_name"` would raise
  `TypeError: got multiple values for keyword argument 'evaluator_name'`. The user's value wins; we
  only default it when absent.

> Deviation from source reference: this spec adds `metric_type` back-inference for an
> `EvaluatorResult` whose `value` is set but `metric_type` is `None`. The source reference left it
> `None`; we fill it so `evaluate_and_submit` does not have to re-infer. Behavior for dict values is
> unchanged (left `None`).

### 5.2 `BaseEvaluator` (`eval/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

from .decorator import _normalize_result
from .types import EvaluatorContext, EvaluatorResult


class BaseEvaluator(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(self, context: EvaluatorContext) -> Any: ...

    def __call__(self, context: EvaluatorContext) -> Optional[EvaluatorResult]:
        try:
            raw = self.evaluate(context)
        except Exception as e:  # noqa: BLE001
            return EvaluatorResult(
                value=None, error=str(e), error_type=type(e).__name__,
                evaluator_name=self.name,
            )
        return _normalize_result(raw, self.name)

    # Parity with decorated functions so the (future) experiment runner can treat both uniformly.
    @property
    def _is_evaluator(self) -> bool:
        return True

    @property
    def _evaluator_name(self) -> str:
        return self.name
```

### 5.3 LLM-judge authoring (two paths)

The SDK provides `evaluator`, `BaseEvaluator`, `EvaluatorContext`, `EvaluatorResult`, `LLMClient`,
and — new in this spec — the structured-output classes and a provider-agnostic `LLMJudge` (§6).
Three supported patterns:

- **Option A — inline client:** user calls their own LLM SDK inside an `@evaluator` function and
  returns a `dict`/`EvaluatorResult`. SDK never sees the client.
- **Option B — `LLMClient` Protocol via `BaseEvaluator`:** user writes one adapter matching
  `LLMClient`, subclasses `BaseEvaluator`, and injects it.
- **Option C — `LLMJudge` (§6):** the user writes *no* evaluator class at all. They configure a
  prompt + a `StructuredOutput` + their own `LLMClient`, and `LLMJudge` handles prompt rendering,
  schema generation, response parsing, and pass/fail assessment. This is Option B made reusable.

The SDK ships **no** pre-baked metric implementations and imports **no** provider libraries in any
of these paths.

### 5.4 `evaluate_and_submit` (`EvalClient` method + module-level)

Run an evaluator on a context, then submit if it produced a non-error result. The span/trace target
follows the same resolution as `submit_evaluation` (§4.1a): use the context's IDs if it carries them,
else auto-bind to the active span, else emit unattached. It does **not** require the context to carry
IDs.

```python
def evaluate_and_submit(
    self,
    evaluator_fn: Callable[[EvaluatorContext], Optional[EvaluatorResult]],
    context: EvaluatorContext,
    *,
    eval_scope: Literal["span", "trace"] = "span",
) -> Optional[EvaluatorResult]:
    result = evaluator_fn(context)
    if result is None or result.error:
        return result
    # Pass IDs through only if the context carries them; otherwise submit_evaluation auto-binds
    # to the active span or emits unattached. None passes cleanly (no incomplete direct ref).
    self.submit_evaluation(
        span_id=context.span_id,
        trace_id=context.trace_id,
        eval_scope=eval_scope,
        label=result.evaluator_name,        # set by wrapper
        value=result.value,
        metric_type=result.metric_type,     # inferred by wrapper
        assessment=result.assessment,
        reasoning=result.reasoning,
        metadata=result.metadata,
        tags=result.tags,
    )
    return result
```

A `None` result (skip) or an errored result is returned without submitting. The error is **not**
swallowed — it remains visible on the returned `EvaluatorResult.error`.

> Note on the validation interaction: `submit_evaluation` rejects an *incomplete* direct reference
> (`span_id` xor `trace_id`). `EvaluatorContext` defaults both to `None`, so a context with neither
> passes through to auto-binding cleanly; a context that somehow carries only one is a genuine error
> and will (correctly) raise.

---

## 6. Surface 3 — Structured outputs + `LLMJudge`

This surface ports the ddtrace structured-output + `LLMJudge` design, **stripped of every provider
factory**. The classes generate JSON schemas, parse judge responses, and compute pass/fail — all
pure data logic with zero LLM dependencies. The LLM call itself is delegated entirely to a
user-supplied `LLMClient`.

### 6.1 `BaseStructuredOutput` and subclasses (`eval/structured_output.py`)

Each structured output knows three things: its `label` (the JSON key the judge must return), how to
render `to_json_schema()`, and (for the typed subclasses) how its thresholds map to a pass/fail
`assessment`.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union


class BaseStructuredOutput(ABC):
    description: str
    reasoning: bool
    reasoning_description: Optional[str]

    @property
    @abstractmethod
    def label(self) -> str: ...

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]: ...

    def _build_schema(self, label_schema: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {self.label: label_schema}
        required = [self.label]
        if self.reasoning:
            properties["reasoning"] = {
                "type": "string",
                "description": self.reasoning_description or "Explanation for the evaluation result",
            }
            required.append("reasoning")
        return {
            "type": "object", "properties": properties,
            "required": required, "additionalProperties": False,
        }


@dataclass
class BooleanStructuredOutput(BaseStructuredOutput):
    description: str
    reasoning: bool = False
    reasoning_description: Optional[str] = None
    pass_when: Optional[bool] = None              # passing condition for assessment

    @property
    def label(self) -> str: return "boolean_eval"

    def to_json_schema(self) -> dict[str, Any]:
        return self._build_schema({"type": "boolean", "description": self.description})


@dataclass
class ScoreStructuredOutput(BaseStructuredOutput):
    description: str
    min_score: float
    max_score: float
    reasoning: bool = False
    reasoning_description: Optional[str] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None

    @property
    def label(self) -> str: return "score_eval"

    def to_json_schema(self) -> dict[str, Any]:
        return self._build_schema({
            "type": "number", "description": self.description,
            "minimum": self.min_score, "maximum": self.max_score,
        })


@dataclass
class CategoricalStructuredOutput(BaseStructuredOutput):
    categories: dict[str, str]                    # value -> description
    reasoning: bool = False
    reasoning_description: Optional[str] = None
    pass_values: Optional[list[str]] = None       # categories that count as pass

    @property
    def label(self) -> str: return "categorical_eval"

    def to_json_schema(self) -> dict[str, Any]:
        any_of = [{"const": v, "description": d} for v, d in self.categories.items()]
        return self._build_schema({"type": "string", "anyOf": any_of})


StructuredOutput = Union[
    BooleanStructuredOutput, ScoreStructuredOutput, CategoricalStructuredOutput, dict[str, Any]
]
```

**Assessment rules** (computed from the parsed value, ported verbatim — keep the edge cases):

- `BooleanStructuredOutput`: `pass` iff `value == pass_when` (only when `pass_when` set, else `None`).
- `CategoricalStructuredOutput`: `pass` iff `value in pass_values` (only when set, else `None`).
- `ScoreStructuredOutput`:
  - both thresholds set, `max_threshold >= min_threshold` → **inclusive** range: `pass` iff
    `min_threshold <= value <= max_threshold`.
  - both set, `max_threshold < min_threshold` → **exclusive** range: `pass` iff
    `value < max_threshold or value > min_threshold`.
  - only `min_threshold` → `pass` iff `value >= min_threshold`.
  - only `max_threshold` → `pass` iff `value <= max_threshold`.
  - neither → `None`.

> **Schema-quirk handling dropped.** The ddtrace factories rewrote schemas per provider (Anthropic
> can't take `minimum`/`maximum`, Vertex needs `enum` instead of `const`, Bedrock stringifies the
> schema, etc.). Since Middleware emits **provider-neutral** schemas to a **user-supplied** client,
> that normalization is now the **user's adapter's** responsibility, not the SDK's. `to_json_schema`
> returns the canonical JSON-Schema form (`const` in `anyOf`, `minimum`/`maximum` on numbers); the
> adapter massages it for its provider if needed. This must be documented in the adapter-writing
> guide.

### 6.2 `LLMJudge` (`eval/judge.py`)

A `BaseEvaluator` subclass that renders a templated prompt from the `EvaluatorContext`, calls the
**user-supplied** `LLMClient` with the structured-output schema, parses the JSON response, and
returns an `EvaluatorResult` with value + reasoning + assessment.

```python
class LLMJudge(BaseEvaluator):
    def __init__(
        self,
        *,
        client: LLMClient,                       # REQUIRED, user-supplied. No provider= shortcut.
        model: str,
        user_prompt: str,                        # supports {{field.path}} templating
        system_prompt: Optional[str] = None,     # no templating
        structured_output: Optional[StructuredOutput] = None,
        model_params: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
    ):
        if client is None:
            raise ValueError("LLMJudge requires a user-supplied 'client' (LLMClient).")
        super().__init__(name=name or "llm_judge")
        self._client = client
        self._model = model
        self._user_prompt = user_prompt
        self._system_prompt = system_prompt
        self._structured_output = structured_output
        self._model_params = model_params

    def evaluate(self, context: EvaluatorContext) -> Any:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": self._render(self._user_prompt, context)})

        json_schema = None
        if self._structured_output is not None:
            json_schema = (
                self._structured_output
                if isinstance(self._structured_output, dict)
                else self._structured_output.to_json_schema()
            )

        # Call convention (matches the public LLMClient signature, §3):
        #   client(messages, model, json_schema=..., model_params=...) -> str
        # First two args are positional; the optionals are always passed by keyword so a user
        # adapter declared as `def adapter(messages, model, json_schema=None, model_params=None)`
        # binds correctly regardless of its parameter order for the optionals.
        raw = self._client(
            messages,
            self._model,
            json_schema=json_schema,
            model_params=self._model_params,
        )
        if self._structured_output is None:
            return raw                            # categorical str → wrapped by BaseEvaluator
        return self._parse_response(raw)          # returns EvaluatorResult
```

Key differences from the ddtrace reference (all intentional):

| Reference | Middleware spec |
|---|---|
| `provider=` + `client_options=` build a client via `_create_*_client` | **Removed.** `client` is required and user-supplied. |
| `LLMClient.__call__(provider, messages, json_schema, model, model_params)` | 4-arg form; `LLMJudge` calls `client(messages, model, json_schema=..., model_params=...)`. |
| `_build_publish_payload` / `_apply_variable_mapping` / `publish()` to register a server-side BYOP evaluator | **Out of scope here.** Online-eval registration is a separate platform concern; omit from v1. |
| Per-provider schema rewriting inside factories | Dropped — adapter's job (§6.1 note). |

**Template rendering** (`_render`) and **response parsing** (`_parse_response`) port directly:

- `_render`: `asdict(context)` then resolve `{{a.b.c}}` dotted paths; `None` → `""`; dict/list →
  `json.dumps(..., indent=2)`. Note context field names are `input`/`output`/`expected_output`/
  `metadata` (this SDK's `EvaluatorContext`), **not** the reference's `input_data`/`output_data`.
  Document the available placeholders accordingly: `{{input}}`, `{{output}}`, `{{expected_output}}`,
  `{{retrieved_contexts}}`, `{{metadata.key}}`.
- `_parse_response`: `json.loads` (raise `ValueError` on bad JSON / non-object); extract
  `data[label]`; type-check against the structured output (`bool`/number/`str`); compute
  `assessment`; attach `reasoning` only if `structured_output.reasoning` is true; stash
  `{"raw_response": data}` in `metadata`. A custom-`dict` schema returns
  `EvaluatorResult(value=data, reasoning=data.get("reasoning"))`.
  - **Custom-`dict` schema has no `assessment`.** Only the three typed `StructuredOutput` subclasses
    compute pass/fail (they carry the thresholds/`pass_*` config). A raw `dict[str, Any]` schema is
    opaque to the SDK, so the result has `assessment=None`. If the user wants pass/fail for a custom
    schema, they set it themselves (return an `EvaluatorResult` with `assessment=...`, or use a typed
    subclass). Document this so it isn't read as a bug.
  - **`metric_type` is set explicitly, not back-inferred, for typed outputs.** `_parse_response`
    knows the structured output's kind, so it sets `metric_type` directly: `BooleanStructuredOutput`
    → `"boolean"`, `ScoreStructuredOutput` → `"score"`, `CategoricalStructuredOutput` →
    `"categorical"`. The custom-`dict` path returns a `dict` *value* with **no** `metric_type`; that
    result then relies on `_normalize_result`'s back-inference (§5.1), which leaves `dict` values'
    `metric_type` as `None` — so a custom-dict judge result will fail
    `submit_evaluation`/`evaluate_and_submit` unless the user supplies a `metric_type`. This coupling
    between §6.2 and §5.1 is intentional but must be called out in both the docstring and the
    adapter guide.

Because `LLMJudge` is a `BaseEvaluator`, its `__call__` already wraps `evaluate()` with the shared
error capture (§5.2): a judge that raises (bad JSON, client error) yields
`EvaluatorResult(error=..., error_type=...)` rather than blowing up an enclosing loop.

### 6.3 Usage

```python
import json
from middleware.llmobs import (
    LLMJudge, ScoreStructuredOutput, EvaluatorContext, evaluate_and_submit,
)

# 1. User writes ONE adapter matching LLMClient (their dep, not the SDK's).
from openai import OpenAI
_openai = OpenAI()

def openai_client(messages, model, json_schema=None, model_params=None):
    kwargs = {"model": model, "messages": messages, **(model_params or {})}
    if json_schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "evaluation", "strict": True, "schema": json_schema},
        }
    return _openai.chat.completions.create(**kwargs).choices[0].message.content or ""

# 2. Configure a judge — no provider library touched by the SDK.
judge = LLMJudge(
    client=openai_client,
    model="gpt-4o-mini",
    name="faithfulness",
    system_prompt="You evaluate answer faithfulness.",
    user_prompt="Context:\n{{retrieved_contexts}}\n\nAnswer:\n{{output}}",
    structured_output=ScoreStructuredOutput(
        description="How grounded the answer is in the context",
        min_score=0, max_score=1, reasoning=True, min_threshold=0.7,
    ),
)

# 3. Run + submit. Inside an active span, the IDs auto-bind — no need to read them off the span.
ctx = EvaluatorContext(input=question, output=answer, retrieved_contexts=docs)
with tracer.start_as_current_span("rag_answer"):
    evaluate_and_submit(judge, ctx)   # judge(ctx) -> EvaluatorResult -> submit_evaluation (auto-bound)
```

### 6.4 End-to-end example (trace → eval → flush)

The whole loop, from registering the tracer to flushing the eval on shutdown. This is the canonical
example to mirror in the README:

```python
from opentelemetry import trace
from middleware.llmobs import (
    register, evaluator, EvaluatorContext, submit_evaluation,
    evaluate_and_submit, flush_evaluations,
)

# 1. Standard tracing setup (unchanged from the existing SDK).
register(service_name="rag-app", auto_instrument=True)
tracer = trace.get_tracer(__name__)

# 2. A pure-function evaluator — zero LLM deps.
@evaluator(name="answer_nonempty")
def answer_nonempty(ctx: EvaluatorContext) -> bool:
    return bool(ctx.output and ctx.output.strip())

# 3. Inside a traced request, run the LLM and attach evals to the active span.
def handle(question: str) -> str:
    with tracer.start_as_current_span("chat") as span:
        answer = run_llm(question)               # produces an LLM span underneath
        ctx = EvaluatorContext(input=question, output=answer)

        # (a) local evaluator -> auto-bound to the active span
        evaluate_and_submit(answer_nonempty, ctx)

        # (b) an externally-computed score -> auto-bound, no IDs threaded through
        submit_evaluation(label="toxicity", value=external_toxicity(answer), metric_type="score")

        return answer

# 4. On shutdown (or in a test), make sure evals are exported.
if not flush_evaluations():
    log.warning("middleware-llmobs: eval flush did not fully complete")
```

Note `handle()` never touches a span ID: `start_as_current_span` installs the span in the OTel
context, and both `evaluate_and_submit` and `submit_evaluation` read it back via `get_current_span()`
(§4.1a). The only place IDs become explicit is when you submit *outside* the active context — then
capture them with `export_current_span()` first.

---

## 7. `EvalClient` and lifecycle (`eval/client.py`, `eval/_emit.py`)

```python
class EvalClient:
    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,     # falls back to OTEL_EXPORTER_OTLP_ENDPOINT
        headers: Optional[dict[str, str]] = None,  # falls back to OTEL_EXPORTER_OTLP_HEADERS (auth)
        ml_app: Optional[str] = None,       # falls back to MW_PROJECT_NAME / OTEL_SERVICE_NAME
    ): ...

    def submit_evaluation(self, *, label, value, ...) -> None: ...
    def evaluate_and_submit(self, evaluator_fn, context) -> Optional[EvaluatorResult]: ...
    def flush(self) -> None: ...
```

Internals (`_emit.py`):

- Lazily build a dedicated `LoggerProvider` (with `BatchLogRecordProcessor` + `OTLPLogExporter`
  pointed at `{endpoint}/v1/logs`) and `MeterProvider` (with `PeriodicExportingMetricReader` +
  `OTLPMetricExporter` pointed at `{endpoint}/v1/metrics`). These are **separate from the global
  trace `TracerProvider`** so eval emission never interferes with span export.
- Endpoint/header resolution reuses `settings.get_env_collector_endpoint`,
  `settings.get_env_client_headers`, and `otel._normalized_endpoint` (extended with `/v1/logs` and
  `/v1/metrics` path constructors alongside the existing `/v1/traces` one).
- Resource attributes on both providers: `service.name`, `mw.account_key` is **not** set by the SDK
  (the auth key travels in the `Authorization` header exactly as in the trace path).
- `flush()` calls `force_flush()` on both providers.

Module-level functions (`submit_evaluation`, `evaluate_and_submit`, `flush_evaluations`) delegate to
a process-wide default `EvalClient` created on first use from env vars.

---

## 8. Testing

Mirror the existing `tests/test_otel.py` style (env reset fixture, mock exporters):

- **Types:** `infer_metric_type` ordering (`True` → `"boolean"`, not `"score"`); `dict` raises.
- **Validation:** each `ValueError` path in §4.2 (both/neither join target, bad label, regex,
  type mismatch, non-serializable metadata).
- **Decorator:** each return type → correct `EvaluatorResult`; exception → `error`/`error_type`
  populated, not raised; `None` → `None`; bad type → `TypeError`; `_is_evaluator`/`_evaluator_name`
  set.
- **BaseEvaluator:** same normalization; `evaluate` raising is captured.
- **Structured outputs (§6.1):** `to_json_schema()` shape for each subclass (label key, `reasoning`
  property toggled, `required`, `additionalProperties: False`, `const`/`anyOf` for categorical,
  `minimum`/`maximum` for score). Assessment rules table — especially the score **exclusive range**
  (`max_threshold < min_threshold`) and the bool-before-anything ordering.
- **LLMJudge (§6.2):** with a **fake** `LLMClient` (a plain Python function returning canned JSON,
  zero provider imports): `_render` substitutes `{{output}}`/`{{metadata.key}}` and blanks unknown
  paths; `_parse_response` returns the right value/assessment/reasoning; bad JSON → captured into
  `EvaluatorResult.error` via `BaseEvaluator.__call__` (does not raise); type mismatch (e.g. judge
  returns a string for a `ScoreStructuredOutput`) → captured error; custom-dict schema path returns
  `value=data`. Assert `LLMJudge(client=None, ...)` raises `ValueError`.
- **Emission:** with a mock `LoggerProvider`/`MeterProvider` (or in-memory exporters), assert one log
  record with `gen_ai.evaluation.*` attributes and the three gauges. Assert the record's own
  `trace_id`/`span_id` are set for an explicit/auto-bound span, `trace_id`-only for
  `eval_scope="trace"`, and **both unset** for `join_on_tag` (tag in
  `gen_ai.evaluation.tag_key`/`tag_value`) and for the unattached case.
- **Auto-binding:** inside `tracer.start_as_current_span(...)`, calling `submit_evaluation` with no
  IDs binds the record to that span's context; outside any span it emits unattached (no
  `trace_id`/`span_id`, no exception). `export_current_span()` returns the IDs inside a span and
  `None` outside.
- **No LLM import (critical):** import the **whole** eval subpackage (`middleware.llmobs.eval`,
  including `structured_output` and `judge`) and assert none of
  `openai`/`anthropic`/`boto3`/`vertexai`/`google.cloud.aiplatform`/`cohere`/`mistralai` appear in
  `sys.modules`. This guards the single biggest deviation from the ddtrace reference.

---

## 9. Open questions (for the platform team)

1. **Tag-based correlation on the ingest side** when emitted as an OTel log record — confirm which
   attribute keys the pipeline matches on (`gen_ai.evaluation.tag_key` / `tag_value` proposed), or
   whether the key/value must instead live in the JSON `body`. (For span/trace correlation we rely on
   the OTel-native log-record `trace_id`/`span_id` fields, same as the trace path — no Datadog-style
   `join_on` envelope is emitted.)
2. **`mw.llm.source` value** — confirm `"sdk-eval"` is an acceptable discriminator vs. the online
   `"eval"` source, or if a shared value + a separate boolean attribute is preferred.
3. **Boolean → score coercion** for the `gen_ai.evaluations.score` gauge (0.0/1.0) — confirm
   dashboards expect this, matching the server reference's `score_value` handling.
4. **`LLMJudge.publish()` / BYOP registration** — the ddtrace reference can register a judge as a
   server-side online evaluator (`_build_publish_payload`, `byop_config`, `_PUBLISH_PROVIDER_MAPPING`).
   Is there a Middleware endpoint for SDK-side evaluator registration? If so it is a future surface;
   if not, confirm we omit `publish()` from v1 (current assumption).
5. **Experiment runner ingestion** — out of scope here; needs a dedicated experiment-record surface
   before we can spec `experiment(...)`.
