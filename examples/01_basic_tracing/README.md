# 01 · Basic tracing

The shortest possible end-to-end use of `middleware-llmobs`: one `register()` call, one real
OpenAI chat completion, one trace landing in your Middleware dashboard.

## What it shows

- **`register(service_name=..., auto_instrument=True)`** — sets up the global TracerProvider over
  Middleware's OTLP HTTP/protobuf endpoint and auto-attaches every installed OpenInference
  instrumentor. Because `openinference-instrumentation-openai` is in `requirements.txt`, every
  `client.chat.completions.create(...)` call becomes an LLM span automatically — no manual
  instrumentor wiring.
- **A parent `"chat"` span** — gives the whole request a top-level name in the trace view. The
  OpenAI LLM span becomes its child.
- **`providers.tracer.force_flush()`** — drains the BatchSpanProcessor before the script exits.
  Short-lived processes need this; long-running servers don't.

## Requirements

Everything you need is in the shared [`examples/requirements.txt`](../requirements.txt):

| Package | Why |
|---|---|
| `middleware-llmobs` | This SDK. |
| `openinference-instrumentation-openai` | Auto-instrumentation for the OpenAI SDK. |
| `openai` | The actual LLM client the script calls. |

## Run

From the `examples/` directory (see [`examples/README.md`](../README.md) for one-time setup):

```bash
python 01_basic_tracing/app.py
```

You'll need these env vars:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=<your-middleware-ingestion-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
```

## What you should see

In the terminal: the model's one-sentence summary plus a usage line with prompt/completion
token counts.

In your Middleware dashboard, under the `basic-tracing-example` service, a trace with:

- A root span named **`chat`** with `input.value` / `output.value` attributes.
- A child LLM span emitted by the OpenAI instrumentor with `gen_ai.*` attributes
  (model, messages, response, token usage). `auto_instrument=True` enables the GenAI
  semantic conventions for you.

## Tweaks worth trying

- Swap the model in `app.py` (`gpt-4o-mini` → `gpt-4o`) and re-run; the LLM span attribute will
  reflect the change.
- Drop `auto_instrument=True` and wire the instrumentor manually:
  ```python
  from openinference.instrumentation.openai import OpenAIInstrumentor
  OpenAIInstrumentor().instrument(tracer_provider=providers.tracer)
  ```
- Remove the `force_flush()` and notice the span sometimes doesn't arrive — that's the
  BatchSpanProcessor losing its buffer when the process exits.
