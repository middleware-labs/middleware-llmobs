"""Example 01 — Basic tracing.

What this script does, end to end:

1. Calls ``middleware.llmobs.register(..., auto_instrument=True)``. That:
     - Creates an OpenTelemetry TracerProvider configured for Middleware's HTTP/protobuf
       OTLP endpoint (``/v1/traces``), reading ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
       ``OTEL_EXPORTER_OTLP_HEADERS`` from the environment.
     - Discovers every installed OpenInference instrumentor via Python entry points and
       attaches it to the new tracer provider. With ``openinference-instrumentation-openai``
       in this folder's ``requirements.txt``, every ``client.chat.completions.create(...)``
       call is automatically wrapped in an LLM span — no manual instrumentor wiring.
     - Also registers a global LoggerProvider + MeterProvider that the eval surface uses
       (not exercised in this example; see ``02_submit_evaluation``).

2. Runs a small "summarise this paragraph" prompt against OpenAI's Chat Completions API.
   Because of the auto-instrumentation in step 1, this produces:
     - A parent span ``"chat"`` (created explicitly below to give the trace a name).
     - A child LLM span emitted by the OpenAI instrumentor, carrying:
         * ``llm.model_name`` / ``llm.invocation_parameters``
         * the input messages and the assistant reply (under OpenInference semantic
           conventions; with ``enable_genai_semconv=True`` you also get ``gen_ai.*``
           attributes — ``auto_instrument=True`` turns that on for you)
         * token usage from the response

3. Flushes the BatchSpanProcessor on the tracer provider so the spans actually leave the
   process before ``python app.py`` exits.

Environment variables required:

    OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
    OTEL_EXPORTER_OTLP_HEADERS="Authorization=<your-middleware-ingestion-key>"
    OPENAI_API_KEY="<your-openai-api-key>"

Optional:

    OTEL_SERVICE_NAME="basic-tracing-example"   # else passed via service_name= below
"""

from __future__ import annotations

import os
import sys

from openai import OpenAI
from opentelemetry import trace

from middleware.llmobs import register

# Hard fail early with a useful message instead of crashing inside the OTLP exporter.
for var in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS", "OPENAI_API_KEY"):
    if not os.environ.get(var):
        sys.exit(f"Missing required environment variable: {var}")

# 1. Register the SDK. Returns a Providers triple — we hold onto it so we can flush at exit.
providers = register(
    service_name="basic-tracing-example",
    auto_instrument=True,  # picks up openinference-instrumentation-openai
)

tracer = trace.get_tracer(__name__)
client = OpenAI()  # OPENAI_API_KEY is read from the environment

# 2. A small real prompt. The parent "chat" span gives the whole call a top-level name in the
# trace view; the OpenAI instrumentor adds the LLM child span automatically.
PROMPT = (
    "Summarise this in one sentence:\n"
    "OpenTelemetry is a CNCF observability framework that standardises how applications "
    "produce traces, metrics, and logs, and how those signals are exported to any compatible "
    "backend over a single wire protocol called OTLP."
)

with tracer.start_as_current_span("chat"):
    # The OpenAI auto-instrumentor records prompt + response on the child LLM span
    # (gen_ai.* attributes), so no manual attribute-setting is needed on the parent.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise technical writer."},
            {"role": "user", "content": PROMPT},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    answer = response.choices[0].message.content or ""

print("\n--- assistant ---")
print(answer)
print("\n--- usage ---")
print(
    f"model={response.model}  "
    f"prompt_tokens={response.usage.prompt_tokens}  "
    f"completion_tokens={response.usage.completion_tokens}"
)

# 3. Drain the BatchSpanProcessor. Without this, short-lived scripts can exit before the spans
# are exported.
providers.tracer.force_flush()
print("\nSpans flushed. Check your Middleware dashboard for the 'basic-tracing-example' service.")
