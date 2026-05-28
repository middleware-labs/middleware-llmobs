# middleware-llmobs

LLM Observability for the [Middleware](https://middleware.io/) SaaS platform, built on
[OpenInference](https://github.com/Arize-ai/openinference) and OpenTelemetry.

`middleware-llmobs` provides a one-line `register()` that configures an OpenTelemetry
`TracerProvider` to export OpenInference traces to your Middleware collector over OTLP/HTTP.

> **Transport:** Only **HTTP/protobuf** is supported. gRPC is intentionally disabled — requesting
> it raises `NotImplementedError`. (The gRPC code path is retained in the source for future use.)

## Install

```bash
pip install middleware-llmobs
```

## Quickstart

Set the standard OTLP environment variables for your Middleware account:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=<your-key>"
export OTEL_SERVICE_NAME="my-llm-app"
```

Then register and auto-instrument every installed OpenInference library:

```python
from middleware.llmobs import register

providers = register(auto_instrument=True)  # batch + HTTP by default
# providers.tracer / providers.logger / providers.meter — the wired OTel providers.
```

### Explicit configuration

```python
from middleware.llmobs import register

providers = register(
    endpoint="https://uid.middleware.io:443",
    headers={"Authorization": "<your-key>"},
    service_name="my-llm-app",
)
```

### Manual instrumentor wiring with GenAI semantic conventions

```python
from middleware.llmobs import register
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor

providers = register(service_name="my-llm-app")

OpenAIInstrumentor().instrument(
    tracer_provider=providers.tracer,
    config=TraceConfig(enable_genai_semconv=True),
)
```

When `auto_instrument=True`, `register()` passes
`TraceConfig(enable_genai_semconv=True)` to each instrumentor automatically (falling back
gracefully for instrumentors that don't accept a `config` argument).

## Environment variables

| Variable | Purpose | Example |
|---|---|---|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Collector endpoint | `https://uid.middleware.io:443` |
| `OTEL_EXPORTER_OTLP_HEADERS` | Export headers (auth) | `Authorization=<key>` |
| `OTEL_SERVICE_NAME` | Service identity | `my-llm-app` |
| `MW_PROJECT_NAME` | Optional project name (defaults to service name) | `my-project` |

## License

Apache-2.0
