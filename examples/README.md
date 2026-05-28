# Examples

Runnable examples for `middleware-llmobs`. Every example shares a single
[`requirements.txt`](requirements.txt) at this folder's root, so one venv covers all of them.

| Folder | What it shows |
|---|---|
| [`01_basic_tracing/`](01_basic_tracing/) | Wire `register()`, call OpenAI, watch an LLM span land in Middleware. |
| [`02_submit_evaluation/`](02_submit_evaluation/) | Run a real OpenAI call, compute evaluations in your code, attach them to the active span via `submit_evaluation`. |

## One-time setup

From this folder:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Then export the credentials every example needs:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=<your-middleware-ingestion-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
```

## Run any example

```bash
python 01_basic_tracing/app.py
python 02_submit_evaluation/app.py
```

Each example's `README.md` explains what to expect in the terminal and in your Middleware
dashboard.
