# 02 · Toxicity LLM-as-judge

A real end-to-end **LLM-as-judge** example. A small "support agent" generates a reply to a
customer message, then an `LLMJudge` (powered by another OpenAI call) classifies the reply for
toxicity and ships the verdict to Middleware.

## What it shows

- **`LLMJudge` with `BooleanStructuredOutput`.** The judge is constrained to return JSON of the
  shape `{"boolean_eval": true|false, "reasoning": "..."}`. `pass_when=False` means the eval
  **passes** when the answer is *not* toxic.
- **BYO `LLMClient` adapter.** The SDK never imports `openai`. A 10-line user-written adapter
  (`openai_judge_client`) bridges the judge to the OpenAI SDK. The
  [`format_schema_for_provider`](../../src/middleware/llmobs/eval/schema.py) helper handles
  OpenAI's `response_format` envelope so the adapter stays trivial.
- **`evaluate_and_submit(judge, ctx)`.** Calls the judge, ships the result via
  `submit_evaluation` on success, **or** `submit_evaluation_error` on failure — errored judges
  surface in the dashboard rather than getting silently dropped.
- **Auto-binding.** Neither call passes `span_id`/`trace_id`. The judge eval auto-correlates to
  the parent `"chat"` span via the active OTel context.
- **Two designed inputs** — one polite, one openly hostile — so you see both `pass` and `fail`
  verdicts side-by-side in the dashboard.

## Requirements

Shared with the other examples in [`examples/requirements.txt`](../requirements.txt):

| Package | Why |
|---|---|
| `middleware-llmobs` | This SDK. |
| `openinference-instrumentation-openai` | Auto-instruments both the support-reply call and the judge call. |
| `openai` | The LLM client (used by both the app and the judge adapter). |

## Run

From the `examples/` directory (see [`examples/README.md`](../README.md) for one-time setup):

```bash
python 02_toxicity_llm_judge/app.py
```

Env vars:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="https://uid.middleware.io:443"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=<your-middleware-ingestion-key>"
export OPENAI_API_KEY="<your-openai-api-key>"
```

## What you should see

In the terminal — two cycles, each printing the customer message, the agent reply, and the
judge's verdict + one-sentence reasoning. Something like:

```
=== customer: Hi! My order #12345 still hasn't arrived after 10 days. ...
--- agent reply ---
Sorry to hear that! Let me look up order #12345 right away ...
  toxicity = False  verdict = pass  reasoning = The reply is helpful and respectful ...

=== customer: Your delivery is GARBAGE. I hate this company ...
--- agent reply ---
I understand your frustration, but please don't insult our team ...
  toxicity = False  verdict = pass  reasoning = ...
```

(The second reply may or may not come back toxic depending on the model's mood — that's part of
why you ship this eval in the first place.)

In your Middleware dashboard, under the `toxicity-judge-example` service:

- Two `chat` traces, each with **two** LLM child spans — one for the agent reply, one for the
  judge call.
- Two `gen_ai.evaluation.*` log records (one per input) carrying `gen_ai.evaluation.verdict`
  (`pass`/`fail`), `gen_ai.evaluation.explanation` (the judge's reasoning), and
  `gen_ai.evaluation.score.value` (1.0 = pass, 0.0 = fail for booleans).
- Eval metric data points on `gen_ai.evaluations.{count,score,outcome}` tagged with the
  `toxicity` label and the verdict.

## Tweaks worth trying

- Change `pass_when=False` to `pass_when=True` and re-run — the verdicts flip. The dashboard
  will then read "pass = the answer **was** toxic," useful as a sanity check.
- Swap the judge model (`gpt-4o-mini` → `gpt-4o`) by editing the `LLMJudge(model=...)` line.
  More capable judges = fewer false positives, more $/eval (which is why `submit_evaluation`
  accepts a `cost_usd` kwarg).
- Break the judge intentionally — point the adapter at `model="not-a-real-model"`. The OpenAI
  call raises, `BaseEvaluator.__call__` catches it, and `evaluate_and_submit` ships a
  `submit_evaluation_error` record instead of crashing the request.
