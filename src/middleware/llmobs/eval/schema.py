"""``format_schema_for_provider`` — turn a canonical JSON Schema into a provider's request kwargs.

This is a **user-facing convenience** for writing :class:`LLMClient` adapters. The user calls it
from their own code, passing the provider name explicitly. It performs **pure dict manipulation
only** — it never imports ``openai``/``anthropic``/``boto3``/``vertexai`` or any provider library,
so the SDK keeps zero LLM dependencies.

It returns the request kwargs to spread into the provider's call::

    kwargs = {"model": model, "messages": messages}
    kwargs.update(format_schema_for_provider(json_schema, "openai"))
    resp = openai_client.chat.completions.create(**kwargs)

The returned keys differ per provider (each wraps structured output differently):

==============  ===================================================
Provider        Returned keys
==============  ===================================================
openai          ``response_format``
azure_openai    ``response_format``
anthropic       ``extra_headers``, ``extra_body``
vertexai        ``generation_config`` (``response_mime_type`` + ``response_schema``)
bedrock         ``outputConfig``
==============  ===================================================

``to_json_schema()`` produces the canonical form (``const`` in ``anyOf``, ``minimum``/``maximum`` on
numbers); this helper rewrites it to satisfy each provider's quirks.
"""

import copy
import json
from typing import Any, Literal

Provider = Literal["openai", "azure_openai", "anthropic", "vertexai", "bedrock"]

_SCHEMA_NAME = "evaluation"


def _strip_number_bounds(schema: dict[str, Any]) -> dict[str, Any]:
    """Move ``minimum``/``maximum`` into the description (Anthropic/Bedrock reject them)."""
    out = copy.deepcopy(schema)
    for prop in out.get("properties", {}).values():
        if not isinstance(prop, dict):
            continue
        if prop.get("type") == "number":
            min_val = prop.pop("minimum", None)
            max_val = prop.pop("maximum", None)
            if min_val is not None or max_val is not None:
                bounds = f" (range: {min_val} to {max_val})"
                prop["description"] = prop.get("description", "") + bounds
        # These providers don't allow 'type' alongside 'anyOf'.
        if "anyOf" in prop:
            prop.pop("type", None)
    return out


def _const_to_enum(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert ``anyOf`` of ``const`` values into an ``enum`` (Vertex rejects ``const``)."""
    out = copy.deepcopy(schema)
    for prop in out.get("properties", {}).values():
        if not isinstance(prop, dict) or "anyOf" not in prop:
            continue
        enum_values = [item["const"] for item in prop["anyOf"] if "const" in item]
        if enum_values:
            prop.pop("anyOf")
            prop["enum"] = enum_values
    return out


def format_schema_for_provider(schema: dict[str, Any], provider: Provider) -> dict[str, Any]:
    """Return the request kwargs that make ``provider`` emit JSON matching ``schema``.

    Args:
        schema: A canonical JSON Schema (e.g. from ``StructuredOutput.to_json_schema()``).
        provider: One of ``"openai"``, ``"azure_openai"``, ``"anthropic"``, ``"vertexai"``,
            ``"bedrock"``.

    Returns:
        A dict of keyword arguments to spread into the provider's create/converse call.

    Raises:
        ValueError: If ``provider`` is not one of the supported values.
    """
    if provider in ("openai", "azure_openai"):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": _SCHEMA_NAME, "strict": True, "schema": schema},
            }
        }

    if provider == "anthropic":
        return {
            "extra_headers": {"anthropic-beta": "structured-outputs-2025-11-13"},
            "extra_body": {
                "output_format": {"type": "json_schema", "schema": _strip_number_bounds(schema)}
            },
        }

    if provider == "vertexai":
        return {
            "generation_config": {
                "response_mime_type": "application/json",
                "response_schema": _const_to_enum(schema),
            }
        }

    if provider == "bedrock":
        return {
            "outputConfig": {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {
                        "jsonSchema": {
                            "name": _SCHEMA_NAME,
                            "schema": json.dumps(_strip_number_bounds(schema)),
                        }
                    },
                }
            }
        }

    raise ValueError(
        f"Unsupported provider {provider!r}. Expected one of: "
        "openai, azure_openai, anthropic, vertexai, bedrock."
    )
