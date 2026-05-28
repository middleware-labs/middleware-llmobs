"""Structured-output specs for LLM-judge evaluators: JSON-schema generation + pass/fail rules.

Pure data logic — no LLM dependency. Schemas are provider-neutral; a user's ``LLMClient`` adapter
massages them for its provider if needed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

from .types import Assessment


class BaseStructuredOutput(ABC):
    """Abstract base for LLM-judge structured outputs."""

    description: str
    reasoning: bool
    reasoning_description: Optional[str]

    @property
    @abstractmethod
    def label(self) -> str:
        """The JSON key the judge must return the evaluation value under."""

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """The JSON schema for the structured response."""

    def _build_schema(self, label_schema: dict[str, Any]) -> dict[str, Any]:
        properties: dict[str, Any] = {self.label: label_schema}
        required = [self.label]
        if self.reasoning:
            properties["reasoning"] = {
                "type": "string",
                "description": (
                    self.reasoning_description or "Explanation for the evaluation result"
                ),
            }
            required.append("reasoning")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


@dataclass
class BooleanStructuredOutput(BaseStructuredOutput):
    """True/false evaluation. ``pass_when`` defines the passing condition for assessment."""

    description: str
    reasoning: bool = False
    reasoning_description: Optional[str] = None
    pass_when: Optional[bool] = None

    @property
    def label(self) -> str:
        return "boolean_eval"

    def to_json_schema(self) -> dict[str, Any]:
        return self._build_schema({"type": "boolean", "description": self.description})


@dataclass
class ScoreStructuredOutput(BaseStructuredOutput):
    """Numeric score within a range. ``min_threshold``/``max_threshold`` drive pass/fail.

    - both set, ``max_threshold >= min_threshold``: inclusive range ``[min, max]``.
    - both set, ``max_threshold < min_threshold``: exclusive range (outside ``(max, min)`` passes).
    - only one set: simple ``>=`` / ``<=`` comparison.
    """

    description: str
    min_score: float
    max_score: float
    reasoning: bool = False
    reasoning_description: Optional[str] = None
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None

    @property
    def label(self) -> str:
        return "score_eval"

    def to_json_schema(self) -> dict[str, Any]:
        return self._build_schema(
            {
                "type": "number",
                "description": self.description,
                "minimum": self.min_score,
                "maximum": self.max_score,
            }
        )


@dataclass
class CategoricalStructuredOutput(BaseStructuredOutput):
    """Select one of predefined categories. ``pass_values`` defines the passing categories."""

    categories: dict[str, str]
    reasoning: bool = False
    reasoning_description: Optional[str] = None
    pass_values: Optional[list[str]] = None

    @property
    def label(self) -> str:
        return "categorical_eval"

    def to_json_schema(self) -> dict[str, Any]:
        any_of = [{"const": value, "description": desc} for value, desc in self.categories.items()]
        return self._build_schema({"type": "string", "anyOf": any_of})


StructuredOutput = Union[
    BooleanStructuredOutput,
    ScoreStructuredOutput,
    CategoricalStructuredOutput,
    dict[str, Any],
]


def compute_assessment(structured_output: BaseStructuredOutput, value: Any) -> Optional[Assessment]:
    """Compute pass/fail from the structured output's thresholds, or ``None`` if unconfigured."""
    if isinstance(structured_output, BooleanStructuredOutput):
        if structured_output.pass_when is not None:
            return "pass" if value == structured_output.pass_when else "fail"
        return None

    if isinstance(structured_output, CategoricalStructuredOutput):
        if structured_output.pass_values is not None:
            return "pass" if value in structured_output.pass_values else "fail"
        return None

    if isinstance(structured_output, ScoreStructuredOutput):
        min_t, max_t = structured_output.min_threshold, structured_output.max_threshold
        if min_t is not None and max_t is not None:
            if max_t >= min_t:
                return "pass" if min_t <= value <= max_t else "fail"
            return "pass" if value < max_t or value > min_t else "fail"
        if min_t is not None:
            return "pass" if value >= min_t else "fail"
        if max_t is not None:
            return "pass" if value <= max_t else "fail"

    return None
