"""Structured JSON parsing and validation for LLM control outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from controller.actuators import ControlAction


class LLMOutputValidationError(ValueError):
    """Raised when an LLM response cannot be parsed into a valid action."""


class LLMControlResponse(BaseModel):
    """Strict response schema expected from the LLM."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    cooling_setpoint: float
    fan_speed: float
    lighting: Literal["ON", "OFF"]
    reason: str = Field(min_length=1, max_length=140)


@dataclass
class LLMOutputParser:
    """Converts raw LLM output into a validated ControlAction."""

    cooling_min: float = 22.0
    cooling_max: float = 27.0
    fan_min: float = 30.0
    fan_max: float = 90.0

    def parse(self, response_text: str | dict[str, Any]) -> ControlAction:
        """Parses and cleans the raw LLM output into a ControlAction."""
        data = self._decode_payload(response_text)
        
        # ---------------------------------------------------------
        # HACKATHON FIX: Coerce the 'lighting' value to strict UPPERCASE
        # ---------------------------------------------------------
        if "lighting" in data:
            val = data["lighting"]
            if val in [True, "true", "True", "On", "on", "ON"]:
                data["lighting"] = "ON"
            elif val in [False, "false", "False", "Off", "off", "OFF"]:
                data["lighting"] = "OFF"
            else:
                data["lighting"] = "ON"  # Safe default fallback

        # 3. Validate the cleaned data using the Pydantic model
        model = self._validate_with_dynamic_bounds(data)
        
        # 4. Explicitly return the ControlAction object to prevent NoneType errors!
        return ControlAction(
            cooling_setpoint=model.cooling_setpoint,
            fan_speed=model.fan_speed,
            lighting=model.lighting,
            reason=model.reason,
        )

    def schema(self) -> dict[str, Any]:
        """Returns the JSON schema for structured generation use-cases."""
        return LLMControlResponse.model_json_schema()

    def _decode_payload(self, raw_response: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(raw_response, dict):
            return raw_response

        if not isinstance(raw_response, str):
            raise LLMOutputValidationError("LLM response must be a JSON string or dict.")

        text = raw_response.strip()
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMOutputValidationError(f"Invalid JSON payload: {exc}") from exc

        if not isinstance(decoded, dict):
            raise LLMOutputValidationError("LLM response JSON must be an object.")
        return decoded

    def _validate_with_dynamic_bounds(self, payload: dict[str, Any]) -> LLMControlResponse:
        try:
            model = LLMControlResponse.model_validate(payload)
        except ValidationError as exc:
            raise LLMOutputValidationError(exc.json()) from exc

        if not (self.cooling_min <= model.cooling_setpoint <= self.cooling_max):
            raise LLMOutputValidationError(
                "cooling_setpoint violates configured cooling bounds."
            )
        if not (self.fan_min <= model.fan_speed <= self.fan_max):
            raise LLMOutputValidationError("fan_speed violates configured fan bounds.")
        return model