"""Deterministic prompt construction for HVAC control decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from controller.actuators import ControlAction
from controller.sensors import SensorSnapshot


@dataclass(frozen=True)
class PromptSafetyLimits:
    """Safety limits communicated to the LLM for bounded decisions."""

    cooling_min: float = 22.0
    cooling_max: float = 27.0
    fan_min: float = 30.0
    fan_max: float = 90.0


class PromptBuilder:
    """Builds deterministic system and user messages for Ollama chat."""

    def __init__(self, limits: PromptSafetyLimits | None = None) -> None:
        self._limits = limits or PromptSafetyLimits()

    def build_system_prompt(self) -> str:
        """Returns strict generation rules for structured HVAC output."""
        return (
            "You are an autonomous smart-building HVAC controller. "
            "Your goal is to balance human comfort (PMV) with aggressive energy reduction. "
            "CRITICAL RULES: "
            "1. ZERO-OCCUPANCY ECO MODE: If 'occupancy' is 0, human comfort does not matter. "
            "You MUST set 'cooling_setpoint' to 27.0 and 'lighting' to OFF to maximize savings. "
            "2. OCCUPIED COMFORT: If 'occupancy' > 0, you MUST maintain 'pmv_comfort' strictly between -0.5 and +0.5. "
            "3. FREE COOLING: If 'outdoor_temperature' is lower than 'indoor_temperature', "
            "maximize 'fan_speed' to use outside air and raise the cooling setpoint to save compressor energy. "
            "4. MICRO-ADJUSTMENTS: Do not swing temperatures wildly. Adjust setpoints gradually based on the previous state. "
            "Respond with JSON only, no markdown, no prose, no extra keys. "
            "Output keys exactly: cooling_setpoint, fan_speed, lighting, reason. "
            f"cooling_setpoint must be {self._limits.cooling_min} to {self._limits.cooling_max}. "
            f"fan_speed must be {self._limits.fan_min} to {self._limits.fan_max}. "
            "lighting must be ON or OFF. "
            "reason must be concise and <= 140 characters. "
            "Keep output under 150 tokens."
        )

    def build_user_prompt(
        self,
        snapshot: SensorSnapshot,
        previous_action: ControlAction | None,
        mcp_observations: dict[str, float | None] | None = None,
        validation_error: str | None = None,
    ) -> str:
        """Returns runtime control context and optional retry guidance."""
        payload = {
            "timestamp_utc": snapshot.timestamp_utc.isoformat(),
            "timestep_index": snapshot.timestep_index,
            "indoor_temperature": snapshot.indoor_temperature,
            "outdoor_temperature": snapshot.outdoor_temperature,
            "energy_consumption": snapshot.energy_consumption,
            "pmv_comfort": snapshot.pmv_comfort,
            "occupancy": snapshot.occupancy,
            "carbon_intensity": snapshot.carbon_intensity,
            "current_hvac_setpoint": snapshot.current_hvac_setpoint,
            "previous_action": {
                "cooling_setpoint": previous_action.cooling_setpoint,
                "fan_speed": previous_action.fan_speed,
                "lighting": previous_action.lighting,
            }
            if previous_action is not None
            else None,
            "mcp_observations": mcp_observations,
        }

        prompt = [
            "Generate the next control action as strict JSON.",
            "Input snapshot:",
            json.dumps(payload, separators=(",", ":")),
        ]

        if validation_error:
            prompt.append(f"Previous output was invalid: {validation_error}")
            prompt.append("Correct it and return valid JSON now.")

        return "\n".join(prompt)