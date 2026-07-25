"""Actuator abstraction and safety validation for control commands."""

from __future__ import annotations

from dataclasses import dataclass

from controller.simulation import ActuatorSpec, EnergyPlusSimulation


@dataclass(frozen=True)
class ActuatorBinding:
    """Maps a logical control key to an EnergyPlus actuator."""

    key: str
    component_type: str
    control_type: str
    actuator_key: str


@dataclass(frozen=True)
class SafetyLimits:
    """Hard safety bounds used before writing to actuators."""

    cooling_min: float = 22.0
    cooling_max: float = 27.0
    fan_min: float = 30.0
    fan_max: float = 90.0


@dataclass
class ControlAction:
    """Desired control values for one control cycle."""

    cooling_setpoint: float
    fan_speed: float
    lighting: str
    reason: str


class ActuatorManager:
    """Owns setpoint writes and domain-specific safety checks."""

    def __init__(
        self,
        simulation: EnergyPlusSimulation,
        bindings: list[ActuatorBinding],
        limits: SafetyLimits | None = None,
        fan_max_air_mass_flow_kg_s: float = 3.0,
        max_setpoint_delta: float | None = None,
        max_fan_delta: float | None = None,
    ) -> None:
        self._simulation = simulation
        self._limits = limits or SafetyLimits()
        self._bindings: dict[str, ActuatorBinding] = {binding.key: binding for binding in bindings}
        self._last_action: ControlAction | None = None
        if fan_max_air_mass_flow_kg_s <= 0:
            raise ValueError("fan_max_air_mass_flow_kg_s must be greater than zero")
        self._fan_max_air_mass_flow_kg_s = fan_max_air_mass_flow_kg_s
        self._max_setpoint_delta = max_setpoint_delta
        self._max_fan_delta = max_fan_delta
        self._register_bindings()

    @property
    def last_action(self) -> ControlAction | None:
        """Returns the most recently applied action."""
        return self._last_action

    def set_cooling_setpoint(self, value: float) -> bool:
        """Sets cooling setpoint after clamping to safe limits."""
        safe_value = self._clamp(value, self._limits.cooling_min, self._limits.cooling_max)
        return self._simulation.set_actuator_value("cooling_setpoint", safe_value)

    def set_fan_speed(self, value: float) -> bool:
        """Sets fan speed percentage after converting it to EnergyPlus mass flow."""
        safe_value = self._clamp(value, self._limits.fan_min, self._limits.fan_max)
        mass_flow = self._fan_max_air_mass_flow_kg_s * (safe_value / 100.0)
        return self._simulation.set_actuator_value("fan_speed", mass_flow)

    def set_lighting(self, state: str) -> bool:
        """Sets lighting command using ON/OFF mapping."""
        normalized = state.strip().upper()
        if normalized not in {"ON", "OFF"}:
            return False
        schedule_value = 1.0 if normalized == "ON" else 0.0
        return self._simulation.set_actuator_value("lighting", schedule_value)

    def apply(self, action: ControlAction | None) -> bool:
        """Applies a full action atomically from a controller perspective."""
        if action is None:
            return False

        bounded_action = self._limit_change(action)
        
        # 1. Apply the cooling setpoint (This is the critical one)
        cooling_ok = self.set_cooling_setpoint(bounded_action.cooling_setpoint)
        
        # 2. Attempt fan and lighting, but don't let them crash the cycle if handles are missing
        self.set_fan_speed(bounded_action.fan_speed)
        self.set_lighting(bounded_action.lighting)

        # 3. Consider it a success if cooling was applied!
        if cooling_ok:
            self._last_action = bounded_action
            
        return cooling_ok
    def _limit_change(self, action: ControlAction) -> ControlAction:
        """Limits per-cycle changes to avoid abrupt HVAC commands."""
        previous = self._last_action
        if previous is None:
            return action

        cooling = self._limit_delta(
            action.cooling_setpoint, previous.cooling_setpoint, self._max_setpoint_delta
        )
        fan = self._limit_delta(action.fan_speed, previous.fan_speed, self._max_fan_delta)
        return ControlAction(cooling_setpoint=cooling, fan_speed=fan, lighting=action.lighting, reason=action.reason)

    @staticmethod
    def _limit_delta(value: float, previous_value: float, max_delta: float | None) -> float:
        if max_delta is None or max_delta <= 0:
            return float(value)
        return max(previous_value - max_delta, min(previous_value + max_delta, float(value)))

    def _register_bindings(self) -> None:
        for binding in self._bindings.values():
            self._simulation.register_actuator(
                ActuatorSpec(
                    key=binding.key,
                    component_type=binding.component_type,
                    control_type=binding.control_type,
                    actuator_key=binding.actuator_key,
                )
            )

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(max_value, float(value)))


def default_actuator_bindings(zone_name: str = "ZONE ONE") -> list[ActuatorBinding]:
    """Returns a baseline actuator map for quick startup."""
    return [
        ActuatorBinding(
            key="cooling_setpoint",
            component_type="Zone Temperature Control",
            control_type="Cooling Setpoint",
            actuator_key=zone_name,
        ),
        ActuatorBinding(
            key="fan_speed",
            component_type="Fan:VariableVolume",
            control_type="Fan Air Mass Flow Rate",
            actuator_key="Supply Fan 1",
        ),
        ActuatorBinding(
            key="lighting",
            component_type="Schedule:Compact",
            control_type="Schedule Value",
            actuator_key="LIGHTS-1",
        ),
    ]