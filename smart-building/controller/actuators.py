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
	) -> None:
		self._simulation = simulation
		self._limits = limits or SafetyLimits()
		self._bindings: dict[str, ActuatorBinding] = {binding.key: binding for binding in bindings}
		self._last_action: ControlAction | None = None
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
		"""Sets fan speed percentage after clamping."""
		safe_value = self._clamp(value, self._limits.fan_min, self._limits.fan_max)
		return self._simulation.set_actuator_value("fan_speed", safe_value)

	def set_lighting(self, state: str) -> bool:
		"""Sets lighting command using ON/OFF mapping."""
		normalized = state.strip().upper()
		if normalized not in {"ON", "OFF"}:
			return False
		schedule_value = 1.0 if normalized == "ON" else 0.0
		return self._simulation.set_actuator_value("lighting", schedule_value)

	def apply(self, action: ControlAction) -> bool:
		"""Applies a full action atomically from a controller perspective."""
		cooling_ok = self.set_cooling_setpoint(action.cooling_setpoint)
		fan_ok = self.set_fan_speed(action.fan_speed)
		light_ok = self.set_lighting(action.lighting)

		success = cooling_ok and fan_ok and light_ok
		if success:
			self._last_action = action
		return success

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
			component_type="Fan",
			control_type="Fan Air Mass Flow Rate",
			actuator_key="SUPPLY FAN",
		),
		ActuatorBinding(
			key="lighting",
			component_type="Schedule:Constant",
			control_type="Schedule Value",
			actuator_key="LIGHTING_ON_OFF",
		),
	]
