"""MCP tool definitions backed by controller sensor and actuator layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Callable
import logging

from controller.actuators import ControlAction
from controller.actuators import ActuatorManager
from controller.sensors import SensorManager


class MCPToolHost(Protocol):
	"""Minimal host contract needed for registering FastMCP tools."""

	def tool(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
		"""Decorator that registers an MCP tool callable."""


@dataclass(frozen=True)
class ControllerToolContext:
	"""Runtime context required by MCP tools."""

	sensors: SensorManager
	actuators: ActuatorManager
	logger: logging.Logger


class ControllerToolRouter:
	"""In-process MCP-compatible tool router for controller data and actions."""

	def __init__(self, context: ControllerToolContext) -> None:
		self._context = context

	def get_temperature(self) -> float | None:
		return self._context.sensors.get_temperature()

	def get_outdoor_temperature(self) -> float | None:
		return self._context.sensors.get_outdoor_temperature()

	def get_energy_usage(self) -> float | None:
		return self._context.sensors.get_energy_usage()

	def get_current_setpoint(self) -> float | None:
		return self._context.sensors.get_current_setpoint()

	def get_occupancy(self) -> float | None:
		return self._context.sensors.get_occupancy()

	def get_comfort_index(self) -> float | None:
		return self._context.sensors.get_comfort_index()

	def get_carbon_intensity(self) -> float | None:
		return self._context.sensors.get_carbon_intensity()

	def set_cooling_setpoint(self, value: float) -> dict[str, Any]:
		ok = self._context.actuators.set_cooling_setpoint(value)
		response = {"success": ok, "requested": float(value)}
		self._context.logger.info("set_cooling_setpoint -> %s", response)
		return response

	def set_fan_speed(self, value: float) -> dict[str, Any]:
		ok = self._context.actuators.set_fan_speed(value)
		response = {"success": ok, "requested": float(value)}
		self._context.logger.info("set_fan_speed -> %s", response)
		return response

	def set_lighting(self, state: str) -> dict[str, Any]:
		normalized = state.strip().upper()
		ok = self._context.actuators.set_lighting(normalized)
		response = {"success": ok, "requested": normalized}
		self._context.logger.info("set_lighting -> %s", response)
		return response

	def get_sensor_payload(self) -> dict[str, float | None]:
		"""Returns all core control sensors as a single payload."""
		return {
			"indoor_temperature": self.get_temperature(),
			"outdoor_temperature": self.get_outdoor_temperature(),
			"energy_consumption": self.get_energy_usage(),
			"pmv_comfort": self.get_comfort_index(),
			"occupancy": self.get_occupancy(),
			"carbon_intensity": self.get_carbon_intensity(),
			"current_hvac_setpoint": self.get_current_setpoint(),
		}

	def call_tool(self, name: str, **kwargs: Any) -> Any:
		"""Calls any exposed MCP tool by name."""
		tools: dict[str, Callable[..., Any]] = {
			"get_temperature": self.get_temperature,
			"get_outdoor_temperature": self.get_outdoor_temperature,
			"get_energy_usage": self.get_energy_usage,
			"get_current_setpoint": self.get_current_setpoint,
			"get_occupancy": self.get_occupancy,
			"get_comfort_index": self.get_comfort_index,
			"get_carbon_intensity": self.get_carbon_intensity,
			"set_cooling_setpoint": self.set_cooling_setpoint,
			"set_fan_speed": self.set_fan_speed,
			"set_lighting": self.set_lighting,
		}
		if name not in tools:
			raise KeyError(f"Unknown MCP tool: {name}")
		return tools[name](**kwargs)

	def apply_action(self, action: ControlAction) -> bool:
		"""Applies a full control action through the actuator layer."""
		return self._context.actuators.apply(action)


class ControllerTools:
	"""Registers all required sensor and actuator MCP tools."""

	def __init__(self, mcp: MCPToolHost, context: ControllerToolContext) -> None:
		self._mcp = mcp
		self._context = context
		self._router = ControllerToolRouter(context)

	def register(self) -> None:
		"""Registers every tool required by the building control specification."""
		self._register_sensor_tools()
		self._register_actuator_tools()

	def _register_sensor_tools(self) -> None:
		@self._mcp.tool()
		def get_temperature() -> float | None:
			"""Returns indoor air temperature in degC."""
			value = self._router.get_temperature()
			self._context.logger.debug("get_temperature -> %s", value)
			return value

		@self._mcp.tool()
		def get_outdoor_temperature() -> float | None:
			"""Returns outdoor temperature in degC."""
			value = self._router.get_outdoor_temperature()
			self._context.logger.debug("get_outdoor_temperature -> %s", value)
			return value

		@self._mcp.tool()
		def get_energy_usage() -> float | None:
			"""Returns current building energy usage indicator."""
			value = self._router.get_energy_usage()
			self._context.logger.debug("get_energy_usage -> %s", value)
			return value

		@self._mcp.tool()
		def get_current_setpoint() -> float | None:
			"""Returns current HVAC cooling setpoint in degC."""
			value = self._router.get_current_setpoint()
			self._context.logger.debug("get_current_setpoint -> %s", value)
			return value

		@self._mcp.tool()
		def get_occupancy() -> float | None:
			"""Returns occupancy value from EnergyPlus."""
			value = self._router.get_occupancy()
			self._context.logger.debug("get_occupancy -> %s", value)
			return value

		@self._mcp.tool()
		def get_comfort_index() -> float | None:
			"""Returns PMV comfort index."""
			value = self._router.get_comfort_index()
			self._context.logger.debug("get_comfort_index -> %s", value)
			return value

		@self._mcp.tool()
		def get_carbon_intensity() -> float | None:
			"""Returns carbon intensity signal."""
			value = self._router.get_carbon_intensity()
			self._context.logger.debug("get_carbon_intensity -> %s", value)
			return value

	def _register_actuator_tools(self) -> None:
		@self._mcp.tool()
		def set_cooling_setpoint(value: float) -> dict[str, Any]:
			"""Sets HVAC cooling setpoint in degC."""
			return self._router.set_cooling_setpoint(value)

		@self._mcp.tool()
		def set_fan_speed(value: float) -> dict[str, Any]:
			"""Sets fan speed as percentage."""
			return self._router.set_fan_speed(value)

		@self._mcp.tool()
		def set_lighting(state: str) -> dict[str, Any]:
			"""Sets lighting ON or OFF."""
			return self._router.set_lighting(state)


def register_controller_tools(mcp: MCPToolHost, context: ControllerToolContext) -> None:
	"""Convenience entry point for registering all controller tools."""
	ControllerTools(mcp=mcp, context=context).register()


def build_controller_tool_router(context: ControllerToolContext) -> ControllerToolRouter:
	"""Builds an in-process MCP router for internal orchestration."""
	return ControllerToolRouter(context)
