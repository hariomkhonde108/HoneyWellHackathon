"""EnergyPlus simulation runtime wrapper based on PyEnergyPlus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import logging
import time

from pyenergyplus.api import EnergyPlusAPI


ControlCallback = Callable[["EnergyPlusSimulation"], None]


@dataclass(frozen=True)
class VariableSpec:
	"""Describes an EnergyPlus output variable binding."""

	key: str
	variable_name: str
	variable_key: str


@dataclass(frozen=True)
class ActuatorSpec:
	"""Describes an EnergyPlus actuator binding."""

	key: str
	component_type: str
	control_type: str
	actuator_key: str


@dataclass(frozen=True)
class SimulationConfig:
	"""Configuration for running EnergyPlus in API mode."""

	idf_path: Path
	weather_path: Path
	output_dir: Path


class EnergyPlusSimulation:
	"""Encapsulates PyEnergyPlus state and callback orchestration."""

	def __init__(
		self,
		config: SimulationConfig,
		logger: logging.Logger | None = None,
	) -> None:
		self._config = config
		self._logger = logger or logging.getLogger(self.__class__.__name__)

		self.api = EnergyPlusAPI()
		self.state = self.api.state_manager.new_state()
		self.exchange = self.api.exchange
		self.runtime = self.api.runtime

		self._variable_specs: dict[str, VariableSpec] = {}
		self._actuator_specs: dict[str, ActuatorSpec] = {}
		self._variable_handles: dict[str, int] = {}
		self._actuator_handles: dict[str, int] = {}
		self._control_callbacks: list[ControlCallback] = []

		self._handles_initialized = False
		self._is_running = False
		self._last_heartbeat_ts = 0.0
		self.timestep_index = 0

	@property
	def is_running(self) -> bool:
		"""Returns true while a simulation run is active."""
		return self._is_running

	@property
	def last_heartbeat_ts(self) -> float:
		"""Returns the unix timestamp of the last control callback."""
		return self._last_heartbeat_ts

	def register_variable(self, spec: VariableSpec) -> None:
		"""Registers and requests a variable for read access."""
		self._variable_specs[spec.key] = spec
		self.exchange.request_variable(self.state, spec.variable_name, spec.variable_key)
		self._logger.debug("Registered variable %s", spec)

	def register_actuator(self, spec: ActuatorSpec) -> None:
		"""Registers an actuator for write access."""
		self._actuator_specs[spec.key] = spec
		self._logger.debug("Registered actuator %s", spec)

	def register_control_callback(self, callback: ControlCallback) -> None:
		"""Registers a controller callback executed on each timestep event."""
		self._control_callbacks.append(callback)

	def get_variable_value(self, variable_key: str) -> float | None:
		"""Reads the current value of a previously registered variable."""
		handle = self._variable_handles.get(variable_key)
		if handle is None or handle < 0:
			return None
		value = self.exchange.get_variable_value(self.state, handle)
		return float(value)

	def set_actuator_value(self, actuator_key: str, value: float) -> bool:
		"""Writes a value to a previously registered actuator."""
		handle = self._actuator_handles.get(actuator_key)
		if handle is None or handle < 0:
			self._logger.warning("Actuator handle unavailable for %s", actuator_key)
			return False
		self.exchange.set_actuator_value(self.state, handle, float(value))
		return True

	def build_energyplus_args(self) -> list[str]:
		"""Builds command-line arguments for the EnergyPlus runtime call."""
		output_dir = self._config.output_dir
		output_dir.mkdir(parents=True, exist_ok=True)
		return [
			"-w",
			str(self._config.weather_path),
			"-d",
			str(output_dir),
			str(self._config.idf_path),
		]

	def run(self) -> int:
		"""Runs EnergyPlus and blocks until completion."""
		self.runtime.callback_begin_zone_timestep_after_init_heat_balance(
			self.state,
			self._on_control_timestep,
		)
		args = self.build_energyplus_args()
		self._is_running = True
		self._logger.info("Starting EnergyPlus with args: %s", args)
		try:
			return_code = self.runtime.run_energyplus(self.state, args)
			self._logger.info("EnergyPlus exited with code %s", return_code)
			return int(return_code)
		finally:
			self._is_running = False

	def stop(self) -> None:
		"""Requests simulation stop through the runtime API."""
		if self._is_running:
			self.runtime.stop_simulation(self.state)
			self._logger.info("Stop signal sent to EnergyPlus")

	def is_healthy(self, max_silence_seconds: float = 30.0) -> bool:
		"""Checks whether callbacks are still arriving within the expected window."""
		if not self._is_running:
			return False
		if self._last_heartbeat_ts == 0.0:
			return True
		return (time.time() - self._last_heartbeat_ts) <= max_silence_seconds

	def _on_control_timestep(self, state_argument: int) -> None:
		"""Internal callback for each zone timestep after initialization."""
		del state_argument
		self._last_heartbeat_ts = time.time()

		if not self.exchange.api_data_fully_ready(self.state):
			return

		if not self._handles_initialized:
			self._initialize_handles()

		self.timestep_index += 1
		for callback in self._control_callbacks:
			callback(self)

	def _initialize_handles(self) -> None:
		"""Initializes all variable and actuator handles once data is ready."""
		for key, spec in self._variable_specs.items():
			handle = self.exchange.get_variable_handle(
				self.state,
				spec.variable_name,
				spec.variable_key,
			)
			self._variable_handles[key] = int(handle)
			if handle < 0:
				self._logger.warning(
					"Invalid variable handle for %s (%s / %s)",
					key,
					spec.variable_name,
					spec.variable_key,
				)

		for key, spec in self._actuator_specs.items():
			handle = self.exchange.get_actuator_handle(
				self.state,
				spec.component_type,
				spec.control_type,
				spec.actuator_key,
			)
			self._actuator_handles[key] = int(handle)
			if handle < 0:
				self._logger.warning(
					"Invalid actuator handle for %s (%s / %s / %s)",
					key,
					spec.component_type,
					spec.control_type,
					spec.actuator_key,
				)

		self._handles_initialized = True
