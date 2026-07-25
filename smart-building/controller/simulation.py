"""EnergyPlus simulation runtime wrapper based on PyEnergyPlus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import logging
import os
import sys
import time


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
	energyplus_install_path: Path | None = None
	reconnect_attempts: int = 2
	reconnect_delay_seconds: float = 2.0


class EnergyPlusSimulation:
	"""Encapsulates PyEnergyPlus state and callback orchestration."""

	def __init__(
		self,
		config: SimulationConfig,
		logger: logging.Logger | None = None,
	) -> None:
		self._config = config
		self._logger = logger or logging.getLogger(self.__class__.__name__)

		self.api = _create_energyplus_api(config.energyplus_install_path)
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
			self._logger.debug("Actuator handle unavailable for %s", actuator_key)
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
		"""Runs EnergyPlus, reconnecting after an unexpected runtime failure."""
		args = self.build_energyplus_args()
		max_attempts = max(0, self._config.reconnect_attempts)

		for attempt in range(max_attempts + 1):
			self.runtime.callback_begin_zone_timestep_after_init_heat_balance(
				self.state,
				self._on_control_timestep,
			)
			self._is_running = True
			self._logger.info("Starting EnergyPlus with args: %s", args)
			try:
				return_code = int(self.runtime.run_energyplus(self.state, args))
			except Exception as exc:
				self._logger.exception("EnergyPlus runtime failure: %s", exc)
				return_code = 1

			if return_code == 0 or attempt >= max_attempts:
				self._is_running = False
				self._logger.info("EnergyPlus exited with code %s", return_code)
				return return_code

			self._logger.debug(
				"EnergyPlus disconnected (code %s); reconnecting (%s/%s)",
				return_code,
				attempt + 1,
				max_attempts,
			)
			self._is_running = False
			time.sleep(max(0.0, self._config.reconnect_delay_seconds))
			self._reset_state()

		return 1  # Unreachable, retained for static type checkers.

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
				self._logger.debug(
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
				self._logger.debug(
					"Invalid actuator handle for %s (%s / %s / %s)",
					key,
					spec.component_type,
					spec.control_type,
					spec.actuator_key,
				)

		self._handles_initialized = True

	def _reset_state(self) -> None:
		"""Creates a fresh EnergyPlus state and restores registered variables."""
		self.state = self.api.state_manager.new_state()
		self._variable_handles.clear()
		self._actuator_handles.clear()
		self._handles_initialized = False
		self.timestep_index = 0
		self._last_heartbeat_ts = 0.0
		for spec in self._variable_specs.values():
			self.exchange.request_variable(self.state, spec.variable_name, spec.variable_key)


def _create_energyplus_api(install_path: Path | None) -> Any:
	"""Loads PyEnergyPlus from an installed EnergyPlus distribution.

	EnergyPlus ships its Python API in ``python_lib`` rather than PyPI.  The
	configured installation path takes precedence, with ``ENERGYPLUS_INSTALL_PATH``
	as a portable deployment override.
	"""
	configured_path = install_path or _path_from_environment()
	if configured_path is not None:
		if not (configured_path / "pyenergyplus").exists():
			raise RuntimeError(
			f"EnergyPlus installation does not contain pyenergyplus: {configured_path}"
		)
		if str(configured_path) not in sys.path:
			sys.path.insert(0, str(configured_path))
		if hasattr(os, "add_dll_directory"):
			os.add_dll_directory(str(configured_path))

	try:
		from pyenergyplus.api import EnergyPlusAPI
	except ImportError as exc:
		hint = (
			"Configure simulation.energyplus.install_path or set "
			"ENERGYPLUS_INSTALL_PATH to the EnergyPlus installation directory."
		)
		raise RuntimeError(f"PyEnergyPlus API is unavailable. {hint}") from exc

	return EnergyPlusAPI()


def _path_from_environment() -> Path | None:
	"""Gets an optional EnergyPlus installation path from the environment."""
	value = os.environ.get("ENERGYPLUS_INSTALL_PATH")
	return Path(value) if value else None
