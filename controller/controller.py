"""Autonomous control orchestration over EnergyPlus timesteps."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import logging

from controller.actuators import (
    ActuatorManager,
    ControlAction,
    SafetyLimits,
    default_actuator_bindings,
)
from controller.sensors import SensorManager, SensorSnapshot, default_sensor_bindings
from controller.simulation import EnergyPlusSimulation, SimulationConfig


class DecisionEngine(Protocol):
    """Contract for external decision providers such as an LLM adapter."""

    def __call__(
        self,
        snapshot: SensorSnapshot,
        previous_action: ControlAction | None,
    ) -> ControlAction:
        """Returns the control action for a snapshot."""


@dataclass(frozen=True)
class ControllerConfig:
    """Configuration for orchestrating control decisions."""

    control_interval_steps: int = 5


class AutonomousController:
    """Coordinates read -> decide -> act loop on EnergyPlus callbacks."""

    def __init__(
        self,
        simulation: EnergyPlusSimulation,
        sensors: SensorManager,
        actuators: ActuatorManager,
        decision_engine: DecisionEngine,
        config: ControllerConfig | None = None,
        telemetry_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._simulation = simulation
        self._sensors = sensors
        self._actuators = actuators
        self._decision_engine = decision_engine
        self._config = config or ControllerConfig()
        self._telemetry_path = telemetry_path
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._baseline_energy: float | None = None

        if self._telemetry_path is not None:
            self._telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            if not self._telemetry_path.exists():
                self._initialize_telemetry_file()

        if self._config.control_interval_steps <= 0:
            raise ValueError("control_interval_steps must be >= 1")

    def start(self) -> int:
        """Starts simulation and autonomous control until EnergyPlus exits."""
        self._simulation.register_control_callback(self._on_control_timestep)
        return self._simulation.run()

    def stop(self) -> None:
        """Stops the underlying simulation runtime."""
        self._simulation.stop()

    @property
    def sensors(self) -> SensorManager:
        """Returns the sensor manager used by this controller."""
        return self._sensors

    @property
    def actuators(self) -> ActuatorManager:
        """Returns the actuator manager used by this controller."""
        return self._actuators

    def _on_control_timestep(self, simulation: EnergyPlusSimulation) -> None:
        snapshot = self._sensors.read_snapshot()
        if simulation.timestep_index % self._config.control_interval_steps != 0:
            self._write_telemetry(snapshot=snapshot, action=self._actuators.last_action, status="observe")
            return

        # --- HACKATHON FIX: Soft warning instead of hard abort ---
        if snapshot.has_missing_required:
            self._logger.warning(
                "Missing required sensors: %s. Forcing AI decision anyway to maintain closed loop.",
                snapshot.missing_sensors,
            )
            # The loop will now continue directly to _resolve_action instead of returning!

        action = self._resolve_action(snapshot)
        success = self._actuators.apply(action)
        if success:
            self._logger.info(
                "Applied action at timestep %s: setpoint=%.2f, fan=%.2f, lighting=%s",
                snapshot.timestep_index,
                action.cooling_setpoint,
                action.fan_speed,
                action.lighting,
            )
            self._write_telemetry(snapshot=snapshot, action=action, status="applied")
        else:
            self._logger.error("Failed to apply control action at timestep %s", snapshot.timestep_index)
            self._write_telemetry(snapshot=snapshot, action=action, status="failed")

    def _resolve_action(self, snapshot: SensorSnapshot) -> ControlAction:
        previous_action = self._actuators.last_action

        try:
            return self._decision_engine(snapshot, previous_action)
        except TimeoutError:
            self._logger.warning("Decision engine timed out; using previous action fallback")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._logger.exception("Decision engine failed: %s", exc)

        if previous_action is not None:
            return previous_action

        baseline_setpoint = snapshot.current_hvac_setpoint if snapshot.current_hvac_setpoint is not None else 24.0
        return ControlAction(
            cooling_setpoint=baseline_setpoint,
            fan_speed=50.0,
            lighting="ON" if (snapshot.occupancy or 0.0) > 0 else "OFF",
            reason="Fallback action due to unavailable previous action or decision timeout.",
        )

    def _initialize_telemetry_file(self) -> None:
        """Creates telemetry CSV with a stable schema for dashboard ingestion."""
        assert self._telemetry_path is not None
        with self._telemetry_path.open("w", newline="", encoding="utf-8") as telemetry_file:
            writer = csv.DictWriter(telemetry_file, fieldnames=self._telemetry_columns())
            writer.writeheader()

    def _write_telemetry(
        self,
        snapshot: SensorSnapshot,
        action: ControlAction | None,
        status: str,
    ) -> None:
        """Appends one control-cycle record for the Streamlit dashboard."""
        if self._telemetry_path is None:
            return

        if self._baseline_energy is None and snapshot.energy_consumption is not None:
            self._baseline_energy = snapshot.energy_consumption

        energy = snapshot.energy_consumption
        savings_pct = None
        if self._baseline_energy is not None and energy is not None and self._baseline_energy > 0:
            savings_pct = ((self._baseline_energy - energy) / self._baseline_energy) * 100.0

        row = {
            "timestamp_utc": snapshot.timestamp_utc.isoformat(),
            "timestep_index": snapshot.timestep_index,
            "indoor_temperature": snapshot.indoor_temperature,
            "outdoor_temperature": snapshot.outdoor_temperature,
            "energy_consumption": snapshot.energy_consumption,
            "pmv_comfort": snapshot.pmv_comfort,
            "occupancy": snapshot.occupancy,
            "carbon_intensity": snapshot.carbon_intensity,
            "current_hvac_setpoint": snapshot.current_hvac_setpoint,
            "cooling_setpoint": action.cooling_setpoint if action else None,
            "fan_speed": action.fan_speed if action else None,
            "lighting": action.lighting if action else None,
            "ai_reason": action.reason if action else "No action available for this cycle.",
            "savings_pct": savings_pct,
            "status": status,
        }

        with self._telemetry_path.open("a", newline="", encoding="utf-8") as telemetry_file:
            writer = csv.DictWriter(telemetry_file, fieldnames=self._telemetry_columns())
            writer.writerow(row)

    @staticmethod
    def _telemetry_columns() -> list[str]:
        return [
            "timestamp_utc",
            "timestep_index",
            "indoor_temperature",
            "outdoor_temperature",
            "energy_consumption",
            "pmv_comfort",
            "occupancy",
            "carbon_intensity",
            "current_hvac_setpoint",
            "cooling_setpoint",
            "fan_speed",
            "lighting",
            "ai_reason",
            "savings_pct",
            "status",
        ]


def build_default_controller(
    idf_path: Path,
    weather_path: Path,
    output_dir: Path,
    decision_engine: DecisionEngine,
    control_interval_steps: int = 5,
    zone_name: str = "SPACE1-1",
    energyplus_install_path: Path | None = None,
    telemetry_path: Path | None = None,
    safety_limits: SafetyLimits | None = None,
    max_setpoint_delta: float | None = None,
    max_fan_delta: float | None = None,
    fan_max_air_mass_flow_kg_s: float = 3.0,
    reconnect_attempts: int = 2,
    reconnect_delay_seconds: float = 2.0,
) -> AutonomousController:
    """Factory that wires simulation, sensors, and actuators with defaults."""
    simulation = EnergyPlusSimulation(
        SimulationConfig(
            idf_path=idf_path,
            weather_path=weather_path,
            output_dir=output_dir,
            energyplus_install_path=energyplus_install_path,
            reconnect_attempts=reconnect_attempts,
            reconnect_delay_seconds=reconnect_delay_seconds,
        )
    )
    sensors = SensorManager(simulation=simulation, bindings=default_sensor_bindings(zone_name=zone_name))
    actuators = ActuatorManager(
        simulation=simulation,
        bindings=default_actuator_bindings(zone_name=zone_name),
        limits=safety_limits or SafetyLimits(),
        max_setpoint_delta=max_setpoint_delta,
        max_fan_delta=max_fan_delta,
        fan_max_air_mass_flow_kg_s=fan_max_air_mass_flow_kg_s,
    )
    return AutonomousController(
        simulation=simulation,
        sensors=sensors,
        actuators=actuators,
        decision_engine=decision_engine,
        config=ControllerConfig(control_interval_steps=control_interval_steps),
        telemetry_path=telemetry_path or (output_dir / "telemetry_metrics.csv"),
    )