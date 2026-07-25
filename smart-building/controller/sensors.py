"""Sensor abstraction for EnergyPlus variables."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from controller.simulation import EnergyPlusSimulation, VariableSpec


@dataclass(frozen=True)
class SensorBinding:
    """Maps a logical sensor name to an EnergyPlus variable."""

    key: str
    variable_name: str
    variable_key: str
    required: bool = True


@dataclass
class SensorSnapshot:
    """A typed snapshot of the main control signals for one cycle."""

    timestamp_utc: datetime
    timestep_index: int
    indoor_temperature: float | None
    outdoor_temperature: float | None
    energy_consumption: float | None
    pmv_comfort: float | None
    occupancy: float | None
    carbon_intensity: float | None
    current_hvac_setpoint: float | None
    missing_sensors: list[str] = field(default_factory=list)

    @property
    def has_missing_required(self) -> bool:
        """True when one or more required sensors are unavailable."""
        return bool(self.missing_sensors)


class SensorManager:
    """Registers and reads control-relevant variables from EnergyPlus."""

    def __init__(
        self,
        simulation: EnergyPlusSimulation,
        bindings: Iterable[SensorBinding],
    ) -> None:
        self._simulation = simulation
        self._bindings: dict[str, SensorBinding] = {b.key: b for b in bindings}
        self._register_bindings()

    def get_temperature(self) -> float | None:
        """Returns indoor air temperature in degC."""
        return self._read("indoor_temperature")

    def get_outdoor_temperature(self) -> float | None:
        """Returns outdoor drybulb temperature in degC."""
        return self._read("outdoor_temperature")

    def get_energy_usage(self) -> float | None:
        """Returns building-level HVAC energy/power indicator."""
        return self._read("energy_consumption")

    def get_occupancy(self) -> float | None:
        """Returns occupancy sensor value."""
        return self._read("occupancy")

    def get_comfort_index(self) -> float | None:
        """Returns PMV comfort index."""
        return self._read("pmv_comfort")

    def get_carbon_intensity(self) -> float | None:
        """Returns carbon intensity signal."""
        return self._read("carbon_intensity")

    def get_current_setpoint(self) -> float | None:
        """Returns current zone thermostat setpoint."""
        return self._read("current_hvac_setpoint")

    def read_snapshot(self) -> SensorSnapshot:
        """Builds a complete snapshot for one controller decision cycle."""
        missing_required: list[str] = []
        values: dict[str, float | None] = {}

        for key, binding in self._bindings.items():
            value = self._simulation.get_variable_value(key)
            values[key] = value
            if value is None and binding.required:
                missing_required.append(key)

        return SensorSnapshot(
            timestamp_utc=datetime.now(timezone.utc),
            timestep_index=self._simulation.timestep_index,
            indoor_temperature=values.get("indoor_temperature"),
            outdoor_temperature=values.get("outdoor_temperature"),
            energy_consumption=values.get("energy_consumption"),
            pmv_comfort=values.get("pmv_comfort"),
            occupancy=values.get("occupancy"),
            carbon_intensity=values.get("carbon_intensity"),
            current_hvac_setpoint=values.get("current_hvac_setpoint"),
            missing_sensors=missing_required,
        )

    def _register_bindings(self) -> None:
        for binding in self._bindings.values():
            self._simulation.register_variable(
                VariableSpec(
                    key=binding.key,
                    variable_name=binding.variable_name,
                    variable_key=binding.variable_key,
                )
            )

    def _read(self, key: str) -> float | None:
        return self._simulation.get_variable_value(key)


def default_sensor_bindings(zone_name: str = "ZONE ONE") -> list[SensorBinding]:
    """Returns a baseline sensor map for quick startup."""
    return [
        SensorBinding(
            key="indoor_temperature",
            variable_name="Zone Mean Air Temperature",
            variable_key=zone_name,
        ),
        SensorBinding(
            key="outdoor_temperature",
            variable_name="Site Outdoor Air Drybulb Temperature",
            variable_key="Environment",
        ),
        SensorBinding(
            key="energy_consumption",
            # MATCHES IDF: Output:Variable, *, Facility Total Electricity Demand Rate, Timestep;
            variable_name="Facility Total Electricity Demand Rate",
            variable_key="Whole Building",
        ),
        SensorBinding(
            key="pmv_comfort",
            variable_name="Zone Thermal Comfort Fanger Model PMV",
            # PMV attaches to the People object inside the zone, not the zone itself
            variable_key=f"{zone_name} People 1", 
            required=False, # Optional fallback applied
        ),
        SensorBinding(
            key="occupancy",
            # MATCHES IDF: Output:Variable, *, People Occupant Count, Timestep;
            variable_name="People Occupant Count",
            variable_key=zone_name,
            required=False, # Optional fallback applied
        ),
        SensorBinding(
            key="carbon_intensity",
            variable_name="Facility Net Purchased Electricity Carbon Emissions Mass Rate",
            variable_key="Whole Building",
            required=False,
        ),
        SensorBinding(
            key="current_hvac_setpoint",
            variable_name="Zone Thermostat Cooling Setpoint Temperature",
            variable_key=zone_name,
        ),
    ]