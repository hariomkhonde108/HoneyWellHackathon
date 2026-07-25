"""Real-time Streamlit dashboard for autonomous building optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import yaml
from streamlit_autorefresh import st_autorefresh

try:
	from dashboard.charts import (
		build_carbon_chart,
		build_comfort_chart,
		build_energy_chart,
		build_temperature_chart,
	)
except ImportError:
	from charts import (
		build_carbon_chart,
		build_comfort_chart,
		build_energy_chart,
		build_temperature_chart,
	)


def main() -> None:
	"""Renders auto-refreshing dashboard from control-cycle telemetry."""
	st.set_page_config(page_title="Smart Building Optimizer", page_icon="", layout="wide")

	project_root = Path(__file__).resolve().parents[1]
	settings = _load_settings(project_root / "config/settings.yaml")

	refresh_seconds = int(settings.get("dashboard", {}).get("refresh_seconds", 2))
	st_autorefresh(interval=refresh_seconds * 1000, key="telemetry_autorefresh")

	telemetry_path = _resolve_telemetry_path(project_root, settings)
	dataframe = _load_telemetry(telemetry_path)

	st.title("AI-Powered Autonomous Smart Building Dashboard")
	st.caption(f"Auto-refresh: every {refresh_seconds}s | Source: {telemetry_path}")

	if dataframe.empty:
		st.warning("No telemetry available yet. Start the controller to stream simulation data.")
		return

	latest = dataframe.iloc[-1]
	_render_metrics(latest)
	_render_charts(dataframe)
	_render_history_table(dataframe)


def _load_settings(config_path: Path) -> dict[str, Any]:
	with config_path.open("r", encoding="utf-8") as config_file:
		loaded = yaml.safe_load(config_file)
	if not isinstance(loaded, dict):
		raise ValueError("settings.yaml must contain a top-level object")
	return loaded


def _resolve_telemetry_path(project_root: Path, settings: dict[str, Any]) -> Path:
	telemetry_settings = settings.get("telemetry", {})
	output_dir = project_root / str(telemetry_settings.get("output_dir", "logs"))
	prefix = str(telemetry_settings.get("file_prefix", "telemetry"))
	return output_dir / f"{prefix}_metrics.csv"


def _load_telemetry(path: Path) -> pd.DataFrame:
	if not path.exists():
		return pd.DataFrame()

	dataframe = pd.read_csv(path)
	if dataframe.empty:
		return dataframe

	if "timestamp_utc" in dataframe.columns:
		dataframe["timestamp_utc"] = pd.to_datetime(dataframe["timestamp_utc"], errors="coerce")
		dataframe = dataframe.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")

	numeric_columns = [
		"indoor_temperature",
		"outdoor_temperature",
		"energy_consumption",
		"pmv_comfort",
		"occupancy",
		"carbon_intensity",
		"current_hvac_setpoint",
		"cooling_setpoint",
		"fan_speed",
		"savings_pct",
	]
	for column in numeric_columns:
		if column in dataframe.columns:
			dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

	return dataframe


def _render_metrics(latest: pd.Series) -> None:
	first_row = st.columns(5)
	second_row = st.columns(5)

	first_row[0].metric("Current Temperature (degC)", _fmt(latest.get("indoor_temperature")))
	first_row[1].metric("Target Temperature (degC)", _fmt(latest.get("current_hvac_setpoint")))
	first_row[2].metric("Occupancy", _fmt(latest.get("occupancy"), decimals=0))
	first_row[3].metric("Energy Usage", _fmt(latest.get("energy_consumption")))
	first_row[4].metric("Comfort Index (PMV)", _fmt(latest.get("pmv_comfort")))

	second_row[0].metric("Carbon Emissions", _fmt(latest.get("carbon_intensity")))
	second_row[1].metric("Cooling Setpoint (degC)", _fmt(latest.get("cooling_setpoint")))
	second_row[2].metric("Fan Speed (%)", _fmt(latest.get("fan_speed")))
	second_row[3].metric("Savings (%)", _fmt(latest.get("savings_pct")))
	second_row[4].metric("Lighting", str(latest.get("lighting", "N/A")))

	ai_reason = latest.get("ai_reason", "")
	st.markdown("### AI Reason")
	st.write(str(ai_reason) if ai_reason else "No AI rationale available for latest cycle.")


def _render_charts(dataframe: pd.DataFrame) -> None:
	col_left, col_right = st.columns(2)
	with col_left:
		st.plotly_chart(build_temperature_chart(dataframe), use_container_width=True)
		st.plotly_chart(build_comfort_chart(dataframe), use_container_width=True)
	with col_right:
		st.plotly_chart(build_energy_chart(dataframe), use_container_width=True)
		st.plotly_chart(build_carbon_chart(dataframe), use_container_width=True)


def _render_history_table(dataframe: pd.DataFrame) -> None:
	st.markdown("### Recent Cycles")
	columns = [
		"timestamp_utc",
		"timestep_index",
		"indoor_temperature",
		"outdoor_temperature",
		"energy_consumption",
		"pmv_comfort",
		"occupancy",
		"carbon_intensity",
		"cooling_setpoint",
		"fan_speed",
		"lighting",
		"savings_pct",
		"status",
	]
	available_columns = [column for column in columns if column in dataframe.columns]
	st.dataframe(dataframe[available_columns].tail(25), use_container_width=True)


def _fmt(value: Any, decimals: int = 2) -> str:
	if value is None or pd.isna(value):
		return "N/A"
	return f"{float(value):.{decimals}f}"


if __name__ == "__main__":
	main()
