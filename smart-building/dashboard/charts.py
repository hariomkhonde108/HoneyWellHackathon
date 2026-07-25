"""Plotly chart utilities for smart-building telemetry visualization."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def build_temperature_chart(dataframe: pd.DataFrame) -> Figure:
	"""Builds indoor and outdoor temperature trends over time."""
	plot_df = dataframe.copy()
	if "timestamp_utc" in plot_df.columns:
		plot_df = plot_df.sort_values("timestamp_utc")

	return px.line(
		plot_df,
		x="timestamp_utc",
		y=["indoor_temperature", "outdoor_temperature", "cooling_setpoint"],
		title="Temperature vs Time",
		labels={"value": "Temperature (degC)", "timestamp_utc": "Time"},
	)


def build_energy_chart(dataframe: pd.DataFrame) -> Figure:
	"""Builds energy usage trend over time."""
	plot_df = dataframe.copy()
	if "timestamp_utc" in plot_df.columns:
		plot_df = plot_df.sort_values("timestamp_utc")

	return px.line(
		plot_df,
		x="timestamp_utc",
		y="energy_consumption",
		title="Energy vs Time",
		labels={"energy_consumption": "Energy Usage", "timestamp_utc": "Time"},
	)


def build_comfort_chart(dataframe: pd.DataFrame) -> Figure:
	"""Builds comfort (PMV) trend over time."""
	plot_df = dataframe.copy()
	if "timestamp_utc" in plot_df.columns:
		plot_df = plot_df.sort_values("timestamp_utc")

	return px.line(
		plot_df,
		x="timestamp_utc",
		y="pmv_comfort",
		title="Comfort vs Time",
		labels={"pmv_comfort": "PMV Comfort Index", "timestamp_utc": "Time"},
	)


def build_carbon_chart(dataframe: pd.DataFrame) -> Figure:
	"""Builds carbon intensity trend over time."""
	plot_df = dataframe.copy()
	if "timestamp_utc" in plot_df.columns:
		plot_df = plot_df.sort_values("timestamp_utc")

	return px.line(
		plot_df,
		x="timestamp_utc",
		y="carbon_intensity",
		title="Carbon vs Time",
		labels={"carbon_intensity": "Carbon Emissions", "timestamp_utc": "Time"},
	)
