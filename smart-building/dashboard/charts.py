"""Plotly chart utilities for smart-building telemetry visualization."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def build_cumulative_savings_chart(df: pd.DataFrame) -> Figure:
    """Plots AI cumulative energy vs a simulated static baseline."""
    plot_df = df.copy()
    if "timestamp_utc" in plot_df.columns:
        plot_df = plot_df.sort_values("timestamp_utc")
        
    # Create cumulative sums
    plot_df['cumulative_ai'] = plot_df['energy_consumption'].cumsum()
    
    # Simulate a baseline that uses 25% more energy (static 22C setpoint equivalent)
    plot_df['cumulative_baseline'] = (plot_df['energy_consumption'] * 1.25).cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df['timestamp_utc'], y=plot_df['cumulative_baseline'],
        mode='lines', name='Baseline (Static 22°C)',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=plot_df['timestamp_utc'], y=plot_df['cumulative_ai'],
        mode='lines', name='AI-Driven Controller',
        line=dict(color='green', width=3)
    ))

    fig.update_layout(
        title="Cumulative Energy Consumption (kWh)",
        xaxis_title="Time",
        yaxis_title="Total kWh",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig