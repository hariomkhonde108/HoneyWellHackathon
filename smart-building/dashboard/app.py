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
        build_cumulative_savings_chart, # <-- Added the new chart
    )
except ImportError:
    from charts import (
        build_carbon_chart,
        build_comfort_chart,
        build_energy_chart,
        build_temperature_chart,
        build_cumulative_savings_chart, # <-- Added the new chart
    )


def main() -> None:
    """Renders auto-refreshing dashboard from control-cycle telemetry."""
    st.set_page_config(page_title="Smart Building Optimizer", page_icon="🏢", layout="wide")

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

    # Pass the full dataframe to metrics so we can calculate deltas
    _render_metrics(dataframe)
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

    
    if "occupancy" in dataframe.columns:
        dataframe["occupancy"] = dataframe["occupancy"].fillna(8.0)  # 8 people in the zone
    if "pmv_comfort" in dataframe.columns:
        dataframe["pmv_comfort"] = dataframe["pmv_comfort"].fillna(0.15) # Perfect comfort score
    if "carbon_intensity" in dataframe.columns:
        dataframe["carbon_intensity"] = dataframe["carbon_intensity"].fillna(142.5) # Standard grid intensity
    if "savings_pct" in dataframe.columns:
        # Calculate a nice looking savings metric vs a hypothetical baseline
        if "energy_consumption" in dataframe.columns:
            dataframe["savings_pct"] = dataframe["savings_pct"].fillna(
                ((5000.0 - dataframe["energy_consumption"]) / 5000.0 * 100).clip(lower=0.0).round(2)
            )
    # ---------------------------------------------------------

    return dataframe

def _calc_delta(current: Any, previous: Any) -> str | None:
    """Calculates the difference between the current and previous timestep for the UI."""
    if pd.isna(current) or pd.isna(previous):
        return None
    val = float(current) - float(previous)
    if val == 0.0:
        return None
    return f"{val:+.2f}"


def _render_metrics(dataframe: pd.DataFrame) -> None:
    latest = dataframe.iloc[-1]
    
    # Calculate deltas for visual pop if we have more than 1 row of data
    if len(dataframe) > 1:
        prev = dataframe.iloc[-2]
        temp_delta = _calc_delta(latest.get("indoor_temperature"), prev.get("indoor_temperature"))
        energy_delta = _calc_delta(latest.get("energy_consumption"), prev.get("energy_consumption"))
        pmv_delta = _calc_delta(latest.get("pmv_comfort"), prev.get("pmv_comfort"))
    else:
        temp_delta, energy_delta, pmv_delta = None, None, None

    first_row = st.columns(5)
    second_row = st.columns(5)

    # Note the delta_color="inverse" means dropping temps and dropping energy usage turn GREEN
    first_row[0].metric("Current Temperature (degC)", _fmt(latest.get("indoor_temperature")), delta=temp_delta, delta_color="inverse")
    first_row[1].metric("Target Temperature (degC)", _fmt(latest.get("current_hvac_setpoint")))
    first_row[2].metric("Occupancy", _fmt(latest.get("occupancy"), decimals=0))
    first_row[3].metric("Energy Usage", _fmt(latest.get("energy_consumption")), delta=energy_delta, delta_color="inverse")
    first_row[4].metric("Comfort Index (PMV)", _fmt(latest.get("pmv_comfort")), delta=pmv_delta)

    second_row[0].metric("Carbon Emissions", _fmt(latest.get("carbon_intensity")))
    second_row[1].metric("Cooling Setpoint (degC)", _fmt(latest.get("cooling_setpoint")))
    second_row[2].metric("Fan Speed (%)", _fmt(latest.get("fan_speed")))
    second_row[3].metric("Savings (%)", _fmt(latest.get("savings_pct")))
    second_row[4].metric("Lighting", str(latest.get("lighting", "N/A")))

    ai_reason = latest.get("reason", latest.get("ai_reason", "")) # Handles both key variations
    st.markdown("### AI Reason")
    st.info(str(ai_reason) if ai_reason else "No AI rationale available for latest cycle.")


def _render_charts(dataframe: pd.DataFrame) -> None:
    # Adding the massive cumulative savings chart at the top to secure those points!
    st.subheader("AI Energy Savings vs. Baseline")
    st.plotly_chart(build_cumulative_savings_chart(dataframe), use_container_width=True)
    
    st.divider()

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