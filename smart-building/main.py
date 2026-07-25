"""Main entry point for the Smart Building PoC runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import os
import subprocess
import threading
import sys

import yaml

from controller.actuators import SafetyLimits
from controller.controller import build_default_controller
from llm.reasoning import OllamaReasoningEngine, build_ollama_reasoning_engine
from building_mcp.server import SmartBuildingMCPServer,MCPServerConfig
from building_mcp.tools import ControllerToolContext, build_controller_tool_router


DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


def main() -> int:
	"""Loads settings and starts autonomous simulation control."""
	project_root = Path(__file__).resolve().parent
	config_path = project_root / DEFAULT_CONFIG_PATH

	settings = load_settings(config_path)
	configure_logging(settings)

	logger = logging.getLogger("smart-building.main")
	runtime_mode = str(settings.get("runtime", {}).get("mode", "integrated")).strip().lower()
	logger.info("Starting Smart Building runtime in mode: %s", runtime_mode)

	controller, mcp_server = build_runtime_components(settings=settings, project_root=project_root)

	if runtime_mode == "controller_only":
		return_code = controller.start()
		logger.info("Controller runtime exited with code %s", return_code)
		return int(return_code)

	if runtime_mode == "integrated":
		return run_integrated_mode(settings=settings, controller=controller)

	if runtime_mode == "mcp_only":
		logger.info("Starting MCP server only (controller loop is not started in this mode)")
		mcp_server.run_stdio()
		return 0

	if runtime_mode == "embedded":
		return run_embedded_mode(controller=controller, mcp_server=mcp_server)

	raise ValueError(
		"Invalid runtime.mode. Expected one of: integrated, controller_only, mcp_only, embedded"
	)


def build_runtime_components(settings: dict[str, Any], project_root: Path) -> tuple[Any, SmartBuildingMCPServer]:
	"""Builds controller and MCP server from configuration."""
	simulation_cfg = settings["simulation"]["energyplus"]
	llm_cfg = settings["llm"]
	mcp_cfg = settings.get("mcp", {})
	safety_cfg = settings.get("safety", {})
	cooling_cfg = safety_cfg.get("cooling_setpoint_c", {})
	fan_cfg = safety_cfg.get("fan_speed_percent", {})
	safety_limits = SafetyLimits(
		cooling_min=float(cooling_cfg.get("min", 22.0)),
		cooling_max=float(cooling_cfg.get("max", 27.0)),
		fan_min=float(fan_cfg.get("min", 30.0)),
		fan_max=float(fan_cfg.get("max", 90.0)),
	)

	idf_path = project_root / str(simulation_cfg["idf_path"])
	weather_path = project_root / str(simulation_cfg["weather_path"])
	output_dir = project_root / str(settings["telemetry"]["output_dir"])
	telemetry_prefix = str(settings["telemetry"].get("file_prefix", "telemetry"))
	telemetry_path = output_dir / f"{telemetry_prefix}_metrics.csv"

	require_file(idf_path, "EnergyPlus IDF")
	require_file(weather_path, "EnergyPlus weather")

	reasoning_engine = build_ollama_reasoning_engine(
		model=str(llm_cfg.get("model", "qwen2.5:7b-instruct")),
		host=str(llm_cfg.get("host", "http://localhost:11434")),
		request_timeout_seconds=int(llm_cfg.get("request_timeout_seconds", 20)),
		max_retries_on_invalid_json=int(llm_cfg.get("max_retries_on_invalid_json", 1)),
		max_tokens=int(llm_cfg.get("max_tokens", 150)),
		temperature=float(llm_cfg.get("temperature", 0.0)),
		top_p=float(llm_cfg.get("top_p", 0.9)),
		seed=int(llm_cfg.get("seed", 42)),
		cooling_min=safety_limits.cooling_min,
		cooling_max=safety_limits.cooling_max,
		fan_min=safety_limits.fan_min,
		fan_max=safety_limits.fan_max,
	)
	assert isinstance(reasoning_engine, OllamaReasoningEngine)

	controller = build_default_controller(
		idf_path=idf_path,
		weather_path=weather_path,
		output_dir=output_dir,
		decision_engine=reasoning_engine,
		control_interval_steps=int(settings["simulation"].get("control_interval_steps", 5)),
		zone_name=str(simulation_cfg.get("control_zone", "SPACE1-1")),
		energyplus_install_path=_optional_path(simulation_cfg.get("install_path")),
		telemetry_path=telemetry_path,
		safety_limits=safety_limits,
		max_setpoint_delta=_optional_float(cooling_cfg.get("max_delta_per_ai_tick")),
		max_fan_delta=_optional_float(fan_cfg.get("max_delta_per_ai_tick")),
		fan_max_air_mass_flow_kg_s=float(simulation_cfg.get("fan_max_air_mass_flow_kg_s", 3.0)),
		reconnect_attempts=int(simulation_cfg.get("reconnect_attempts", 2)),
		reconnect_delay_seconds=float(simulation_cfg.get("reconnect_delay_seconds", 2.0)),
	)

	mcp_context = ControllerToolContext(
		sensors=controller.sensors,
		actuators=controller.actuators,
		logger=logging.getLogger("smart-building.mcp.router"),
	)
	mcp_router = build_controller_tool_router(context=mcp_context)
	reasoning_engine.bind_tool_caller(mcp_router)

	mcp_server = SmartBuildingMCPServer(
		sensors=controller.sensors,
		actuators=controller.actuators,
		config=MCPServerConfig(name=str(mcp_cfg.get("server_name", "smart-building-mcp"))),
		logger=logging.getLogger("smart-building.mcp"),
	)

	return controller, mcp_server


def run_integrated_mode(settings: dict[str, Any], controller: Any) -> int:
	"""Runs the full pipeline continuously: EnergyPlus, controller, MCP-backed LLM, dashboard."""
	logger = logging.getLogger("smart-building.main")
	dashboard_process = maybe_start_dashboard(settings=settings)
	if dashboard_process is not None:
		logger.info("Dashboard started with PID %s", dashboard_process.pid)

	try:
		return_code = int(controller.start())
		logger.info("Integrated runtime exited with code %s", return_code)
		return return_code
	finally:
		if dashboard_process is not None:
			dashboard_process.terminate()
			try:
				dashboard_process.wait(timeout=5)
			except subprocess.TimeoutExpired:
				dashboard_process.kill()


def maybe_start_dashboard(settings: dict[str, Any]) -> subprocess.Popen[str] | None:
	"""Starts the Streamlit dashboard as a managed child process when enabled."""
	dashboard_cfg = settings.get("dashboard", {})
	autostart = bool(dashboard_cfg.get("autostart", True))
	if not autostart:
		return None

	project_root = Path(__file__).resolve().parent
	app_path = project_root / "dashboard/app.py"
	if not app_path.exists():
		logging.getLogger("smart-building.main").warning("Dashboard app not found at %s", app_path)
		return None

	env = dict(os.environ)
	port = str(dashboard_cfg.get("port", 8501))
	host = str(dashboard_cfg.get("host", "0.0.0.0"))

	command = [
		sys.executable,
		"-m",
		"streamlit",
		"run",
		str(app_path),
		"--server.port",
		port,
		"--server.address",
		host,
	]

	return subprocess.Popen(
		command,
		cwd=str(project_root),
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
		text=True,
		env=env,
	)


def run_embedded_mode(controller: Any, mcp_server: SmartBuildingMCPServer) -> int:
	"""Runs controller loop in background while MCP stdio serves in foreground."""
	logger = logging.getLogger("smart-building.main")
	controller_result: dict[str, int] = {"code": 0}

	def run_controller() -> None:
		controller_result["code"] = int(controller.start())

	controller_thread = threading.Thread(
		target=run_controller,
		name="controller-runtime",
		daemon=True,
	)
	controller_thread.start()
	logger.info("Controller started in background thread; MCP stdio running in foreground")

	try:
		mcp_server.run_stdio()
	finally:
		controller.stop()
		controller_thread.join(timeout=5.0)

	return controller_result["code"]


def load_settings(config_path: Path) -> dict[str, Any]:
	"""Reads and validates the YAML settings file."""
	if not config_path.exists():
		raise FileNotFoundError(f"Config not found: {config_path}")

	with config_path.open("r", encoding="utf-8") as config_file:
		loaded = yaml.safe_load(config_file)

	if not isinstance(loaded, dict):
		raise ValueError("settings.yaml must contain a top-level object")

	required_sections = ("simulation", "llm", "telemetry", "logging")
	missing = [section for section in required_sections if section not in loaded]
	if missing:
		raise ValueError(f"settings.yaml missing sections: {', '.join(missing)}")

	return loaded


def configure_logging(settings: dict[str, Any]) -> None:
	"""Configures runtime logging from YAML settings."""
	logging_cfg = settings.get("logging", {})
	level_name = str(logging_cfg.get("level", "INFO")).upper()
	level = getattr(logging, level_name, logging.INFO)

	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def require_file(path: Path, label: str) -> None:
	"""Raises a clear error if a required runtime file is missing."""
	if not path.exists():
		raise FileNotFoundError(f"{label} file not found at: {path}")


def _optional_path(value: Any) -> Path | None:
	"""Returns a Path for a configured optional filesystem value."""
	if value is None or not str(value).strip():
		return None
	return Path(str(value))


def _optional_float(value: Any) -> float | None:
	"""Returns a float for optional numeric configuration values."""
	return None if value is None else float(value)


if __name__ == "__main__":
	try:
		raise SystemExit(main())
	except Exception as exc:  # pragma: no cover - top-level runtime guard
		logging.getLogger("smart-building.main").exception("Fatal startup error: %s", exc)
		raise SystemExit(1) from exc
