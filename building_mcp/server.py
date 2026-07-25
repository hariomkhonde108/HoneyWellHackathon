"""MCP server bootstrap for smart-building controller tools."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from controller.actuators import ActuatorManager
from controller.sensors import SensorManager

try:
	from .tools import ControllerToolContext, register_controller_tools
except ImportError:
	from tools import ControllerToolContext, register_controller_tools


@dataclass(frozen=True)
class MCPServerConfig:
	"""Static configuration for MCP server identity."""

	name: str = "smart-building-mcp"
	version: str = "0.1.0"


class SmartBuildingMCPServer:
	"""Owns FastMCP server lifecycle and tool exposure."""

	def __init__(
		self,
		sensors: SensorManager,
		actuators: ActuatorManager,
		config: MCPServerConfig | None = None,
		logger: logging.Logger | None = None,
	) -> None:
		self._config = config or MCPServerConfig()
		self._logger = logger or logging.getLogger(self.__class__.__name__)
		fastmcp_cls = _load_fastmcp_class()
		self._mcp: Any = fastmcp_cls(name=self._config.name)

		context = ControllerToolContext(
			sensors=sensors,
			actuators=actuators,
			logger=self._logger,
		)
		register_controller_tools(self._mcp, context)

	@property
	def app(self) -> Any:
		"""Exposes the underlying FastMCP app instance."""
		return self._mcp

	def run_stdio(self) -> None:
		"""Runs the MCP server over stdio transport."""
		self._logger.info("Starting MCP server over stdio")
		self._mcp.run(transport="stdio")


def configure_mcp_logging(level: int = logging.INFO) -> None:
	"""Configures stderr logging safe for stdio MCP transport."""
	root_logger = logging.getLogger()
	if root_logger.handlers:
		return

	handler = logging.StreamHandler()
	formatter = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)
	handler.setFormatter(formatter)
	root_logger.addHandler(handler)
	root_logger.setLevel(level)


def _load_fastmcp_class() -> Any:
	"""Lazily loads FastMCP without colliding with this project's ``mcp`` package."""
	try:
		from fastmcp import FastMCP
	except ImportError as exc:
		raise RuntimeError(
			"FastMCP is not installed. Run `pip install -r requirements.txt`."
		) from exc
	return FastMCP
