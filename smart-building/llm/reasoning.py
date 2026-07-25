"""Ollama-backed reasoning engine for structured control actions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Protocol
import logging

from ollama import Client
from ollama import RequestError, ResponseError

from controller.actuators import ControlAction
from controller.sensors import SensorSnapshot
from llm.parser import LLMOutputParser, LLMOutputValidationError
from llm.prompt import PromptBuilder


@dataclass(frozen=True)
class OllamaConfig:
	"""Runtime configuration for Ollama chat inference."""

	model: str = "qwen2.5:7b-instruct"
	host: str = "http://localhost:11434"
	request_timeout_seconds: int = 20
	max_retries_on_invalid_json: int = 1
	max_tokens: int = 150
	temperature: float = 0.0
	top_p: float = 0.9
	seed: int = 42


class MCPToolCaller(Protocol):
	"""Protocol for in-process MCP tool invocation."""

	def call_tool(self, name: str, **kwargs: Any) -> Any:
		"""Calls a named MCP tool."""


class OllamaReasoningEngine:
	"""Decision engine that returns validated ControlAction objects."""

	def __init__(
		self,
		config: OllamaConfig | None = None,
		prompt_builder: PromptBuilder | None = None,
		parser: LLMOutputParser | None = None,
		tool_caller: MCPToolCaller | None = None,
		logger: logging.Logger | None = None,
	) -> None:
		self._config = config or OllamaConfig()
		self._prompt_builder = prompt_builder or PromptBuilder()
		self._parser = parser or LLMOutputParser()
		self._tool_caller = tool_caller
		self._logger = logger or logging.getLogger(self.__class__.__name__)
		self._client = Client(host=self._config.host)

	def bind_tool_caller(self, tool_caller: MCPToolCaller) -> None:
		"""Binds an MCP tool caller used to fetch live context before inference."""
		self._tool_caller = tool_caller

	def __call__(
		self,
		snapshot: SensorSnapshot,
		previous_action: ControlAction | None,
	) -> ControlAction:
		"""Generates one control action with retry-on-validation-failure."""
		retries = max(0, self._config.max_retries_on_invalid_json)
		last_error: str | None = None
		mcp_observations = self._read_mcp_observations(snapshot)

		for attempt in range(retries + 1):
			try:
				response_payload = self._request_action(
					snapshot=snapshot,
					previous_action=previous_action,
					mcp_observations=mcp_observations,
					validation_error=last_error,
				)
				action = self._parser.parse(response_payload)
				self._logger.info("Validated LLM action on attempt %s", attempt + 1)
				return action
			except LLMOutputValidationError as exc:
				last_error = str(exc)
				self._logger.warning("Invalid LLM JSON on attempt %s: %s", attempt + 1, last_error)
				continue

		raise LLMOutputValidationError(
			f"LLM failed to produce valid JSON after {retries + 1} attempt(s)."
		)

	def _request_action(
		self,
		snapshot: SensorSnapshot,
		previous_action: ControlAction | None,
		mcp_observations: dict[str, float | None],
		validation_error: str | None,
	) -> str | dict[str, Any]:
		messages = [
			{"role": "system", "content": self._prompt_builder.build_system_prompt()},
			{
				"role": "user",
				"content": self._prompt_builder.build_user_prompt(
					snapshot=snapshot,
					previous_action=previous_action,
					mcp_observations=mcp_observations,
					validation_error=validation_error,
				),
			},
		]

		options = {
			"temperature": self._config.temperature,
			"top_p": self._config.top_p,
			"num_predict": self._config.max_tokens,
			"seed": self._config.seed,
		}

		try:
			result = self._chat_with_timeout(messages=messages, options=options)
		except (RequestError, ResponseError) as exc:
			raise TimeoutError(f"Ollama request failed: {exc}") from exc
		except Exception as exc:  # pragma: no cover - SDK/network defensive guard
			message = str(exc).lower()
			if "timeout" in message or "timed out" in message:
				raise TimeoutError(f"Ollama timeout: {exc}") from exc
			raise

		message = result.get("message") if isinstance(result, dict) else None
		content = message.get("content") if isinstance(message, dict) else None
		if content is None:
			raise LLMOutputValidationError("Ollama response missing message content.")
		return content

	def _chat_with_timeout(
		self,
		messages: list[dict[str, str]],
		options: dict[str, float | int],
	) -> dict[str, Any]:
		"""Runs an Ollama chat request with a hard timeout bound."""

		with ThreadPoolExecutor(max_workers=1) as executor:
			future = executor.submit(
				self._client.chat,
				model=self._config.model,
				messages=messages,
				format="json",
				options=options,
			)
			try:
				result = future.result(timeout=self._config.request_timeout_seconds)
			except FutureTimeoutError as exc:
				future.cancel()
				raise TimeoutError(
					f"Ollama request exceeded {self._config.request_timeout_seconds}s timeout"
				) from exc

		if not isinstance(result, dict):
			raise LLMOutputValidationError("Ollama response is not a JSON object.")
		return result

	def _read_mcp_observations(self, snapshot: SensorSnapshot) -> dict[str, float | None]:
		"""Reads sensor context via MCP tools; falls back to local snapshot values."""
		fallback = {
			"indoor_temperature": snapshot.indoor_temperature,
			"outdoor_temperature": snapshot.outdoor_temperature,
			"energy_consumption": snapshot.energy_consumption,
			"pmv_comfort": snapshot.pmv_comfort,
			"occupancy": snapshot.occupancy,
			"carbon_intensity": snapshot.carbon_intensity,
			"current_hvac_setpoint": snapshot.current_hvac_setpoint,
		}

		if self._tool_caller is None:
			return fallback

		tool_map = {
			"indoor_temperature": "get_temperature",
			"outdoor_temperature": "get_outdoor_temperature",
			"energy_consumption": "get_energy_usage",
			"pmv_comfort": "get_comfort_index",
			"occupancy": "get_occupancy",
			"carbon_intensity": "get_carbon_intensity",
			"current_hvac_setpoint": "get_current_setpoint",
		}

		observations = dict(fallback)
		for key, tool_name in tool_map.items():
			try:
				value = self._tool_caller.call_tool(tool_name)
				if isinstance(value, (float, int)) or value is None:
					observations[key] = None if value is None else float(value)
			except Exception as exc:  # pragma: no cover - keep loop resilient
				self._logger.warning("MCP tool %s failed: %s", tool_name, exc)

		return observations


def build_ollama_reasoning_engine(
	*,
	model: str = "qwen2.5:7b-instruct",
	host: str = "http://localhost:11434",
	request_timeout_seconds: int = 20,
	max_retries_on_invalid_json: int = 1,
	max_tokens: int = 150,
	temperature: float = 0.0,
	top_p: float = 0.9,
	seed: int = 42,
) -> OllamaReasoningEngine:
	"""Factory for creating a configured Ollama Qwen2.5 reasoning engine."""
	config = OllamaConfig(
		model=model,
		host=host,
		request_timeout_seconds=request_timeout_seconds,
		max_retries_on_invalid_json=max_retries_on_invalid_json,
		max_tokens=max_tokens,
		temperature=temperature,
		top_p=top_p,
		seed=seed,
	)
	return OllamaReasoningEngine(config=config)
