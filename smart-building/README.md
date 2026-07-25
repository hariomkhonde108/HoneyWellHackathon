# AI-Powered Autonomous Smart Building Optimization System

Production-quality Proof of Concept for a hackathon project that combines EnergyPlus, MCP, and a local open-source LLM (Ollama + Qwen2.5 7B Instruct) to optimize HVAC controls autonomously.

## Status

The controller is fully implemented: it registers PyEnergyPlus variables and
actuators, reads live telemetry through MCP tools, requests a schema-validated
Ollama decision, applies safe control commands, and continuously records
dashboard telemetry. It falls back to the previous safe action on an LLM timeout
or invalid output and skips cycles with missing required sensors.

## Project Structure

smart-building/

- controller/
  - controller.py
  - simulation.py
  - sensors.py
  - actuators.py
- llm/
  - prompt.py
  - reasoning.py
  - parser.py
- mcp/
  - server.py
  - tools.py
- dashboard/
  - app.py
  - charts.py
- config/
  - settings.yaml
- logs/
- energyplus/
  - building.idf
  - weather.epw
- utils/
  - logger.py
- main.py
- requirements.txt
- README.md

## Planned Autonomous Control Loop

EnergyPlus -> Sensor Collection -> MCP Tools -> LLM Reasoning -> Structured JSON Action -> Validation -> Actuation -> Repeat

## Tech Stack

- Python 3.12
- EnergyPlus (via PyEnergyPlus API)
- MCP (Model Context Protocol)
- Ollama + Qwen2.5 7B Instruct
- Streamlit + Plotly
- Pandas
- YAML configuration

## Prerequisites

1. Python 3.12 installed.
2. EnergyPlus 23.x installed and accessible on the host machine.
3. Ollama installed and running locally.
4. Model pulled locally:
   - `ollama pull qwen2.5:7b-instruct`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Set `simulation.energyplus.install_path` in `config/settings.yaml` to your
   EnergyPlus installation, or set `ENERGYPLUS_INSTALL_PATH`. This project is
   preconfigured for `C:/EnergyPlusV26-1-0`.
4. The repository includes the EnergyPlus `5ZoneAirCooled` reference model and
   Chicago TMY3 weather data. The configured controlled zone is `SPACE1-1`.

## Run

Start Ollama and pull the model once:

```powershell
ollama pull qwen2.5:7b-instruct
```

Start the integrated controller and dashboard:

```powershell
.\.venv\Scripts\python.exe main.py
```

The dashboard is served on `http://localhost:8501`. To expose only the MCP
server over standard input/output, set `runtime.mode: mcp_only` in the YAML.

## Notes

- PyEnergyPlus is supplied by the local EnergyPlus installation, not PyPI.
- The supplied EnergyPlus reference model uses `Supply Fan 1` and `LIGHTS-1`;
  change the bindings in `controller/actuators.py` when using a different IDF.
