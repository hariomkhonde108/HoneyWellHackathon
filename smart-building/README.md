# AI-Powered Autonomous Smart Building Optimization System

Production-quality Proof of Concept scaffold for a hackathon project that combines EnergyPlus, MCP, and a local open-source LLM (Ollama + Qwen2.5 7B Instruct) to optimize HVAC controls autonomously.

## Status

This repository currently contains complete project structure and configuration scaffolding only.
No application logic has been implemented yet by design.

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
3. Place your actual EnergyPlus assets:
   - `energyplus/building.idf`
   - `energyplus/weather.epw`
4. Update any local paths/settings in `config/settings.yaml`.

## Next Implementation Steps

1. Implement EnergyPlus runtime wrapper and callback lifecycle in `controller/simulation.py`.
2. Implement sensor and actuator interfaces in `controller/sensors.py` and `controller/actuators.py`.
3. Implement MCP server/tool registry in `mcp/server.py` and `mcp/tools.py`.
4. Implement deterministic prompting and JSON parser in `llm/prompt.py` and `llm/parser.py`.
5. Implement orchestration loop and fault-tolerance in `controller/controller.py`.
6. Implement dashboard metrics and charts in `dashboard/app.py` and `dashboard/charts.py`.

## Notes

- This scaffold intentionally avoids placeholder logic and runtime behavior.
- The next phase should implement modules incrementally with tests per subsystem.
