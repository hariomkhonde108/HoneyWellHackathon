# System Architecture: Autonomous Smart Building Optimization

## 1. Overview
This Proof-of-Concept (PoC) demonstrates a fully autonomous, closed-loop Building Management System (BMS). It utilizes a physics-based simulation engine (EnergyPlus) paired with a local Open-Source Large Language Model (Qwen2.5 7B) to dynamically adjust HVAC and lighting setpoints. The architecture is 100% edge-deployable, requiring no cloud connectivity, ensuring maximum data privacy and zero API latency.
## 2. Tool-Calling Architecture & MCP Integration
To achieve a scalable and modular design, the system leverages the **Model Context Protocol (MCP)** to decouple the AI reasoning engine from the underlying Python control execution.

* **The Controller (Python):** Orchestrates the EnergyPlus simulation via the `PyEnergyPlus` API. It acts as the state manager, reading sensor snapshots at every timestep and enforcing strict safety boundaries on all actuator commands.
* **The MCP Router:** Translates raw EnergyPlus sensor data (Indoor Temp, PMV, Occupancy, Energy Demand) into standardized MCP tool outputs.
* **The LLM Agent:** The Qwen2.5 model does not interact with the building directly. Instead, it queries the MCP tools to gather current context, reasons over the thermodynamic and comfort trade-offs, and outputs a strict JSON payload containing the computed Energy Conservation Measures (ECMs).
* **Forward Injection:** The Python controller parses the JSON, validates it against the `ActuatorManager` safety limits (e.g., maximum setpoint delta per tick), and injects the overriding setpoints back into the active EnergyPlus memory state.

## 3. Prompt Engineering & Reliability Strategies
To ensure the LLM acts as a deterministic controller rather than a conversational chatbot, several rigid prompt engineering strategies were implemented:

* **Strict JSON Enforcement:** The system prompt forces the LLM to output a predefined JSON schema (`cooling_setpoint`, `fan_speed`, `lighting`, `reason`).
* **Dynamic Constraint Injection:** The prompt dynamically includes hard safety constraints (e.g., cooling limits of 22°C - 27°C) to ground the model's reasoning within physical equipment limits before it generates a response.
* **Multi-Objective Prioritization:** The prompt explicitly instructs the LLM on how to handle trade-offs based on live states. For example: *"If Occupancy is 0, human comfort (PMV) constraints are lifted; prioritize aggressive energy reduction."*
* **Pydantic Fallback Loops:** If the LLM generates a hallucinated type (e.g., `lighting: "On"` instead of strict uppercase `"ON"`), a custom `LLMOutputParser` intercepts and coerces the output to prevent Pydantic validation crashes, ensuring the loop never breaks during an extended simulation horizon.

## 4. Prompt Latency Management
In a real-time BMS, decision latency must be shorter than the physical thermal dynamics of the building.

* **Local Inference:** By hosting a quantized 7-billion parameter model (`qwen2.5:7b-instruct`) via Ollama locally, round-trip network latency is completely eliminated.
* **Optimized Token Limits:** The generation request is strictly bounded (`max_tokens: 150`) to prevent the LLM from generating excessively long reasoning chains, ensuring the forward injection happens synchronously within the EnergyPlus timestep callback.

## 5. System Workflow Diagram
The following diagram illustrates the closed-loop execution framework where telemetry continuously streams from the simulation, is processed by the AI, and is injected back as live control actions.

```mermaid
graph TD
    subgraph Simulation Environment
        EP[EnergyPlus Engine] 
    end

    subgraph Edge Control Layer
        PC[Python Controller & PyEnergyPlus API]
        AM[ActuatorManager / Safety Validator]
    end

    subgraph Cognitive Engine
        MCP[MCP Router]
        LLM((Qwen2.5 7B LLM))
    end

    %% Flow of data
    EP -->|1. Stream Telemetry: Temp, PMV, Energy| PC
    PC -->|2. Format Sensor Snapshot| MCP
    MCP -->|3. Tool Context & Constraints| LLM
    LLM -->|4. Strict JSON Output ECMs| MCP
    MCP -->|5. Parsed Payload| AM
    AM -->|6. Validated Setpoints & Overrides| PC
    PC -->|7. Forward Injection| EP

    %% Styling
    style EP fill:#1e88e5,stroke:#005cb2,stroke-width:2px,color:#fff
    style LLM fill:#43a047,stroke:#00701a,stroke-width:2px,color:#fff
    style PC fill:#fb8c00,stroke:#c56000,stroke-width:2px,color:#fff


