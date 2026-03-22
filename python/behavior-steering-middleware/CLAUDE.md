# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Three behavior-steering middleware examples for LangGraph agents using `langchain.agents.middleware.AgentMiddleware`. Each demonstrates how to shape agent behavior at the orchestration layer without modifying underlying models.

Steering content is injected as `<system>`-tagged messages appended to the conversation via `request.override(messages=...)`. The system prompt is static — it only declares that `<system>` tags will appear. State updates flow via `ExtendedModelResponse` + `Command(update={...})`.

## Commands

```bash
poetry install                              # Install dependencies
cp .env.example .env                        # Then add your OPENAI_API_KEY
poetry run pytest tests/ -v                 # Run all tests (108 tests)
poetry run mypy src/                        # Type checking (strict mode)
poetry run ruff check src/ tests/           # Lint
poetry run ruff format src/ tests/          # Format
poetry run python -m bsm_examples.drift_detection.demo   # Requires OPENAI_API_KEY
poetry run python -m bsm_examples.turn_urgency.demo
poetry run python -m bsm_examples.phase_steering.demo
```

## Architecture

Three independent packages under `src/bsm_examples/`, each with consistent layout:

```
models.py      — Pydantic config, enums, dataclasses
state.py       — AgentState subclass (TypedDict with NotRequired fields)
detector.py    — Pure logic class (or calculator.py)
middleware.py   — AgentMiddleware subclass only
demo.py        — create_agent + ChatOpenAI demo
__init__.py    — Direct imports
```

### Drift Detection → Worker Heartbeat Monitor (`drift_detection/`)
Monitors whether the agent uses a specific tool (e.g. `send_message`) within N turns. `HeartbeatDetector` tracks turns since last target tool call. `HeartbeatMiddleware` injects `<system>` reminders when the silence threshold is exceeded. `HeartbeatConfig` controls `target_tool`, `max_silent_turns`, `max_interventions`, `cooldown_turns`.

### Turn Urgency → Shopping Recommendation Agent (`turn_urgency/`)
Calculates urgency level (EXPLORATION → BALANCED → EFFICIENCY → TERMINAL) from remaining turn budget. Shopping-themed prompts shift from "gather details" to "recommend NOW." `TurnUrgencyConfig` adds `required_completion_tool` to force a specific tool call (e.g. `recommend_product`) by the terminal turn. `UrgencyCalculator` is stateless; `TurnUrgencyMiddleware` tracks completion and escalates.

### Phase Steering → Tool-Triggered Personalization (`phase_steering/`)
Guides agents through DISCOVERY → CURATION → RECOMMENDATION → COMPLETE. Phase transitions are triggered by specific tool calls (e.g. `note_preferences` triggers DISCOVERY→CURATION), detected in `wrap_tool_call` after the tool executes. `wrap_model_call` injects phase guidance. `wrap_tool_call` also enforces per-phase tool allow-lists. No heuristics — transitions are deterministic.

## Caveats

- Requires `langchain>=1.2` for the `AgentMiddleware` API.
- Demos require `OPENAI_API_KEY` in `.env` (copy from `.env.example`). Uses `gpt-4.1-mini`.
- Ruff: line-length 100, target Python 3.11, lint rules E/F/I/W.
- mypy strict with `ignore_missing_imports = true`.
