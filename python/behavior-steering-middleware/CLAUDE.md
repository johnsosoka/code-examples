# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Example implementations of three Behavior-Steering Middleware (BSM) patterns for LangGraph agents. These accompany a blog post and demonstrate how to shape agent behavior at the orchestration layer without modifying underlying models.

All three middlewares extend `langchain.agents.middleware.AgentMiddleware` and integrate with `create_agent` via `wrap_model_call`, `after_model`, and `wrap_tool_call` hooks. Steering content is injected as `<system>`-tagged messages appended to the conversation via `request.override(messages=...)` — the system prompt is never modified at runtime, it only sets expectations that `<system>` tags will appear. Observability data is pushed into agent state via `ExtendedModelResponse` + `Command(update={...})`.

## Commands

```bash
poetry install                              # Install dependencies
poetry run pytest tests/ -v                 # Run all tests
poetry run pytest tests/test_drift_detection.py -v  # Run single test file
poetry run mypy src/                        # Type checking (strict mode)
poetry run ruff check src/ tests/           # Lint
poetry run ruff format src/ tests/          # Format
poetry run python -m bsm_examples.drift_detection.demo   # Run demo (requires OPENAI_API_KEY)
poetry run python -m bsm_examples.turn_urgency.demo
poetry run python -m bsm_examples.phase_steering.demo
```

## Architecture

Three independent middleware patterns under `src/bsm_examples/`, each following the same structure: middleware (core logic), state/types (data models), and demo (runnable example with `create_agent`).

**Common pattern across all three:** Each middleware extends `AgentMiddleware` and implements `wrap_model_call` to intercept model calls. Steering content is appended as `<system>`-tagged `SystemMessage` objects to `request.messages` via `request.override(messages=...)` — the system prompt stays static and only declares that `<system>` tags will appear. State updates use `ExtendedModelResponse(model_response=response, command=Command(update={...}))`. Each defines an `AgentState` subclass with `NotRequired` fields for observability.

### Drift Detection (`drift_detection/`)
Monitors tool usage via a sliding window (`deque`) with recency-weighted frequency scoring. `DriftDetector` calculates a 0.0-1.0 drift score; `DriftSteeringMiddleware` extends `AgentMiddleware` with `wrap_model_call` (injection) and `after_model` (drift report). `DriftAgentState` extends `AgentState` with `drift_score`, `intervention_count`, `drift_report`. Has cooldown and max-intervention limits.

### Turn Urgency (`turn_urgency/`)
Calculates urgency level (EXPLORATION -> BALANCED -> EFFICIENCY -> TERMINAL) from remaining turn budget. `UrgencyCalculator` is stateless; `TurnUrgencyMiddleware` extends `AgentMiddleware` with `wrap_model_call` to inject urgency-tagged modifiers. `UrgencyAgentState` extends `AgentState` with `urgency_level`, `turns_used`, `turns_remaining`, `is_terminal`. Supports three turn-counting modes.

### Phase Steering (`phase_steering/`)
Guides agents through GATHER -> ANALYZE -> RECOMMEND -> COMPLETE phases. `PhaseSteeringMiddleware` extends `AgentMiddleware` with `wrap_model_call` (phase detection + guidance injection) and `wrap_tool_call` (per-phase tool enforcement). Uses `PhaseDetector` strategy pattern internally. `PhaseState` (Pydantic) tracks rich internal state; `PhaseAgentState` (TypedDict) is the lightweight agent state schema. Uses lazy `__getattr__` imports in `__init__.py` to avoid circular dependencies.

## Caveats

- Requires `langchain>=1.2` for the `AgentMiddleware` API (`wrap_model_call`, `ExtendedModelResponse`, etc.).
- Demos require an OpenAI API key in `.env` (copy from `.env.example`). Uses `gpt-4.1-mini`.
- Ruff config: line-length 100, target Python 3.11, lint rules E/F/I/W.
- mypy is strict mode with `ignore_missing_imports = true`.
