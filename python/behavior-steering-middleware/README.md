# Behavior-Steering Middleware Examples

Three working examples of behavior-steering middleware patterns for LangGraph agents. These patterns
demonstrate how to shape agent behavior at the orchestration layer without modifying underlying models.

## Overview

Behavior-Steering Middleware (BSM) intercepts agent execution flows to dynamically adjust behavior
based on runtime context. Each middleware extends `AgentMiddleware` from `langchain.agents.middleware`
and hooks into `create_agent` via `wrap_model_call`, `after_model`, and `wrap_tool_call`. Steering
content is injected as `<system>`-tagged messages appended to the conversation — the system prompt is
static and only declares that `<system>` tags will appear. State updates flow back to the agent via
`ExtendedModelResponse` and `Command(update={...})`.

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Behavior Drift Detection** | Monitors tool usage patterns and corrects when agents stop using tools | Preventing silent degradation in tool-calling agents |
| **Turn-Based Urgency** | Escalates behavior as turn budget is consumed | Resource-constrained conversations (support, triage) |
| **Phase-Based Steering** | Guides agents through multi-phase workflows | Research, analysis, and recommendation workflows |

## Quick Start

```bash
# Install dependencies
poetry install

# Configure API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run individual demos (each requires OPENAI_API_KEY)
poetry run python -m bsm_examples.drift_detection.demo
poetry run python -m bsm_examples.turn_urgency.demo
poetry run python -m bsm_examples.phase_steering.demo

# Run all tests
poetry run pytest tests/ -v
```

## Pattern 1: Behavior Drift Detection

**Problem:** Agents gradually stop using tools and fall back on parametric knowledge, producing
plausible-sounding but unverified answers.

**Solution:** Monitor tool usage with a recency-weighted sliding window. When the drift score
exceeds a threshold, inject a corrective `<system>` message into the next model call.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from bsm_examples.drift_detection.middleware import (
    DriftDetectionConfig,
    DriftSteeringMiddleware,
    ToolUsagePattern,
)
from bsm_examples.drift_detection.state import DriftAgentState


@tool
def calculate(a: float, b: float, operation: str) -> str:
    """Perform arithmetic. Operations: add, subtract, multiply, divide."""
    ...


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units. Supports miles/km, feet/meters."""
    ...


pattern = ToolUsagePattern(
    name="calculator_agent",
    expected_tools={"calculate": 0.8, "unit_convert": 0.2},
    required_tools=["calculate"],
    minimum_frequencies={"calculate": 0.5},
)
config = DriftDetectionConfig(
    intervention_threshold=0.6,
    cooldown_turns=1,
    max_interventions_per_conversation=3,
)
middleware = DriftSteeringMiddleware(pattern, config)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    tools=[calculate, unit_convert],
    system_prompt=(
        "You are a calculator assistant. Always use tools for math. "
        "You will receive <system> tags with behavioral guidance — follow them."
    ),
    middleware=[middleware],
    state_schema=DriftAgentState,
)

result = agent.invoke({"messages": [HumanMessage(content="What is 15 * 23?")]})
drift_score = result.get("drift_score", 0.0)
intervention_count = result.get("intervention_count", 0)
```

**Key features:**
- Sliding window (`deque`) with recency-weighted frequency scoring
- Drift score 0.0 (no drift) to 1.0 (maximum drift)
- Intervention cooldown to prevent message spam
- Required-tool enforcement
- `DriftAgentState` exposes `drift_score`, `intervention_count`, and `drift_report`

## Pattern 2: Turn-Based Urgency

**Problem:** Agents maintain consistent behavior regardless of remaining turn budget, wasting turns
on exploration when delivery is needed.

**Solution:** Compute urgency from remaining turns and append an urgency-tagged `<system>` modifier
to every model call, shifting the agent from exploratory to decisive as the budget is consumed.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from bsm_examples.turn_urgency import TurnUrgencyConfig, TurnUrgencyMiddleware
from bsm_examples.turn_urgency.types import UrgencyAgentState


@tool
def analyze_code(code_snippet: str) -> str:
    """Analyze a code snippet for common issues."""
    ...


config = TurnUrgencyConfig(max_turns=5)
middleware = TurnUrgencyMiddleware(config)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    tools=[analyze_code],
    system_prompt=(
        "You are a Python debugging assistant. "
        "You will receive <system> tags with behavioral guidance — follow them."
    ),
    middleware=[middleware],
    state_schema=UrgencyAgentState,
)

result = agent.invoke({"messages": messages})
urgency_level = result.get("urgency_level")   # "EXPLORATION" | "BALANCED" | ...
turns_remaining = result.get("turns_remaining")
```

**Urgency levels:**

| Level | Budget remaining | Injected behavior |
|-------|-----------------|-------------------|
| **EXPLORATION** | >60% | Ask clarifying questions, explore alternatives |
| **BALANCED** | 30-60% | Direct answers, limited follow-ups |
| **EFFICIENCY** | <30% | Prioritize delivery, make assumptions |
| **TERMINAL** | Final turn | Deliver immediately, no questions |

`UrgencyAgentState` exposes `urgency_level`, `turns_used`, `turns_remaining`, and `is_terminal`.
Three turn-counting modes are supported: pairs, human-only, and agent-only.

## Pattern 3: Phase-Based Steering

**Problem:** Multi-step workflows (gather → analyze → recommend) lack structure — agents jump phases
prematurely or get stuck in one phase indefinitely.

**Solution:** Detect the current phase from state and tool usage, inject phase-appropriate
`<system>` guidance on every model call, and enforce per-phase tool allowlists via `wrap_tool_call`.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from bsm_examples.phase_steering.config import PhaseBehavior, PhaseConfig, PhaseTransition
from bsm_examples.phase_steering.middleware import PhaseSteeringMiddleware
from bsm_examples.phase_steering.phase import Phase
from bsm_examples.phase_steering.state import PhaseAgentState


@tool
def search(query: str) -> str:
    """Search for information on a topic."""
    ...


@tool
def compare(items: str) -> str:
    """Compare multiple approaches. Pass comma-separated items."""
    ...


@tool
def synthesize(analysis: str) -> str:
    """Synthesize analysis into actionable recommendations."""
    ...


config = PhaseConfig(
    phases={
        Phase.GATHER: PhaseBehavior(
            system_prompt_addition="Collect information. Use search and fetch_document.",
            allowed_tools=["search", "fetch_document"],
            max_iterations=5,
        ),
        Phase.ANALYZE: PhaseBehavior(
            system_prompt_addition="Compare approaches and identify trade-offs.",
            allowed_tools=["compare"],
            max_iterations=3,
        ),
        Phase.RECOMMEND: PhaseBehavior(
            system_prompt_addition="Synthesize clear, actionable recommendations.",
            allowed_tools=["synthesize"],
            max_iterations=2,
        ),
        Phase.COMPLETE: PhaseBehavior(
            system_prompt_addition="Present the final summary to the user.",
            allowed_tools=[],
            max_iterations=1,
        ),
    },
    transitions=[
        PhaseTransition(from_phase=Phase.GATHER, to_phase=Phase.ANALYZE),
        PhaseTransition(from_phase=Phase.ANALYZE, to_phase=Phase.RECOMMEND),
        PhaseTransition(from_phase=Phase.RECOMMEND, to_phase=Phase.COMPLETE),
    ],
    initial_phase=Phase.GATHER,
    enable_auto_transition=True,
)
middleware = PhaseSteeringMiddleware(config)

agent = create_agent(
    model=ChatOpenAI(model="gpt-4.1-mini"),
    tools=[search, compare, synthesize],
    system_prompt=(
        "You are a research assistant. "
        "You will receive <system> tags with phase guidance "
        "— follow them and use only the tools specified."
    ),
    middleware=[middleware],
    state_schema=PhaseAgentState,
)

result = agent.invoke({"messages": messages})
current_phase = result.get("current_phase")
phase_iteration = result.get("phase_iteration")
```

**Phase detection priority:** explicit state field > tool usage patterns > state content > default.
`PhaseAgentState` exposes `current_phase`, `phase_iteration`, and `total_iterations`.

## Project Structure

```
python/behavior-steering-middleware/
├── .env.example
├── pyproject.toml
├── README.md
├── src/bsm_examples/
│   ├── __init__.py
│   ├── drift_detection/
│   │   ├── __init__.py
│   │   ├── middleware.py    # DriftDetector + DriftSteeringMiddleware
│   │   ├── state.py         # DriftAgentState
│   │   └── demo.py          # Runnable demonstration (requires OPENAI_API_KEY)
│   ├── turn_urgency/
│   │   ├── __init__.py
│   │   ├── middleware.py    # UrgencyCalculator + TurnUrgencyMiddleware
│   │   ├── types.py         # TurnUrgencyConfig, UrgencyLevel, UrgencyAgentState
│   │   └── demo.py          # Runnable demonstration (requires OPENAI_API_KEY)
│   └── phase_steering/
│       ├── __init__.py
│       ├── middleware.py    # PhaseSteeringMiddleware
│       ├── phase.py         # Phase enum
│       ├── config.py        # PhaseConfig, PhaseBehavior, PhaseTransition
│       ├── state.py         # PhaseState (Pydantic) + PhaseAgentState (TypedDict)
│       └── demo.py          # Runnable demonstration (requires OPENAI_API_KEY)
└── tests/
    ├── helpers.py
    ├── test_drift_detection.py
    ├── test_turn_urgency.py
    └── test_phase_steering.py
```

## Technical Details

### Dependencies

- **Python:** 3.11+
- **LangChain:** 1.2+ (required for `AgentMiddleware`, `ExtendedModelResponse`)
- **LangGraph:** 1.0+ (modern `StateGraph` patterns)
- **LangChain-OpenAI:** for `ChatOpenAI` in demos
- **Pydantic:** 2.0+ (configuration and state models)

### Code Quality

```bash
# Type checking (strict mode)
poetry run mypy src/

# Linting
poetry run ruff check src/ tests/

# Formatting
poetry run ruff format src/ tests/

# Testing
poetry run pytest tests/ -v
```

### Design Principles

1. **Middleware pattern:** Each example extends `AgentMiddleware` and hooks into `create_agent`
   without modifying core agent or model logic.
2. **Static system prompt:** The system prompt declares that `<system>` tags will appear but is
   never mutated at runtime. All steering content is appended as separate messages.
3. **Injection via `request.override`:** `wrap_model_call` appends `SystemMessage` objects with
   `<system>` tags to `request.messages` before the model call proceeds.
4. **State observability:** Middleware writes diagnostics back to agent state via
   `ExtendedModelResponse` + `Command(update={...})`, making drift scores, urgency levels, and
   phase tracking accessible on every `agent.invoke` result.
5. **Pydantic configuration:** Type-safe, validated configuration with sensible defaults across all
   three patterns.

## Further Reading

These examples accompany a blog post on Agent Behavior-Steering Middleware patterns. Each pattern
addresses a specific failure mode in LLM agent systems:

- **Drift:** Silent degradation of tool usage over a conversation
- **Urgency:** Mismatched agent behavior relative to remaining resource budget
- **Phases:** Unstructured multi-step workflows that skip or stall in critical phases

## License

MIT License - See LICENSE file in parent repository.
