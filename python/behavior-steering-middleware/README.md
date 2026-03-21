# Behavior-Steering Middleware Examples

Three working examples of behavior-steering middleware patterns for LangGraph agents. These patterns demonstrate how to shape agent behavior at the orchestration layer without modifying underlying models.

## Overview

Behavior-Steering Middleware (BSM) intercepts agent execution flows to dynamically adjust behavior based on runtime context. This repository contains three distinct patterns:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Behavior Drift Detection** | Monitors tool usage patterns and corrects when agents stop using tools | Preventing silent degradation in tool-calling agents |
| **Turn-Based Urgency** | Escalates behavior as turn budget is consumed | Resource-constrained conversations (support, triage) |
| **Phase-Based Steering** | Guides agents through multi-phase workflows | Research, analysis, and recommendation workflows |

## Quick Start

```bash
# Install dependencies
poetry install

# Run individual demos
poetry run python -m bsm_examples.drift_detection.demo
poetry run python -m bsm_examples.turn_urgency.demo
poetry run python -m bsm_examples.phase_steering.demo

# Run all tests
poetry run pytest tests/ -v
```

## Pattern 1: Behavior Drift Detection

**Problem:** Agents gradually stop using tools and fall back on parametric knowledge, producing plausible-sounding but incorrect answers.

**Solution:** Monitor tool usage patterns and inject corrective reminders when drift is detected.

```python
from bsm_examples.drift_detection import (
    DriftDetectionConfig,
    DriftSteeringMiddleware,
    ToolUsagePattern,
)

# Define expected tool usage
pattern = ToolUsagePattern(
    name="calculator_agent",
    expected_tools={"calculate": 0.8, "unit_convert": 0.2},
    required_tools=["calculate"],
)

# Configure drift detection
config = DriftDetectionConfig(
    pattern=pattern,
    threshold=0.6,
    reminder_message="Remember to use your available tools.",
)

# Apply middleware
middleware = DriftSteeringMiddleware(pattern, config)
```

**Key Features:**
- Sliding window tool usage tracking
- Weighted drift score calculation (0.0 = no drift, 1.0 = max drift)
- Intervention cooldown to prevent spam
- Required/forbidden tool enforcement

## Pattern 2: Turn-Based Urgency

**Problem:** Agents maintain consistent behavior regardless of remaining conversation budget, wasting turns on exploration when delivery is needed.

**Solution:** Dynamically modify system prompts based on remaining turn budget, shifting from exploratory → focused → urgent.

```python
from bsm_examples.turn_urgency import (
    TurnUrgencyConfig,
    TurnUrgencyMiddleware,
    UrgencyLevel,
)

# Configure urgency thresholds
config = TurnUrgencyConfig(
    max_turns=5,
    thresholds={
        UrgencyLevel.EXPLORATION: 0.6,  # >60% budget remaining
        UrgencyLevel.BALANCED: 0.3,     # 30-60% budget
        UrgencyLevel.EFFICIENCY: 0.0,   # 0-30% budget
    },
)

# Apply middleware
middleware = TurnUrgencyMiddleware(config)
```

**Urgency Levels:**

| Level | Budget | Behavior |
|-------|--------|----------|
| **EXPLORATION** | >60% | Ask clarifying questions, explore alternatives |
| **BALANCED** | 30-60% | Direct answers, limited follow-ups |
| **EFFICIENCY** | <30% | Prioritize delivery, make assumptions |
| **TERMINAL** | Final turn | Deliver immediately, no questions |

## Pattern 3: Phase-Based Steering

**Problem:** Multi-step workflows (gather → analyze → recommend) lack structure—agents jump phases prematurely or get stuck in one phase.

**Solution:** Detect current phase from state/tool usage and inject phase-appropriate guidance.

```python
from bsm_examples.phase_steering import (
    Phase,
    PhaseConfig,
    PhaseSteeringMiddleware,
)

# Define phase behaviors
config = PhaseConfig(
    phases={
        Phase.GATHER: PhaseBehavior(
            description="Collect information",
            allowed_tools=["search", "fetch"],
            max_iterations=5,
        ),
        Phase.ANALYZE: PhaseBehavior(
            description="Analyze findings",
            allowed_tools=["compare", "evaluate"],
            max_iterations=3,
        ),
        Phase.RECOMMEND: PhaseBehavior(
            description="Synthesize recommendations",
            allowed_tools=["synthesize"],
            max_iterations=2,
        ),
    },
    transitions=[
        PhaseTransition(from_phase=Phase.GATHER, to_phase=Phase.ANALYZE),
        PhaseTransition(from_phase=Phase.ANALYZE, to_phase=Phase.RECOMMEND),
    ],
)

# Apply middleware
middleware = PhaseSteeringMiddleware(config)
```

**Phase Detection:**
- Based on tool usage patterns
- State content analysis
- Iteration limits per phase
- Explicit transition signals

## Project Structure

```
python/behavior-steering-middleware/
├── pyproject.toml
├── README.md
├── src/bsm_examples/
│   ├── __init__.py
│   ├── drift_detection/
│   │   ├── __init__.py
│   │   ├── middleware.py    # Core drift detection logic
│   │   ├── state.py         # State type definitions
│   │   └── demo.py          # Runnable demonstration
│   ├── turn_urgency/
│   │   ├── __init__.py
│   │   ├── middleware.py    # Urgency calculation and injection
│   │   ├── types.py         # Config and type definitions
│   │   └── demo.py          # Runnable demonstration
│   └── phase_steering/
│       ├── __init__.py
│       ├── middleware.py    # Phase detection and guidance
│       ├── phase.py         # Phase enum and transitions
│       ├── config.py        # Phase configuration
│       ├── state.py         # Phase state management
│       └── demo.py          # Runnable demonstration
└── tests/
    ├── test_drift_detection.py   # 31 tests
    ├── test_turn_urgency.py      # 34 tests
    └── test_phase_steering.py    # 34 tests
```

## Technical Details

### Dependencies

- **Python:** 3.11+
- **LangGraph:** 1.0+ (modern StateGraph patterns)
- **Pydantic:** 2.0+ (configuration models)
- **LangChain:** 1.0+ (message types)

### Code Quality

```bash
# Type checking
poetry run mypy src/

# Linting
poetry run ruff check src/ tests/

# Formatting
poetry run ruff format src/ tests/

# Testing
poetry run pytest tests/ -v
```

### Design Principles

1. **Middleware Pattern:** Each example wraps LangGraph execution without modifying core agent logic
2. **Pydantic Configuration:** Type-safe, validated configuration with sensible defaults
3. **State Management:** TypedDict-based state with steering metadata
4. **Observability:** Logging and reporting for debugging steering decisions

## Demo Outputs

### Drift Detection

```
--- Turn 2 ---
User: Convert 5 miles to kilometers

⚡ INTERVENTION TRIGGERED ⚡
   Drift score: 0.85
   Reminder: [SYSTEM REMINDER] You appear to be responding without using tools...

Agent: [Tool call: unit_convert(...)]
Drift Score: 0.00 ✓
```

### Turn Urgency

```
[Turn 1/5] Urgency: EXPLORATION
  [Injected modifier: You have ample turns remaining...]

[Turn 4/5] Urgency: EFFICIENCY
  [Injected modifier: You have limited turns remaining...]

[Turn 5/5] Urgency: TERMINAL ⚠️ FINAL TURN
  [Injected modifier: FINAL TURN: Deliver your best solution...]
```

### Phase Steering

```
[Phase: Gather] Starting information gathering...
  → Phase guidance: [PHASE: Gather]

--- Phase Transition: Gather → Analyze ---

[Phase: Analyze] Starting analysis...
  → Phase guidance: [PHASE: Analyze]

--- Phase Transition: Analyze → Recommend ---
```

## Further Reading

These examples accompany a blog post on Agent Behavior-Steering Middleware patterns. Each pattern addresses a specific failure mode in LLM agent systems:

- **Drift:** Silent degradation of tool usage
- **Urgency:** Mismatched behavior to resource constraints  
- **Phases:** Unstructured multi-step workflows

## License

MIT License - See LICENSE file in parent repository.
