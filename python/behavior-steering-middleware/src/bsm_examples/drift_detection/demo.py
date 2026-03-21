"""Demonstration of Behavior Drift Detection middleware.

This module provides a runnable demonstration showing how the drift detection
middleware detects and corrects behavior drift in an agent that occasionally
"forgets" to use tools.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from bsm_examples.drift_detection.middleware import (
    DriftDetectionConfig,
    DriftDetector,
    ToolUsagePattern,
)

# Configure logging to see drift detection in action
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_tool(a: float, b: float, operation: str) -> dict[str, Any]:
    """Calculator tool for demo purposes.

    Args:
        a: First operand
        b: Second operand
        operation: One of "add", "subtract", "multiply", "divide"

    Returns:
        Result dictionary with the calculation result
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else float("inf"),
    }

    op_func = operations.get(operation, lambda x, y: 0)
    result = op_func(a, b)

    return {"result": result, "operation": operation, "operands": [a, b]}


def unit_convert_tool(value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
    """Unit conversion tool for demo purposes.

    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Result dictionary with converted value
    """
    # Simple conversion factors (just for demo)
    conversions = {
        ("miles", "km"): 1.60934,
        ("km", "miles"): 0.621371,
        ("feet", "meters"): 0.3048,
        ("meters", "feet"): 3.28084,
    }

    factor = conversions.get((from_unit, to_unit), 1.0)
    result = value * factor

    return {
        "result": result,
        "from": f"{value} {from_unit}",
        "to": f"{result:.2f} {to_unit}",
    }


class SimulatedDriftAgent:
    """
    Simulates an agent that drifts from tool usage over time.

    For demo purposes - replaces actual LLM calls with deterministic
    behavior that exhibits drift patterns.
    """

    def __init__(self, drift_after_turn: int = 3) -> None:
        """Initialize the simulated agent.

        Args:
            drift_after_turn: Turn number after which agent starts drifting
        """
        self.turn = 0
        self.drift_after = drift_after_turn
        self.drifted = False
        self.tool_calls: list[dict[str, Any]] = []

    def invoke(self, messages: list[BaseMessage]) -> dict[str, list[BaseMessage]]:
        """Simulate agent invocation.

        Args:
            messages: Input messages

        Returns:
            Response with either tool calls or direct answer
        """
        self.turn += 1

        # Check for drift reminder in messages
        has_reminder = any(
            "SYSTEM REMINDER" in str(m.content)
            for m in messages
            if hasattr(m, "content")
        )

        if has_reminder:
            # Correct back to using tools
            self.drifted = False
            logger.info("  [Agent received reminder - correcting behavior]")
        elif self.turn > self.drift_after and not self.drifted:
            # Start drifting
            self.drifted = True
            logger.info("  [Agent starting to drift - not using tools]")

        # Get the last user message to determine what to do
        last_user_msg = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m.content
                break

        if self.drifted:
            # Return direct answer without tool call (simulating drift)
            content = self._generate_direct_answer(last_user_msg)
            return {"messages": [AIMessage(content=content)]}
        else:
            # Return tool call
            tool_call = self._determine_tool_call(last_user_msg)
            self.tool_calls.append(tool_call)
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": tool_call["name"],
                                "args": tool_call["args"],
                                "id": f"call_{self.turn}",
                            }
                        ],
                    )
                ]
            }

    def _generate_direct_answer(self, query: str | None) -> str:
        """Generate a direct answer without using tools."""
        if query is None:
            return "I can help you with that."

        # Simple pattern matching for demo
        if "*" in query or "multiply" in query.lower():
            return "That would be 345."
        elif "/" in query or "divide" in query.lower():
            return "The result is 12."
        elif "+" in query or "add" in query.lower():
            return "That equals 100."
        elif "convert" in query.lower() or "miles" in query.lower():
            return "That's about 8.05 kilometers."
        else:
            return "I can help you calculate that."

    def _determine_tool_call(self, query: str | None) -> dict[str, Any]:
        """Determine which tool to call based on query."""
        if query is None:
            return {"name": "calculate", "args": {"a": 0, "b": 0, "operation": "add"}}

        query_lower = query.lower()

        # Check for conversion queries
        if "convert" in query_lower or "miles" in query_lower:
            return {
                "name": "unit_convert",
                "args": {"value": 5.0, "from_unit": "miles", "to_unit": "km"},
            }

        # Check for math operations
        if "*" in query or "multiply" in query_lower or "times" in query_lower:
            return {
                "name": "calculate",
                "args": {"a": 15.0, "b": 23.0, "operation": "multiply"},
            }
        elif "/" in query or "divide" in query_lower:
            return {
                "name": "calculate",
                "args": {"a": 144.0, "b": 12.0, "operation": "divide"},
            }
        elif "+" in query or "plus" in query_lower or "add" in query_lower:
            return {
                "name": "calculate",
                "args": {"a": 99.0, "b": 1.0, "operation": "add"},
            }
        else:
            return {
                "name": "calculate",
                "args": {"a": 25.0, "b": 4.0, "operation": "multiply"},
            }


def run_drift_detection_demo() -> None:
    """Run the behavior drift detection demonstration.

    This demo simulates an agent that:
    1. Starts by correctly using tools (turns 1-3)
    2. Begins drifting - answering without tools (turns 4-5)
    3. Gets corrected by the middleware (turn 6)
    4. Resumes proper tool usage (turns 7+)
    """
    print("\n" + "=" * 60)
    print("BEHAVIOR DRIFT DETECTION DEMO")
    print("=" * 60)

    # Define the expected tool usage pattern
    pattern = ToolUsagePattern(
        name="calculator_agent",
        expected_tools={
            "calculate": 0.8,  # Should use calculator 80% of the time
            "unit_convert": 0.2,  # Should use unit converter 20% of the time
        },
        required_tools=["calculate"],  # Must use calculator at least once
        minimum_frequencies={"calculate": 0.5},  # At least 50% usage
    )

    # Configure drift detection
    config = DriftDetectionConfig(
        history_window_size=5,
        recency_bias=1.5,
        warning_threshold=0.3,
        intervention_threshold=0.6,
        max_interventions_per_conversation=3,
        cooldown_turns=1,
        include_statistics=True,
    )

    print(f"\nPattern: {pattern.name}")
    print(f"Expected tools: {pattern.expected_tools}")
    print(f"Required: {pattern.required_tools}")
    print(f"Intervention threshold: {config.intervention_threshold}")
    print()

    # Create the simulated agent that will drift
    agent = SimulatedDriftAgent(drift_after_turn=3)

    # Create the drift detector (we'll use it directly for the demo)
    detector = DriftDetector(pattern, config)

    # Demo conversation turns
    turns = [
        "What is 15 * 23?",
        "Convert 5 miles to kilometers",
        "What is 144 / 12?",
        "What's 99 + 1?",  # Drift starts here
        "And 50 * 2?",  # Drift continues
        "What about 25 * 4?",  # Intervention should happen
        "And 1000 / 10?",  # Corrected behavior
        "Convert 10 miles to km",  # Sustained correction
    ]

    messages: list[BaseMessage] = [
        SystemMessage(
            content="You are a helpful calculator assistant. Always use the calculate or unit_convert tools for math problems."
        )
    ]

    intervention_triggered = False

    for i, user_input in enumerate(turns, 1):
        print(f"--- Turn {i} ---")
        print(f"User: {user_input}")

        # Add user message
        messages.append(HumanMessage(content=user_input))

        # Check if we should intervene before calling the agent
        if detector.should_intervene():
            intervention_triggered = True
            drift_score = detector.calculate_drift_score()
            print("\n⚡ INTERVENTION TRIGGERED ⚡")
            print(f"   Drift score: {drift_score:.2f}")

            # Generate and inject reminder
            weighted_observed = detector._calculate_weighted_frequencies()
            underused_tools = [
                tool
                for tool, expected in pattern.expected_tools.items()
                if weighted_observed.get(tool, 0.0) < expected * 0.5
            ]
            tools_to_mention = underused_tools or list(
                pattern.expected_tools.keys()
            )
            tools_str = ", ".join(f"`{t}`" for t in tools_to_mention[:3])

            reminder = config.reminder_template.format(tools=tools_str)
            if config.include_statistics:
                reminder += f" [Current drift score: {drift_score:.2f}]"

            print(f"   Reminder: {reminder}")
            messages.insert(0, SystemMessage(content=reminder))
            detector.record_intervention()

        # Call the simulated agent
        response = agent.invoke(messages)
        ai_message = response["messages"][0]

        # Display agent response
        if ai_message.tool_calls:
            tool_info = []
            for tc in ai_message.tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("args", {})
                tool_info.append(f"{name}({args})")
            print(f"Agent: [Tool call: {', '.join(tool_info)}]")
        else:
            print(f'Agent: "{ai_message.content}" [NO TOOL CALL]')

        # Add AI message to history
        messages.append(ai_message)

        # Record this turn for drift detection
        detector.record_turn(messages)

        # Show drift score
        drift_score = detector.calculate_drift_score()
        status = "✓" if drift_score < config.warning_threshold else "⚠"
        print(f"Drift Score: {drift_score:.2f} {status}")
        print()

    # Final report
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)

    report = detector.get_drift_report()
    print("\nFinal Statistics:")
    print(f"  Total turns: {report['turn_count']}")
    print(f"  Interventions: {report['intervention_count']}")
    print(f"  Final drift score: {report['drift_score']:.2f}")

    if intervention_triggered:
        print("\n✓ Drift was detected and corrected successfully!")
    else:
        print("\n✓ Agent maintained proper tool usage throughout.")

    print("\nTool Usage Comparison:")
    for tool, data in report["tool_comparison"].items():
        print(
            f"  {tool}: expected={data['expected']:.2f}, "
            f"observed={data['observed']:.2f}"
        )

    print()


def main() -> None:
    """Entry point for the demo."""
    run_drift_detection_demo()


if __name__ == "__main__":
    main()
