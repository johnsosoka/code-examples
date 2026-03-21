"""Demonstration of turn-based urgency middleware.

This demo showcases progressive urgency escalation through a simulated
5-turn conversation where the agent helps debug a Python script.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from bsm_examples.turn_urgency import (
    TurnUrgencyConfig,
    TurnUrgencyMiddleware,
    UrgencyLevel,
)


def print_header() -> None:
    """Print demo header."""
    print("=" * 70)
    print("Turn-Based Urgency Middleware Demo")
    print("Task: Python debugging with 5-turn budget")
    print("=" * 70)
    print()


def print_config(config: TurnUrgencyConfig) -> None:
    """Print configuration details."""
    print("Configuration:")
    print(f"  Max turns: {config.max_turns}")
    print(
        f"  Thresholds: EXPLORATION > {config.thresholds.balanced_max:.0%} > "
        f"BALANCED > {config.thresholds.efficiency_max:.0%} > EFFICIENCY > 0% > TERMINAL"
    )
    print()


def print_turn(
    turn_number: int,
    max_turns: int,
    urgency_level: UrgencyLevel,
    user_message: str,
    agent_response: str,
    is_transition: bool = False,
) -> None:
    """Print a single turn of the conversation.

    Args:
        turn_number: Current turn number (1-based).
        max_turns: Maximum allowed turns.
        urgency_level: Current urgency level.
        user_message: User's input message.
        agent_response: Agent's response.
        is_transition: Whether this turn includes an urgency transition.
    """
    print("-" * 70)

    # Show urgency indicator
    if urgency_level == UrgencyLevel.TERMINAL:
        print(f"[Turn {turn_number}/{max_turns}] Urgency: {urgency_level.name} ⚠️ FINAL TURN")
    elif is_transition:
        prev_level = get_previous_level(urgency_level)
        print(f"[Turn {turn_number}/{max_turns}] Urgency: {prev_level.name} -> {urgency_level.name}")
    else:
        print(f"[Turn {turn_number}/{max_turns}] Urgency: {urgency_level.name} (continuing)")

    print(f"User: {user_message}")
    print()
    print(f"Agent: {agent_response}")
    print()


def get_previous_level(current: UrgencyLevel) -> UrgencyLevel:
    """Get the previous urgency level for display purposes.

    Args:
        current: Current urgency level.

    Returns:
        Previous urgency level in the progression.
    """
    progression = [
        UrgencyLevel.EXPLORATION,
        UrgencyLevel.BALANCED,
        UrgencyLevel.EFFICIENCY,
        UrgencyLevel.TERMINAL,
    ]
    idx = progression.index(current)
    return progression[max(0, idx - 1)]


def simulate_agent_response(
    urgency_level: UrgencyLevel,
    user_message: str,
    turn_context: dict[str, str],
) -> str:
    """Simulate agent responses based on urgency level.

    This function demonstrates how an agent's behavior changes
    as urgency escalates.

    Args:
        urgency_level: Current urgency level.
        user_message: User's input.
        turn_context: Context about the conversation state.

    Returns:
        Simulated agent response.
    """
    match urgency_level:
        case UrgencyLevel.EXPLORATION:
            return (
                "I'd be happy to help debug your TypeError. To provide the most "
                "accurate assistance, could you share:\n\n"
                "1. The full error message including the traceback\n"
                "2. The relevant code snippet where the error occurs\n"
                "3. What the script is supposed to do\n\n"
                "TypeErrors often occur when operations are performed on incompatible "
                "types—common causes include concatenating strings with numbers, "
                "calling methods on None, or passing wrong argument types to functions."
            )

        case UrgencyLevel.BALANCED:
            return (
                "A NoneType error on line 42 typically means you're trying to "
                "call a method or access an attribute on a variable that's None.\n\n"
                "Common fixes:\n"
                "• Add a null check: `if my_var is not None:`\n"
                "• Use optional chaining: `my_var?.method()`\n"
                "• Provide a default: `my_var = get_value() or default_value`\n\n"
                "Can you share the specific line 42 code so I can give you the exact fix?"
            )

        case UrgencyLevel.EFFICIENCY:
            if turn_context.get("turn") == "3":
                return (
                    "The issue is that `data` is None. Add a guard:\n\n"
                    "```python\n"
                    "result = data.process() if data else None\n"
                    "```\n\n"
                    "Also check where `data` is assigned—it may be failing to load."
                )
            else:
                return (
                    "Add error handling around the data loading:\n\n"
                    "```python\n"
                    "try:\n"
                    "    data = load_data()\n"
                    "    result = data.process() if data else None\n"
                    "except Exception as e:\n"
                    "    print(f'Error: {e}')\n"
                    "    result = None\n"
                    "```\n\n"
                    "This handles both the None case and any loading errors."
                )

        case UrgencyLevel.TERMINAL:
            return (
                "Here's a complete solution with logging:\n\n"
                "```python\n"
                "import logging\n"
                "logging.basicConfig(level=logging.INFO)\n\n"
                "data = load_data()\n"
                "if data:\n"
                "    result = data.process()\n"
                "    logging.info(f'Processed: {result}')\n"
                "else:\n"
                "    logging.warning('Data loading failed')\n"
                "    result = None\n"
                "```\n\n"
                "[SESSION END - Turn budget exhausted]\n"
                "For additional features (retries, async loading), start a new session."
            )


def run_demo() -> None:
    """Run the turn-based urgency demo.

    Simulates a 5-turn conversation demonstrating how agent behavior
    shifts from exploratory to urgent as the turn budget is consumed.
    """
    # Setup
    config = TurnUrgencyConfig(max_turns=5)
    middleware = TurnUrgencyMiddleware(config)

    print_header()
    print_config(config)

    # Define conversation flow
    conversation: list[dict[str, str | dict[str, str]]] = [
        {
            "user": "My Python script is broken. It says 'TypeError' when I run it.",
            "context": {"turn": "1"},
        },
        {
            "user": "It's on line 42. Something about 'NoneType'.",
            "context": {"turn": "2"},
        },
        {
            "user": "It's something like result = data.process()",
            "context": {"turn": "3"},
        },
        {
            "user": "That fixed it but now there's another error.",
            "context": {"turn": "4"},
        },
        {
            "user": "What about logging?",
            "context": {"turn": "5"},
        },
    ]

    # Initialize conversation with system prompt
    from langchain_core.messages import BaseMessage
    messages: list[BaseMessage] = [
        SystemMessage(content="You are a helpful coding assistant.")
    ]

    last_urgency: UrgencyLevel | None = None

    # Process each turn
    for turn_num, turn_data_item in enumerate(conversation, start=1):
        user_msg = str(turn_data_item["user"])
        context_raw = turn_data_item["context"]
        context = {str(k): str(v) for k, v in context_raw.items()}  # type: ignore[union-attr]

        # Add user message
        messages.append(HumanMessage(content=user_msg))

        # Process through middleware to get urgency and modified messages
        modified_messages, urgency = middleware.process_messages(messages)

        # Check if this is a transition
        is_transition = last_urgency is not None and last_urgency != urgency.current_level
        last_urgency = urgency.current_level

        # Get the urgency modifier that would be injected
        urgency_modifier = middleware._calculator.get_prompt_modifier(urgency)

        # Simulate agent response based on urgency level
        agent_response = simulate_agent_response(urgency.current_level, user_msg, context)

        # Print the turn
        print_turn(
            turn_num,
            config.max_turns,
            urgency.current_level,
            user_msg,
            agent_response,
            is_transition,
        )

        # Show the prompt modifier that was injected
        print(f"  [Injected modifier: {urgency_modifier[:50]}...]")
        print()

        # Add simulated agent response to history
        messages.append(AIMessage(content=agent_response))

    # Print summary
    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print()
    print("Urgency transitions observed:")
    print("  • EXPLORATION (Turn 1): Asked 3 clarifying questions")
    print("  • BALANCED (Turn 2): Asked 1 targeted follow-up")
    print("  • EFFICIENCY (Turns 3-4): 0 questions, direct solutions")
    print("  • TERMINAL (Turn 5): Immediate delivery, next steps provided")
    print()
    print("Key insight: Agent behavior adapts to remaining turn budget,")
    print("shifting from exploration → efficiency → terminal delivery.")


def main() -> None:
    """Main entry point for the demo."""
    run_demo()


if __name__ == "__main__":
    main()
