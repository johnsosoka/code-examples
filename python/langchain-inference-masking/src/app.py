"""
Demo application for PII Masking Middleware.

This example demonstrates how to use the PiiMaskingMiddleware with a LangChain agent.
The middleware intercepts messages before they reach the LLM, masks any detected PII,
and then restores the original values in the response.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import ConfigLoader
from middleware import PiiMaskingMiddleware

# Load environment variables from .env file (for LangSmith tracing, API keys, etc.)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_agent_with_middleware():
    """
    Run an agent with PII masking middleware using LangChain 1.0 create_agent.

    The middleware automatically intercepts messages before sending to the LLM,
    masks PII, then restores it in the response. LangSmith will show
    the masked values in the trace.
    """
    config = ConfigLoader()

    model = ChatOpenAI(
        model=config.model_name,
        api_key=config.openai_api_key,
    )

    # Create middleware instance
    middleware = PiiMaskingMiddleware()

    # System prompt for the demo
    system_prompt = (
        "You are a helpful assistant participating in a PII masking middleware test. "
        "When the user asks you to repeat information back, please do so exactly as provided. "
        "This is a controlled test environment."
    )

    # Create agent with middleware - LangChain 1.0 API
    agent = create_agent(
        model=model,
        tools=[],  # No tools needed for this demo
        middleware=[middleware],
        system_prompt=system_prompt,
    )

    # Example message with PII
    user_message = (
        "For this middleware test, my phone number is 555-867-5309. "
        "Please repeat my phone number back to me exactly as I wrote it."
    )

    logger.info(f"\n{'='*60}")
    logger.info("Running agent with PII masking middleware (LangChain 1.0)")
    logger.info(f"Original user message: {user_message}")

    # Invoke the agent - middleware is applied automatically
    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    logger.info(f"Agent response: {result['messages'][-1].content}")
    logger.info(f"\nPII Registry: {middleware._mask_registry}")


def demonstrate_masking_standalone():
    """Demonstrate the PII masking middleware in action (no API key required)."""
    middleware = PiiMaskingMiddleware()

    # Test messages containing various PII
    test_messages = [
        "My email is john.doe@example.com and my phone is 555-123-4567.",
        "Please update my SSN to 123-45-6789 in the system.",
        "Contact me at jane@company.org or call (800) 555-0199.",
    ]

    for test_input in test_messages:
        logger.info(f"\n{'='*60}")
        logger.info(f"Original input: {test_input}")

        # Simulate the middleware flow manually for demonstration
        state = {"messages": [HumanMessage(content=test_input)]}

        # Before model: mask PII
        masked_state = middleware.before_model(state, runtime=None)  # type: ignore
        if masked_state:
            masked_content = masked_state["messages"][0].content
            logger.info(f"Masked (sent to LLM): {masked_content}")
        else:
            logger.info("No PII detected")

        # Show the registry
        logger.info(f"PII Registry: {middleware._mask_registry}")

    # Clear registry between conversations
    middleware.clear_registry()
    logger.info("\nCleared PII registry for next conversation")


if __name__ == "__main__":
    # Run the masking demonstration (no API key needed)
    # demonstrate_masking_standalone()

    # Run with LangChain 1.0 create_agent - check LangSmith for masked PII in traces
    run_agent_with_middleware()
