"""
Simple agent demo - PII masking middleware without tools.

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
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_simple_demo():
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
