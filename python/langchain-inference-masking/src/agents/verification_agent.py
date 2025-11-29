"""Verification agent demo - PII masking middleware with verification tool."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import ConfigLoader
from middleware import PiiMaskingMiddleware
from tools import verify_identity

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_verification_demo():
    """Run the verification agent demo with PII masking middleware."""
    config = ConfigLoader()

    model = ChatOpenAI(
        model=config.model_name,
        api_key=config.openai_api_key,
    )

    middleware = PiiMaskingMiddleware()

    system_prompt = (
        "You are a helpful assistant that can verify phone numbers and SSNs. "
        "When the user provides a phone number or SSN and asks for verification, "
        "use the verify_identity tool with the identifier exactly as provided."
    )

    agent = create_agent(
        model=model,
        tools=[verify_identity],
        middleware=[middleware],
        system_prompt=system_prompt,
    )

    user_message = (
        "I need to verify my phone number 555-867-5309. "
        "Please run verification on it."
    )

    logger.info(f"\n{'='*60}")
    logger.info("Running verification agent with PII masking middleware")
    logger.info(f"Original user message: {user_message}")

    result = agent.invoke({"messages": [HumanMessage(content=user_message)]})

    logger.info(f"Agent response: {result['messages'][-1].content}")
    logger.info(f"\nPII Registry: {middleware._mask_registry}")
