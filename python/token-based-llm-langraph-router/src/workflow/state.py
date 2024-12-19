from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated


class State(TypedDict):
    """TypedDict for the entire state structure."""
    # The sequence of messages exchanged in the conversation
    user_query: str
    # Annotated is required when branching
    result: Annotated[BaseMessage, "result"]
