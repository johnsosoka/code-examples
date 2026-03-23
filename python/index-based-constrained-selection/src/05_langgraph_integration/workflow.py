"""LangGraph 1.0+ workflow demonstrating index-based constrained selection.

This example shows how to integrate the IndexedRegistry pattern into a
LangGraph workflow using modern LangGraph 1.0+ APIs.
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..common.registry import IndexedRegistry


class QuoteSelectionOutput(BaseModel):
    """Structured output for quote selection."""
    selected_indices: list[int] = Field(
        description="Indices of sentences to quote"
    )
    reasoning: str = Field(
        description="Why these sentences were selected"
    )


class IndexSelectionState(TypedDict):
    """State for the index-based selection workflow."""
    query: str
    essay_text: str
    sentences: list[str]
    selected_indices: list[int]
    selected_quotes: list[str]
    reasoning: str


def chunk_node(state: IndexSelectionState) -> IndexSelectionState:
    """Split essay into indexed sentences."""
    import stanza
    
    # Initialize stanza pipeline (downloads model on first run)
    nlp = stanza.Pipeline(
        lang='en',
        processors='tokenize',
        download_method=stanza.DownloadMethod.REUSE_RESOURCES,
        verbose=False
    )
    
    doc = nlp(state["essay_text"])
    sentences = [sentence.text for sentence in doc.sentences]
    
    return {
        **state,
        "sentences": sentences,
    }


def select_node(state: IndexSelectionState) -> IndexSelectionState:
    """LLM selects sentence indices."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(QuoteSelectionOutput)
    
    # Build context with indexed sentences
    context = "\n".join(
        f"[{i}] {sentence}"
        for i, sentence in enumerate(state["sentences"])
    )
    
    prompt = f"""You are a research assistant that extracts relevant quotes from text.

Given the user's query and the indexed sentences below, select the indices of sentences 
that are most relevant to answer the query.

## User Query:
{state["query"]}

## Indexed Sentences:
{context}

Select the most relevant sentence indices."""
    
    result = structured_llm.invoke(prompt)
    
    return {
        **state,
        "selected_indices": result.selected_indices,
        "reasoning": result.reasoning,
    }


def retrieve_node(state: IndexSelectionState) -> IndexSelectionState:
    """Deterministically retrieve quotes by index."""
    # Create registry and retrieve selected sentences
    registry = IndexedRegistry(
        {i: sentence for i, sentence in enumerate(state["sentences"])}
    )
    
    selected_quotes = []
    for idx in state["selected_indices"]:
        try:
            quote = registry.get(idx)
            selected_quotes.append(quote)
        except IndexError:
            # Skip invalid indices
            continue
    
    return {
        **state,
        "selected_quotes": selected_quotes,
    }


def create_selection_workflow() -> StateGraph:
    """
    Create a LangGraph 1.0+ workflow for index-based quote selection.
    
    Returns:
        Compiled StateGraph ready for invocation
    """
    # Initialize the graph with state schema
    workflow = StateGraph(IndexSelectionState)
    
    # Add nodes
    workflow.add_node("chunk", chunk_node)
    workflow.add_node("select", select_node)
    workflow.add_node("retrieve", retrieve_node)
    
    # Define edges using LangGraph 1.0+ START constant
    workflow.add_edge(START, "chunk")
    workflow.add_edge("chunk", "select")
    workflow.add_edge("select", "retrieve")
    workflow.add_edge("retrieve", END)
    
    return workflow.compile()


if __name__ == "__main__":
    # Example usage
    essay = """
    Artificial intelligence has transformed healthcare delivery in remarkable ways.
    Machine learning algorithms can now detect certain cancers with greater accuracy than human radiologists.
    However, these systems also raise important ethical questions about privacy and bias.
    Healthcare providers must carefully evaluate AI tools before deployment.
    Patient safety should always remain the top priority in medical AI applications.
    """
    
    workflow = create_selection_workflow()
    
    result = workflow.invoke({
        "query": "What are the benefits of AI in healthcare?",
        "essay_text": essay,
        "sentences": [],
        "selected_indices": [],
        "selected_quotes": [],
        "reasoning": "",
    })
    
    print("Selected Quotes:")
    for quote in result["selected_quotes"]:
        print(f"  - {quote}")
    print(f"\nReasoning: {result['reasoning']}")
