"""
WRONG WAY: Free-form quote generation.

This module demonstrates the problematic approach of asking an LLM to
generate quotes directly from source text. The LLM may:
- Paraphrase instead of quoting exactly
- Hallucinate words or phrases that weren't in the original
- Merge multiple sentences incorrectly
- Change punctuation or capitalization

This approach sacrifices verifiability and accuracy for convenience.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class FreeFormQuotes(BaseModel):
    """Output model for free-form quote extraction.
    
    WARNING: This approach allows the LLM to generate or modify text,
    which may result in inaccurate or hallucinated quotes.
    """
    quotes: List[str] = Field(
        description="Relevant quotes extracted from the essay about the topic"
    )
    reasoning: str = Field(
        description="Brief explanation of why these quotes were selected"
    )


def extract_quotes_wrong_way(
    essay_text: str,
    topic: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> FreeFormQuotes:
    """Extract quotes using the WRONG way: free-form generation.
    
    This function demonstrates why you should NOT ask an LLM to generate
    quotes directly. The LLM may paraphrase, hallucinate, or otherwise
    modify the original text.
    
    Args:
        essay_text: The full text of the essay to extract quotes from.
        topic: The topic to find relevant quotes about.
        model: The OpenAI model to use for extraction.
        temperature: Sampling temperature (0 for deterministic output).
        
    Returns:
        FreeFormQuotes object containing the generated quotes and reasoning.
        
    Example:
        >>> with open("essay.txt") as f:
        ...     essay = f.read()
        >>> result = extract_quotes_wrong_way(essay, "artificial intelligence")
        >>> print(result.quotes)
        ['AI is transforming healthcare...']  # May not match original exactly!
    """
    # Initialize the LLM
    llm = ChatOpenAI(model=model, temperature=temperature)
    
    # Create structured output
    structured_llm = llm.with_structured_output(FreeFormQuotes)
    
    # Build the prompt - this is the WRONG approach
    # We're asking the LLM to GENERATE quotes, not select them
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that extracts relevant quotes from essays.
        
Your task is to extract quotes about a specific topic from the provided essay.

IMPORTANT: Extract the exact text as it appears in the essay. Do not paraphrase 
or modify the quotes in any way."""),
        ("human", """Essay:
{essay}

Please extract relevant quotes about: {topic}

Return the quotes exactly as they appear in the text.""")
    ])
    
    # Chain the prompt with the structured LLM
    chain = prompt | structured_llm
    
    # Invoke the chain
    result = chain.invoke({
        "essay": essay_text,
        "topic": topic
    })
    
    return result


def verify_quotes_accuracy(
    original_text: str,
    quotes: List[str]
) -> dict:
    """Verify how accurate the generated quotes are compared to the original.
    
    This helper function demonstrates the problems with free-form generation
    by checking if each quote actually exists in the original text.
    
    Args:
        original_text: The original essay text.
        quotes: The list of quotes to verify.
        
    Returns:
        Dictionary with verification results including:
        - total_quotes: Number of quotes provided
        - exact_matches: Number of quotes found exactly in original
        - partial_matches: Number of quotes found with minor differences
        - not_found: Number of quotes not found in original
        - accuracy_score: Percentage of quotes that match exactly
    """
    results = {
        "total_quotes": len(quotes),
        "exact_matches": 0,
        "partial_matches": 0,
        "not_found": 0,
        "details": []
    }
    
    for quote in quotes:
        # Clean up the quote for comparison
        clean_quote = quote.strip().strip('"').strip("'")
        
        if clean_quote in original_text:
            results["exact_matches"] += 1
            results["details"].append({
                "quote": quote[:100] + "..." if len(quote) > 100 else quote,
                "status": "exact_match"
            })
        elif any(clean_quote[:50] in original_text for _ in [0] if len(clean_quote) >= 50):
            # Check if first 50 chars match (partial match heuristic)
            results["partial_matches"] += 1
            results["details"].append({
                "quote": quote[:100] + "..." if len(quote) > 100 else quote,
                "status": "partial_match"
            })
        else:
            results["not_found"] += 1
            results["details"].append({
                "quote": quote[:100] + "..." if len(quote) > 100 else quote,
                "status": "not_found"
            })
    
    # Calculate accuracy score
    if results["total_quotes"] > 0:
        results["accuracy_score"] = (results["exact_matches"] / results["total_quotes"]) * 100
    else:
        results["accuracy_score"] = 0.0
        
    return results