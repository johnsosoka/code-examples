"""
RIGHT WAY: Index-based sentence selection.

This module demonstrates the correct approach to quote extraction:
1. Split the text into sentences using stanza (Stanford NLP)
2. Index each sentence with a unique identifier
3. Present indexed sentences to the LLM
4. LLM returns indices of relevant sentences
5. Retrieve exact sentences deterministically by index

This approach guarantees perfect accuracy - you get exactly the text
that was in the original, with no possibility of hallucination or
modification.
"""

from typing import Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
import stanza
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class IndexedSentences:
    """Indexed sentences from a document, ready for LLM selection.

    Attributes:
        sentences: Mapping of index to exact sentence text.
        formatted_context: Pre-built "[0] Sentence..." string for LLM prompts.
    """
    sentences: dict[int, str] = field(default_factory=dict)
    formatted_context: str = ""

    def get(self, indices: list[int]) -> list[str]:
        """Look up sentences by index. Invalid indices are skipped."""
        return [self.sentences[i] for i in indices if i in self.sentences]


def index_sentences(text: str, language: str = "en") -> IndexedSentences:
    """Split text into sentences and return an indexed lookup.

    Uses Stanford's stanza library for robust sentence segmentation
    that handles abbreviations, quotations, and edge cases.

    Args:
        text: The full text to segment.
        language: Stanza language code (default: "en").

    Returns:
        IndexedSentences with a sentence dict and formatted context string.
    """
    nlp = stanza.Pipeline(lang=language, processors="tokenize", verbose=False)
    doc = nlp(text)

    sentences = {i: s.text.strip() for i, s in enumerate(doc.sentences)}
    formatted_context = "\n".join(f"[{i}] {text}" for i, text in sentences.items())

    return IndexedSentences(sentences=sentences, formatted_context=formatted_context)


class SelectedQuote(BaseModel):
    """A single quote selection: an index paired with its reasoning."""
    index: int = Field(
        description="Index of the selected sentence (0-based)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this sentence is relevant"
    )


class QuoteSelection(BaseModel):
    """Structured output for index-based quote selection.

    The LLM returns a list of selections, each pairing a sentence index
    with a reason for its relevance — rather than generating text itself.
    """
    selections: list[SelectedQuote] = Field(
        description="List of selected sentences with per-quote reasoning"
    )


def extract_quotes_right_way(
    essay_text: str,
    topic: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> dict[str, Any]:
    """Extract quotes using the RIGHT way: index-based selection.
    
    This function demonstrates the correct approach to quote extraction:
    1. Split essay into indexed sentences using stanza
    2. Present indexed sentences to LLM
    3. LLM selects relevant indices
    4. Retrieve exact sentences deterministically
    
    This guarantees 100% accuracy - the quotes are retrieved exactly
    as they appear in the original text, with no possibility of
    hallucination or modification.
    
    Args:
        essay_text: The full text of the essay to extract quotes from.
        topic: The topic to find relevant quotes about.
        model: The OpenAI model to use for selection.
        temperature: Sampling temperature (0 for deterministic output).
        
    Returns:
        Dictionary containing:
        - quotes: List of exact quotes from the original text
        - selections: List of SelectedQuote objects (index + reasoning)
        - indexed: The IndexedSentences lookup for verification
    """
    # Step 1: Index the sentences
    indexed = index_sentences(essay_text)

    # Step 2: Initialize LLM with structured output
    llm = ChatOpenAI(model=model, temperature=temperature)
    structured_llm = llm.with_structured_output(QuoteSelection)

    # Step 3: Build the prompt
    # CRITICAL: We ask the LLM to SELECT indices, not generate text
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that selects relevant sentences from essays.

Your task is to identify which sentences are relevant to a given topic.

You will be provided with a list of sentences, each prefixed with an index number
in the format [0], [1], [2], etc.

IMPORTANT: Return ONLY the indices of relevant sentences. Do NOT generate or
modify any text. Simply select the numbers of sentences that are relevant."""),
        ("human", """Here are the sentences from the essay:

{sentences}

Please select the indices of sentences that are relevant to the topic: {topic}

Return the indices as a list of numbers.""")
    ])

    # Step 4: Run the chain
    chain = prompt | structured_llm
    selection: QuoteSelection = chain.invoke({
        "sentences": indexed.formatted_context,
        "topic": topic
    })

    # Step 5: Deterministically retrieve exact sentences by index
    indices = [s.index for s in selection.selections]
    quotes = indexed.get(indices)

    return {
        "quotes": quotes,
        "selections": selection.selections,
        "indexed": indexed
    }


def verify_quotes_deterministic(
    original_text: str,
    quotes: list[str],
    indexed: IndexedSentences,
) -> dict:
    """Verify that quotes are exact matches from the indexed sentences.

    Every quote retrieved via index lookup is guaranteed to be an exact
    substring of the original text.

    Args:
        original_text: The original essay text.
        quotes: The list of quotes to verify.
        indexed: The IndexedSentences used for extraction.

    Returns:
        Dictionary with verification results.
    """
    known_sentences = set(indexed.sentences.values())
    results: dict[str, Any] = {
        "total_quotes": len(quotes),
        "exact_matches": 0,
        "accuracy_score": 0.0,
        "all_in_registry": True,
        "details": [],
    }

    for quote in quotes:
        in_index = quote in known_sentences
        in_original = quote in original_text

        if in_index and in_original:
            results["exact_matches"] += 1
            status = "exact_match"
        else:
            results["all_in_registry"] = False
            status = "error"

        results["details"].append({
            "quote": quote[:100] + "..." if len(quote) > 100 else quote,
            "status": status,
            "in_registry": in_index,
            "in_original": in_original,
        })

    if results["total_quotes"] > 0:
        results["accuracy_score"] = (results["exact_matches"] / results["total_quotes"]) * 100
    else:
        results["accuracy_score"] = 100.0

    return results