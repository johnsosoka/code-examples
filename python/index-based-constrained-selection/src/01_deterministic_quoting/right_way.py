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

from typing import List, Dict
from dataclasses import dataclass
from pydantic import BaseModel, Field
import stanza
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class IndexedSentence:
    """A sentence with its index and text.
    
    Attributes:
        index: The unique index of this sentence (0-based).
        text: The exact text of the sentence.
    """
    index: int
    text: str


class IndexedSentenceRegistry:
    """Registry for managing indexed sentences from a document.
    
    This class handles:
    - Sentence segmentation using stanza NLP
    - Index assignment
    - Deterministic retrieval by index
    
    Example:
        >>> registry = IndexedSentenceRegistry(essay_text)
        >>> print(registry.get_formatted_sentences())
        [0] First sentence here.
        [1] Second sentence here.
        >>> registry.get_sentences_by_indices([0, 2])
        ['First sentence here.', 'Third sentence here.']
    """
    
    def __init__(self, text: str, language: str = "en"):
        """Initialize the registry with text, splitting into sentences.
        
        Args:
            text: The full text to segment into sentences.
            language: The language code for stanza (default: "en" for English).
        """
        self.text = text
        self.language = language
        self.sentences: List[IndexedSentence] = []
        self._split_sentences()
    
    def _split_sentences(self) -> None:
        """Split the text into sentences using stanza NLP.
        
        This uses Stanford's stanza library for robust sentence segmentation
        that handles abbreviations, quotations, and other edge cases better
        than simple regex-based approaches.
        """
        # Initialize stanza pipeline (downloads model if needed)
        nlp = stanza.Pipeline(
            lang=self.language,
            processors='tokenize',
            verbose=False
        )
        
        # Process the text
        doc = nlp(self.text)
        
        # Extract sentences with indices
        self.sentences = [
            IndexedSentence(index=i, text=sentence.text.strip())
            for i, sentence in enumerate(doc.sentences)
        ]
    
    def get_formatted_sentences(self) -> str:
        """Get all sentences formatted with their indices.
        
        Returns:
            A formatted string with each sentence prefixed by its index
            in the format "[0] Sentence text.\n[1] Next sentence."
        """
        lines = [f"[{s.index}] {s.text}" for s in self.sentences]
        return "\n".join(lines)
    
    def get_sentences_by_indices(self, indices: List[int]) -> List[str]:
        """Retrieve exact sentences by their indices.
        
        This is the key method that enables deterministic quoting.
        Given a list of indices, it returns the exact text of those
        sentences from the original document.
        
        Args:
            indices: List of sentence indices to retrieve.
            
        Returns:
            List of sentence texts in the order specified by indices.
            Invalid indices are skipped.
            
        Example:
            >>> registry.get_sentences_by_indices([0, 5, 3])
            ['First sentence.', 'Sixth sentence.', 'Fourth sentence.']
        """
        result = []
        for idx in indices:
            if 0 <= idx < len(self.sentences):
                result.append(self.sentences[idx].text)
        return result
    
    def get_sentence_count(self) -> int:
        """Return the total number of sentences."""
        return len(self.sentences)
    
    def get_sentence_at(self, index: int) -> str:
        """Get a single sentence by index.
        
        Args:
            index: The sentence index.
            
        Returns:
            The sentence text, or empty string if index is invalid.
        """
        if 0 <= index < len(self.sentences):
            return self.sentences[index].text
        return ""


class QuoteSelection(BaseModel):
    """Structured output for index-based quote selection.
    
    The LLM returns this structure containing the indices of sentences
    it deems relevant to the topic, rather than generating the text itself.
    """
    selected_indices: List[int] = Field(
        description="Indices of sentences relevant to the topic (0-based)"
    )
    reasoning: str = Field(
        description="Brief explanation of why these sentences were selected"
    )


def extract_quotes_right_way(
    essay_text: str,
    topic: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0
) -> Dict[str, any]:
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
        - selected_indices: The indices that were selected
        - reasoning: LLM's explanation for the selection
        - registry: The IndexedSentenceRegistry for further use
        
    Example:
        >>> with open("essay.txt") as f:
        ...     essay = f.read()
        >>> result = extract_quotes_right_way(essay, "artificial intelligence")
        >>> print(result['quotes'])
        ['Machine learning algorithms can now analyze medical images...']
        >>> # These quotes are guaranteed to be exact matches!
    """
    # Step 1: Create indexed registry
    registry = IndexedSentenceRegistry(essay_text)
    
    # Step 2: Get formatted sentences for the prompt
    formatted_sentences = registry.get_formatted_sentences()
    
    # Step 3: Initialize LLM with structured output
    llm = ChatOpenAI(model=model, temperature=temperature)
    structured_llm = llm.with_structured_output(QuoteSelection)
    
    # Step 4: Build the prompt
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
    
    # Step 5: Run the chain
    chain = prompt | structured_llm
    selection = chain.invoke({
        "sentences": formatted_sentences,
        "topic": topic
    })
    
    # Step 6: Deterministically retrieve exact sentences by index
    # This is the key step - we get the EXACT text from the registry
    quotes = registry.get_sentences_by_indices(selection.selected_indices)
    
    return {
        "quotes": quotes,
        "selected_indices": selection.selected_indices,
        "reasoning": selection.reasoning,
        "registry": registry
    }


def verify_quotes_deterministic(
    original_text: str,
    quotes: List[str],
    registry: IndexedSentenceRegistry
) -> dict:
    """Verify that quotes are exact matches from the registry.
    
    This function demonstrates the perfect accuracy of the index-based approach.
    Every quote is guaranteed to be an exact substring of the original text.
    
    Args:
        original_text: The original essay text.
        quotes: The list of quotes to verify.
        registry: The IndexedSentenceRegistry used for extraction.
        
    Returns:
        Dictionary with verification results:
        - total_quotes: Number of quotes
        - exact_matches: Always equal to total_quotes (guaranteed)
        - accuracy_score: Always 100.0%
        - all_in_registry: Whether all quotes are in the registry
    """
    results = {
        "total_quotes": len(quotes),
        "exact_matches": 0,
        "accuracy_score": 0.0,
        "all_in_registry": True,
        "details": []
    }
    
    # Build set of all registry sentences for O(1) lookup
    registry_sentences = {s.text for s in registry.sentences}
    
    for quote in quotes:
        is_in_registry = quote in registry_sentences
        is_in_original = quote in original_text
        
        if is_in_registry and is_in_original:
            results["exact_matches"] += 1
            status = "exact_match"
        else:
            results["all_in_registry"] = False
            status = "error"  # This should never happen!
        
        results["details"].append({
            "quote": quote[:100] + "..." if len(quote) > 100 else quote,
            "status": status,
            "in_registry": is_in_registry,
            "in_original": is_in_original
        })
    
    # Calculate accuracy (should always be 100%)
    if results["total_quotes"] > 0:
        results["accuracy_score"] = (results["exact_matches"] / results["total_quotes"]) * 100
    else:
        results["accuracy_score"] = 100.0  # No quotes is technically 100% accurate
    
    return results