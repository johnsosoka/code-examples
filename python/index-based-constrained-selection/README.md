# Index-Based Constrained Selection: LLM Safety Patterns

> **"Don't make the LLM generate the solution; make it select from pre-validated solutions, then execute deterministically."**

This repository demonstrates a general safety pattern for Large Language Models: **constraining outputs to indices into pre-validated registries** rather than free-form generation.

## The Pattern

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Domain Expert  │────▶│  Indexed Registry│────▶│  LLM Selection  │
│  (Pre-validation)│    │  (Validated Set) │     │  (Indices Only) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                           │
                              │                           │
                              ▼                           ▼
                       ┌─────────────────────────────────────────┐
                       │  DETERMINISTIC EXECUTION                │
                       │  - No hallucination in critical path    │
                       │  - Guaranteed valid outputs             │
                       │  - Full audit trail                     │
                       └─────────────────────────────────────────┘
```

## Examples

This repository contains 5 examples showing the **WRONG way** (free-form generation) vs. the **RIGHT way** (index-based selection), plus a LangGraph integration:

### 1. Deterministic Quoting ([`src/01_deterministic_quoting/`](src/01_deterministic_quoting/))

**Problem:** LLMs paraphrase or hallucinate when asked to generate quotes.

**Solution:** Index sentences using Stanford NLP, have LLM select indices, retrieve exact text deterministically.

```bash
python -m src.01_deterministic_quoting.demo
```

**Key Insight:** *Never ask an LLM to generate text when you need exact quotes.*

---

### 2. Web Automation ([`src/02_web_automation/`](src/02_web_automation/))

**Problem:** LLM-generated XPath selectors are brittle and hallucination-prone.

**Solution:** Present accessibility tree with indexed elements, LLM selects by index, system retrieves validated selector.

```bash
python -m src.02_web_automation.demo
```

**Key Insight:** *The LLM never sees or generates XPaths—only selects from a controlled registry.*

---

### 3. Text-to-SQL ([`src/03_text_to_sql/`](src/03_text_to_sql/))

**Problem:** Free-form SQL generation risks injection attacks and syntax errors.

**Solution:** Pre-validated query templates with type-safe parameters. LLM selects template index, system validates and renders.

```bash
python -m src.03_text_to_sql.demo
```

**Key Insight:** *Impossible to generate dangerous SQL when the LLM can only select from safe templates.*

---

### 4. Configuration Selection ([`src/04_config_selection/`](src/04_config_selection/))

**Problem:** LLM-generated configs have invalid values, missing fields, and hallucinated options.

**Solution:** Pydantic-validated config templates. LLM selects by use case, system guarantees valid configuration.

```bash
python -m src.04_config_selection.demo
```

**Key Insight:** *Separate reasoning ("which config fits?") from generation ("write me a config").*

---

### 5. LangGraph Integration ([`src/05_langgraph_integration/`](src/05_langgraph_integration/))

**Problem:** How do you integrate index-based selection into an agentic workflow?

**Solution:** A LangGraph 1.0+ workflow with three nodes: chunk (sentence splitting), select (LLM picks indices), and retrieve (deterministic lookup).

```bash
python -m src.05_langgraph_integration.workflow
```

**Key Insight:** *The pattern composes naturally into graph-based pipelines — each node has a single responsibility.*

---

## Installation

```bash
# Clone the repository
git clone https://github.com/johnsosoka/code-examples.git
cd code-examples/python/index-based-constrained-selection

# Install dependencies with Poetry
poetry install

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run any example
poetry run python -m src.01_deterministic_quoting.demo
```

### Requirements

- Python 3.11+
- Poetry
- OpenAI API key

## Project Structure

```
index-based-constrained-selection/
├── pyproject.toml              # Poetry dependencies
├── README.md                   # This file
├── src/
│   ├── common/                 # Shared utilities
│   │   ├── registry.py         # IndexedRegistry generic class
│   │   └── models.py           # Pydantic models for structured outputs
│   ├── 01_deterministic_quoting/
│   │   ├── wrong_way.py        # Free-form quote generation
│   │   ├── right_way.py        # Index-based sentence selection
│   │   ├── demo.py             # Comparison runner
│   │   └── sample_essay.txt    # Test data
│   ├── 02_web_automation/
│   │   ├── wrong_way.py        # XPath generation
│   │   ├── right_way.py        # Element index selection
│   │   ├── mock_page.py        # Test web page
│   │   └── demo.py
│   ├── 03_text_to_sql/
│   │   ├── wrong_way.py        # Free-form SQL generation
│   │   ├── right_way.py        # Template selection
│   │   ├── schema.py           # DB schema & templates
│   │   └── demo.py
│   ├── 04_config_selection/
│   │   ├── wrong_way.py        # Free-form config generation
│   │   ├── right_way.py        # Config template selection
│   │   ├── config_models.py    # Pydantic config models
│   │   └── demo.py
│   └── 05_langgraph_integration/
│       └── workflow.py          # LangGraph workflow integration
└── tests/                      # Test suite
```

## Common Components

### IndexedRegistry

The core pattern implementation—a generic registry for index-based selection:

```python
from src.common.registry import IndexedRegistry

# Create registry with pre-validated items
registry = IndexedRegistry({
    0: "Option A",
    1: "Option B",
    2: "Option C",
})

# LLM sees only indices and descriptions
context = registry.get_context()
# {0: "Option A", 1: "Option B", 2: "Option C"}

# LLM selects index (e.g., returns 1)
# System retrieves deterministically
item = registry.get(1)  # "Option B"
```

### Structured Output Models

All examples use Pydantic models with LangChain's `with_structured_output()`:

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class SelectionResult(BaseModel):
    selected_index: int = Field(description="Selected index")
    reasoning: str = Field(description="Why this was selected")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(SelectionResult)

result = structured_llm.invoke("Select the best option from: [0] A, [1] B, [2] C")
# result.selected_index is guaranteed to be an integer
```

## When to Use This Pattern

### ✅ Use When:

- **Safety-critical**: Healthcare, finance, legal (exact quotes matter)
- **High failure cost**: Production automation, customer-facing systems
- **Validation is expensive**: Security review required for each item
- **Domain is bounded**: Finite set of valid options
- **Maintenance burden**: Free-form outputs require frequent updates

### ❌ Don't Use When:

- **Creativity required**: Creative writing, brainstorming
- **Unbounded domain**: Open-ended research, novel problem-solving
- **Rapid prototyping**: Speed matters more than correctness
- **User expects natural language**: Chat interfaces
- **Registry would be huge**: Millions of items

## Comparison: Free-Form vs. Index-Based

| Aspect | Free-Form Generation | Index-Based Selection |
|--------|---------------------|----------------------|
| **Hallucination Risk** | High | Zero (in selected item) |
| **Syntax Errors** | Common | Impossible |
| **Security Validation** | Post-hoc (hard) | Pre-hoc (easy) |
| **Maintenance** | High (brittle) | Low (registry updates) |
| **Flexibility** | Unlimited | Constrained to registry |
| **Debuggability** | Hard | Easy (index X was selected) |

## References

- [Deterministic Quoting: Making LLMs Safer for Healthcare](https://mattyyeung.github.io/deterministic-quoting) - Matt Yeung
- [Playwright MCP](https://github.com/microsoft/playwright-mcp) - Microsoft's accessibility tree approach
- [PICARD](https://arxiv.org/abs/2109.05093) - Constrained decoding for text-to-SQL
- [RouteLLM](https://github.com/lm-sys/RouteLLM) - Model routing by index selection

## License

MIT License - See [LICENSE](LICENSE) for details.

---

*Built by [John Sosoka](https://johnsosoka.com) for educational purposes.*

*Remember: "The safest code is the code that never runs—or the code that can't possibly be wrong."*
