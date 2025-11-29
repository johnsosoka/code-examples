# LangChain PII Masking Middleware

This example demonstrates a custom LangChain middleware that masks Personally Identifiable Information (PII) during LLM inference. The middleware intercepts messages before they reach the model, replaces detected PII with hash placeholders, and restores original values in responses.

## How It Works

```
User Input ──> before_model() ──> LLM ──> after_model() ──> Response
                   │                           │
                   ▼                           ▼
            Mask PII with              Restore original
            placeholders               values from registry
```

The `PiiMaskingMiddleware` class implements two hooks:

- **`before_model`**: Scans messages for PII patterns, replaces matches with deterministic hash placeholders (e.g., `[EMAIL:a1b2c3d4]`), and stores the mapping
- **`after_model`**: Restores original PII values in the model's response using the stored mapping

## Supported PII Types

| Type | Pattern Example |
|------|-----------------|
| Email | `user@example.com` |
| Phone | `555-123-4567`, `(800) 555-0199` |
| SSN | `123-45-6789` |

## Requirements

- Python 3.11+
- Poetry

## Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Configure your environment:
   ```bash
   cp src/config/config-example.yml src/config/config.yml
   ```

3. Add your OpenAI API key to `src/config/config.yml`

## Running the Example

```bash
cd src
poetry run python app.py
```

The demo shows:
1. PII detection and masking on sample messages
2. The placeholder format used
3. The internal registry mapping placeholders to original values

## Example Output

```
Original input: My email is john.doe@example.com and my phone is 555-123-4567.
Masked (sent to LLM): My email is [EMAIL:a1b2c3d4] and my phone is [PHONE:e5f6g7h8].
PII Registry: {'[EMAIL:a1b2c3d4]': 'john.doe@example.com', '[PHONE:e5f6g7h8]': '555-123-4567'}
```

## Usage in Your Own Code

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from middleware import PiiMaskingMiddleware

model = ChatOpenAI(model="gpt-4o-mini")
middleware = PiiMaskingMiddleware()

agent = create_react_agent(
    model=model,
    tools=[],
    # middleware=[middleware]  # LangChain 1.0+ supports this parameter
)

# For manual middleware invocation (pre-1.0):
state = {"messages": [HumanMessage(content="My email is test@example.com")]}
masked_state = middleware.before_model(state, runtime=None)
# ... invoke agent with masked_state ...
# ... then call middleware.after_model(response_state, runtime=None) ...
```

## Notes

- The middleware uses in-memory storage for the PII registry. Call `middleware.clear_registry()` between conversations if needed.
- Hash placeholders are deterministic—the same PII value always produces the same placeholder within a session.
- This is a demonstration example. For production use, consider using [Microsoft Presidio](https://github.com/microsoft/presidio) for more comprehensive PII detection.
