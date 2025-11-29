# LangChain PII Masking Middleware

LangChain 1.0 middleware that masks Personally Identifiable Information (PII) during LLM inference. Intercepts messages before they reach the model, replaces detected PII with hash placeholders, and restores original values in responses. Includes examples of both basic masking and integration with agent tools.

## How It Works

```
User Input ──> wrap_model_call() ──> Response
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    Mask PII in           Unmask PII in
    request messages      response messages
         │                       ▲
         └──────> LLM ───────────┘
                    │
         (Tools receive masked values,
          resolve via PiiRegistry singleton)
```

The `PiiMaskingMiddleware` extends `AgentMiddleware` and implements the `wrap_model_call` hook to intercept model requests and responses. When tools are present, they receive masked placeholders and use the global `PiiRegistry` to resolve real values for backend operations.

## Supported PII Types

| Type | Pattern Example |
|------|-----------------|
| Email | `user@example.com` |
| Phone | `555-123-4567`, `(800) 555-0199` |
| SSN | `123-45-6789` |

## Requirements

- Python 3.11+
- Poetry
- OpenAI API key

## Setup

1. Install dependencies:
   ```bash
   poetry install
   ```

2. Configure environment variables in `.env`:
   ```bash
   OPENAI_API_KEY=sk-...

   # Optional: Enable LangSmith tracing
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=langchain-inference-masking
   ```

## Project Structure

```
src/
├── app.py                    # Entry point for demos
├── agents/
│   ├── simple_agent.py       # Demo 1: Basic PII masking (no tools)
│   └── verification_agent.py # Demo 2: PII masking with tool integration
├── services/
│   ├── pii_registry.py       # Global singleton registry for PII mappings
│   └── verification_service.py # Mock backend service for verification demo
├── tools/
│   └── verification_tools.py # verify_identity tool (resolves masked values)
└── middleware/
    └── pii_masking.py        # Core PII masking middleware
```

## Running the Demos

**Demo 1: Basic PII Masking (No Tools)**
```bash
poetry run python src/app.py
```
Shows middleware masking PII in user input and LLM responses without any tool interactions.

**Demo 2: Tool Integration**
```bash
poetry run python src/app.py verify
```
Demonstrates how tools receive masked placeholders, resolve them via `PiiRegistry`, and call backend services with real values.

Example output:
```
verify_identity tool received: [PHONE:59c0b4a6]
Resolved [PHONE:59c0b4a6] -> 555-867-5309 (matched: [PHONE:59c0b4a6])
VerificationService.verify_phone called with: 555-867-5309
```

## LangSmith Tracing

When enabled, LangSmith traces show the middleware behavior:

- **LangGraph model node**: Original user input and unmasked final response
- **ChatOpenAI node**: Masked PII in both input and output (LLM never sees real PII)

## Usage

**Basic Agent (No Tools)**
```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from middleware.pii_masking import PiiMaskingMiddleware

model = ChatOpenAI(model="gpt-4o-mini")
middleware = PiiMaskingMiddleware()

agent = create_agent(
    model=model,
    tools=[],
    middleware=[middleware],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({
    "messages": [HumanMessage(content="My email is test@example.com")]
})
```

**Agent with Tools**
```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from middleware.pii_masking import PiiMaskingMiddleware
from services.pii_registry import PiiRegistry
from tools.verification_tools import create_verification_tool

# Middleware uses global PiiRegistry singleton
middleware = PiiMaskingMiddleware()

# Tool resolves masked values via PiiRegistry.get_instance()
verification_tool = create_verification_tool()

agent = create_agent(
    model=model,
    tools=[verification_tool],
    middleware=[middleware],
    system_prompt="You are an identity verification assistant.",
)

# Tool receives "[PHONE:hash]" and resolves to real value internally
result = agent.invoke({
    "messages": [HumanMessage(content="Verify phone 555-867-5309")]
})
```

## Implementation Details

**Middleware Flow**

The middleware uses the `wrap_model_call` hook to intercept model requests and responses:

```python
def wrap_model_call(
    self,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # Mask PII in request messages
    masked_request = request.override(messages=masked_messages)

    # Call LLM with masked messages
    response = handler(masked_request)

    # Unmask PII in response
    return ModelResponse(result=unmasked_messages, ...)
```

This modifies the `ModelRequest` before it reaches the LLM, ensuring PII never leaves your application.

**Tool Integration Pattern**

When tools are present, the middleware and tools coordinate via the global `PiiRegistry` singleton:

1. **Middleware** masks PII and stores mappings: `registry.register("[PHONE:hash]", "555-867-5309")`
2. **LLM** sees only masked values and generates tool calls with placeholders
3. **Tool** receives `[PHONE:hash]` as input parameter
4. **Tool** resolves via `registry.get_original("[PHONE:hash]")` → `"555-867-5309"`
5. **Tool** calls backend service with real value
6. **Middleware** unmasks response before returning to user

This pattern ensures PII is protected during inference while allowing tools to access real values for backend operations.

## Notes

- In-memory storage: Call `middleware.clear_registry()` between conversations to clear the PII registry
- Deterministic placeholders: SHA-256 hash truncated to 8 characters—same PII value produces the same placeholder
- Production use: For comprehensive PII detection, consider [Microsoft Presidio](https://github.com/microsoft/presidio)
