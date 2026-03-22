# Tool-Triggered Phase Steering

## wrap_model_call -- Inject Phase Guidance

```mermaid
flowchart TD
    A[/"wrap_model_call receives request"/] --> B["Look up PhaseBehavior<br/>for current phase"]
    B --> C["Build guidance: phase name,<br/>allowed tools, completion criteria"]
    C --> D["Inject &lt;framework_instruction&gt;<br/>into last HumanMessage"]
    D --> E[Call LLM]
    E --> F["Increment iteration counters"]
    F --> G[Return state updates]

    style D fill:#f9e2af,color:#1e1e2e
```

## wrap_tool_call -- Gate Tools and Trigger Transitions

```mermaid
flowchart TD
    A[/"wrap_tool_call receives request"/] --> B{"Transition<br/>pending?"}

    B -- Yes --> C["Block tool: return<br/>&lt;framework_instruction&gt;<br/>report to user first"]

    B -- No --> D{"Tool in phase<br/>allow-list?"}
    D -- No --> E["Block tool: return error<br/>listing permitted tools"]

    D -- Yes --> F[Execute tool]
    F --> G{"Trigger tool<br/>for this phase?"}

    G -- Yes --> H["Advance phase<br/>Set transition_pending = true"]
    H --> I["Append &lt;framework_instruction&gt;<br/>transition notice to result"]

    G -- No --> J[Return tool result]
    I --> J

    style B fill:#89b4fa,color:#1e1e2e
    style D fill:#89b4fa,color:#1e1e2e
    style G fill:#89b4fa,color:#1e1e2e
    style C fill:#f38ba8,color:#1e1e2e
    style E fill:#f38ba8,color:#1e1e2e
    style I fill:#f9e2af,color:#1e1e2e
```

## Phase Progression

```mermaid
stateDiagram-v2
    [*] --> DISCOVERY
    DISCOVERY --> CURATION : note_preferences called
    CURATION --> RECOMMENDATION : shortlist_items called
    RECOMMENDATION --> COMPLETE : present_recommendation called
    COMPLETE --> [*]

    DISCOVERY : browse_catalog, note_preferences
    CURATION : filter_products, compare_options, shortlist_items
    RECOMMENDATION : present_recommendation
    COMPLETE : (no tools)
```

## Lifecycle Note

The `before_agent` hook clears `transition_pending` at the start of each agent
invocation, reopening the tool gate for the new phase.
