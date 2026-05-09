# LangGraph Quick Reference Card

**Keep this open during your interview for instant lookup!**

---

## Essential Code Snippets

### 1. Basic Agent Setup

```python
from langgraph_agent_boilerplate import create_agent_graph
from langchain_core.messages import HumanMessage

agent = create_agent_graph()
config = {"configurable": {"thread_id": "conv-1"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config=config
)

print(result["messages"][-1].content)
```

### 2. Create a Tool

```python
from langchain_core.tools import tool

@tool
def tool_name(param: str) -> str:
    """
    Tool description - the LLM uses this to decide when to call it.

    Args:
        param: Parameter description

    Returns:
        What the tool returns
    """
    # Implementation
    return f"Result: {param}"
```

### 3. Add Tool to Agent

```python
agent = create_agent_graph(tools=[tool1, tool2, tool3])
```

### 4. Custom Node

```python
def my_node(state: AgentState) -> AgentState:
    """Custom processing node."""
    messages = state["messages"]

    # Do work
    result = process(messages)

    # Return partial state (LangGraph merges it)
    return {"messages": messages + [AIMessage(content=result)]}
```

### 5. Add Node to Graph

```python
# In create_agent_graph():
workflow.add_node("node_name", my_node)
workflow.add_edge("previous_node", "node_name")
```

### 6. Conditional Edge

```python
def route_function(state: AgentState) -> str:
    """Decide next node based on state."""
    if condition:
        return "node_a"
    return "node_b"

workflow.add_conditional_edges(
    "source_node",
    route_function,
    {"node_a": "node_a", "node_b": "node_b"}
)
```

### 7. Test with Mock

```python
from unittest.mock import Mock, patch

@patch("module.ChatAnthropic")
def test_agent(mock_llm):
    mock_llm.return_value.invoke.return_value = AIMessage(content="test")
    agent = create_agent_graph()
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert len(result["messages"]) == 2
```

---

## State Management

### Access State

```python
def my_node(state: AgentState):
    messages = state["messages"]  # Get messages
    last_message = messages[-1]   # Get last message
```

### Update State

```python
def my_node(state: AgentState):
    messages = state["messages"]
    new_message = AIMessage(content="Response")

    # Return partial update (gets merged automatically)
    return {"messages": messages + [new_message]}
```

### Custom State

```python
from typing import TypedDict, Annotated

class CustomState(TypedDict):
    messages: Annotated[list, "Conversation history"]
    user_id: str
    context: dict

# Use it
workflow = StateGraph(CustomState)
```

---

## Message Types

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# User message
HumanMessage(content="User's question")

# AI response
AIMessage(content="AI's answer")

# Tool result
ToolMessage(content="Tool output", tool_call_id="123")
```

---

## Common Patterns

### Pattern: Mock API Call

```python
@tool
def api_call(query: str) -> str:
    """Call external API."""
    # Quick mock for interview
    return f"Mock result for: {query}"

    # Real implementation:
    # response = requests.get(f"https://api.example.com?q={query}")
    # return response.json()["data"]
```

### Pattern: Error Handling

```python
@tool
def safe_tool(param: str) -> str:
    """Tool with error handling."""
    try:
        result = risky_operation(param)
        return result
    except ValueError as e:
        return f"Invalid input: {e}"
    except Exception as e:
        return f"Error occurred: {e}"
```

### Pattern: Multi-turn Conversation

```python
agent = create_agent_graph()
config = {"configurable": {"thread_id": "same-id"}}

# Turn 1
result1 = agent.invoke(
    {"messages": [HumanMessage(content="My name is Alice")]},
    config=config
)

# Turn 2 (remembers context)
result2 = agent.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config  # Same thread_id
)
# Agent responds: "Your name is Alice"
```

### Pattern: Streaming

```python
for chunk in agent.stream(
    {"messages": [HumanMessage(content="Hello")]},
    config=config
):
    print(chunk)  # Real-time updates
```

---

## Debugging

### Print All Messages

```python
for i, msg in enumerate(result["messages"]):
    print(f"{i}: {msg.__class__.__name__}: {msg.content}")
```

### Check if Tool Was Called

```python
last_msg = result["messages"][-2]  # Second to last
if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    print("Tool was called:", last_msg.tool_calls)
else:
    print("No tool call")
```

### Visualize Graph Steps

```python
for step in agent.stream(
    {"messages": [HumanMessage(content="test")]},
    config=config
):
    print("=== STEP ===")
    print(step)
```

---

## LangGraph Architecture

```
StateGraph
├── Nodes (processing steps)
│   ├── Agent Node (LLM calls)
│   ├── Tool Node (function execution)
│   └── Custom Nodes (your logic)
├── Edges (connections)
│   ├── Normal Edge (always A → B)
│   └── Conditional Edge (decide next node)
├── State (memory)
│   └── Persisted by MemorySaver
└── Entry Point (where to start)
```

---

## Key Concepts in 3 Sentences Each

**State**: The agent's memory at any point. In this boilerplate, it's a list of messages. Gets updated by each node and persists across interactions.

**Node**: A processing step that takes state, does work, returns updated state. Examples: calling LLM, executing tool, validating input.

**Edge**: Defines flow between nodes. Normal edges always go A→B, conditional edges decide dynamically based on state.

**Tool**: A Python function the LLM can call. Must be decorated with `@tool` and have clear docstring describing when to use it.

**Checkpointing**: Automatic state saving after each node. Enables conversation history and ability to pause/resume.

---

## Interview Talking Points

### "How does LangGraph work?"

"LangGraph uses a graph structure where nodes are processing steps and edges define flow. State passes between nodes, getting updated as it goes. A conditional edge function decides whether to route to tools or end the conversation."

### "Why use LangGraph vs simple LLM call?"

"LangGraph enables multi-step reasoning, tool use, and stateful conversations. Unlike a single LLM call, it can loop, make decisions, call external functions, and maintain context across interactions."

### "How do tools work?"

"Tools are Python functions that the LLM can request to call. When we bind tools to the LLM, it gets their descriptions. If the LLM includes tool_calls in its response, the ToolNode executes them and returns results for the agent to process."

### "What is the agent's flow?"

"START → Agent (LLM thinks) → Conditional check: if tool needed, go to Tool Node then back to Agent; if not, go to END. State is preserved throughout via MemorySaver checkpointing."

---

## Common Mistakes to Avoid

❌ Forgetting thread_id → No conversation memory
✅ Always use consistent thread_id for related messages

❌ Not returning partial state → Breaking state updates
✅ Return `{"messages": messages + [new_msg]}`

❌ Poor tool descriptions → LLM doesn't know when to use
✅ Write clear, specific docstrings

❌ Not testing incrementally → Debugging nightmare
✅ Test after each addition

❌ Overcomplicating the solution → Wasting time
✅ Start simple, add complexity if needed

---

## Time Savers

### 1. Copy-Paste Starting Template

```python
from langgraph_agent_boilerplate import create_agent_graph
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"

agent = create_agent_graph(tools=[my_tool])
config = {"configurable": {"thread_id": "interview"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Test")]},
    config=config
)

print(result["messages"][-1].content)
```

### 2. Quick Test

```python
# Verify agent works
python -c "from langgraph_agent_boilerplate import create_agent_graph; from langchain_core.messages import HumanMessage; agent = create_agent_graph(); print(agent.invoke({'messages': [HumanMessage(content='hi')]}, {'configurable': {'thread_id': 'test'}})['messages'][-1].content)"
```

### 3. Mock Tool Template

```python
@tool
def mock_tool(param: str) -> str:
    """Mock tool for fast testing."""
    return f"Mock result for: {param}"
```

---

## Keyboard Shortcuts (Common)

**VS Code**:
- `Cmd/Ctrl + D`: Select next occurrence
- `Cmd/Ctrl + /`: Comment line
- `Cmd/Ctrl + Shift + K`: Delete line
- `Opt/Alt + Up/Down`: Move line

**Terminal**:
- `Ctrl + C`: Stop running process
- `Ctrl + L`: Clear screen
- `Up Arrow`: Previous command

---

## Emergency Commands

```bash
# Install missing package
pip install package-name

# Check what's installed
pip list | grep langgraph

# Run quick test
python -c "import langgraph; print('OK')"

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Kill Python process
pkill -9 python
```

---

## Last-Minute Reminders

✅ **Read requirements carefully** - Make sure you understand what's being asked
✅ **Ask clarifying questions** - Better to ask than build wrong thing
✅ **Start simple** - Get something working, then add complexity
✅ **Test incrementally** - Don't write 100 lines before testing
✅ **Explain as you go** - Interviewers want to hear your thinking
✅ **Handle errors gracefully** - Show you think about edge cases
✅ **Stay calm** - You've got a working boilerplate and full documentation

---

**You're ready! Trust the boilerplate, explain clearly, and implement confidently.** 🚀
