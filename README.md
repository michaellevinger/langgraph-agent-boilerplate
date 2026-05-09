# LangGraph Agent Boilerplate

A clean, well-documented boilerplate for building AI agents using LangGraph and Claude. Perfect for interviews or learning LangGraph fundamentals.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or
uv pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY="your-api-key-here"

# Optional: Enable LangSmith tracing
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGCHAIN_TRACING_V2=true
```

### 3. Run Example

```bash
python example.py
```

### 4. Run Tests

```bash
pytest test_langgraph_agent.py -v
```

## What's Included

- **`langgraph_agent_boilerplate.py`** - Main boilerplate with extensive documentation
- **`example.py`** - Simple working examples
- **`test_langgraph_agent.py`** - Comprehensive test suite (23 tests)
- **`requirements.txt`** - All dependencies
- **`QUICK_REFERENCE.md`** - Handy cheat sheet

## Basic Usage

```python
from langgraph_agent_boilerplate import create_agent_graph
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

# Define a tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}: sunny, 72°F"

# Create agent with tools
agent = create_agent_graph(tools=[get_weather])

# Run conversation
result = agent.invoke(
    {"messages": [HumanMessage(content="What's the weather in NYC?")]},
    config={"configurable": {"thread_id": "user-123"}}
)

# Get response
print(result["messages"][-1].content)
```

## Architecture

```
START → [Agent] → Needs tools?
            ↓         ↓    ↓
            ↓       Yes   No
            ↓         ↓    ↓
            ↓     [Tools] END
            ↓         ↓
            └─────────┘

State: {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
Persistence: MemorySaver (remembers conversation history)
```

### Key Concepts

1. **State** - The conversation history and context
2. **Nodes** - Steps in the workflow (agent reasoning, tool execution)
3. **Edges** - Connections between nodes (conditional routing)
4. **Checkpointing** - Saves state after each step (enables memory)

## Examples

### Simple Conversation

```python
agent = create_agent_graph()

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 2+2?")]},
    config={"configurable": {"thread_id": "math-1"}}
)
```

### Multi-Turn Conversation

```python
agent = create_agent_graph()
config = {"configurable": {"thread_id": "chat-1"}}

# First message
agent.invoke(
    {"messages": [HumanMessage(content="My name is Alice")]},
    config=config
)

# Follow-up (agent remembers)
result = agent.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config  # Same thread_id!
)
# Response: "Your name is Alice"
```

### Tool Usage

```python
@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

agent = create_agent_graph(tools=[calculate])

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 15 * 23?")]},
    config={"configurable": {"thread_id": "calc-1"}}
)
# Agent will use the calculator tool
```

## Files to Keep

**Essential:**
- `langgraph_agent_boilerplate.py` - Main code
- `test_langgraph_agent.py` - Tests
- `example.py` - Working examples
- `requirements.txt` - Dependencies
- `README.md` - This file
- `QUICK_REFERENCE.md` - Cheat sheet

**Optional:**
- `visualize_graph.py` - Generate graph diagrams
- `agent_graph.png` - Visual representation

**Delete the rest** - All other files are debugging artifacts from setup.

## Testing

```bash
# Run all tests
pytest test_langgraph_agent.py -v

# Run with coverage
pytest test_langgraph_agent.py --cov=langgraph_agent_boilerplate

# Run specific test
pytest test_langgraph_agent.py::TestBasicWorkflow -v
```

## Common Interview Tasks

### Add a New Tool

```python
@tool
def search_web(query: str) -> str:
    """Search the web."""
    # Implementation here
    return f"Results for {query}"

agent = create_agent_graph(tools=[search_web])
```

### Add System Prompt

```python
from langchain_core.messages import SystemMessage, HumanMessage

# Prepend system message to conversation
agent.invoke({
    "messages": [
        SystemMessage(content="You are a helpful coding assistant."),
        HumanMessage(content="Help me debug this code")
    ]
}, config={"configurable": {"thread_id": "dev-1"}})
```

### Streaming Responses

```python
for chunk in agent.stream(
    {"messages": [HumanMessage(content="Tell me a story")]},
    config={"configurable": {"thread_id": "story-1"}}
):
    print(chunk)  # Real-time updates
```

## Model Configuration

Using Claude Sonnet 4.6 by default. To change:

```python
# In langgraph_agent_boilerplate.py, line ~374:
llm = ChatAnthropic(
    model="claude-opus-4-7",  # or claude-haiku-4-5-20251001
    temperature=0,
)
```

## Troubleshooting

### "No module named 'langgraph'"
```bash
pip install -r requirements.txt
```

### "Authentication error"
```bash
# Check API key is set
echo $ANTHROPIC_API_KEY
```

### "Model not found"
Make sure using a valid Claude 4 model:
- `claude-sonnet-4-6` (balanced, default)
- `claude-opus-4-7` (most capable)
- `claude-haiku-4-5-20251001` (fastest, cheapest)

## Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [Claude API Docs](https://docs.anthropic.com/)
- [LangSmith](https://smith.langchain.com/) - Observability platform

## License

MIT - Use freely for interviews, projects, or learning!
