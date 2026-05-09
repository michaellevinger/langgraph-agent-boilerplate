"""
Simple example showing how to use the LangGraph agent boilerplate.

Run this after setting environment variables:
    export ANTHROPIC_API_KEY="your-key-here"
    export LANGSMITH_API_KEY="your-key-here"  # Optional
    export LANGCHAIN_TRACING_V2=true  # Optional
"""

from langgraph_agent_boilerplate import create_agent_graph
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Define a simple tool
@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # In a real app, this would call a weather API
    return f"The weather in {location} is sunny and 72°F."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return "Invalid expression"

# Create the agent graph with tools
print("Creating agent with tools...")
graph = create_agent_graph(tools=[get_weather, calculate])

# Example 1: Simple question (no tools needed)
print("\n" + "="*60)
print("Example 1: Simple Question")
print("="*60)
result = graph.invoke(
    {"messages": [HumanMessage(content="What is 2+2?")]},
    config={"configurable": {"thread_id": "example-1"}}
)
print(f"Response: {result['messages'][-1].content}")

# Example 2: Using a tool
print("\n" + "="*60)
print("Example 2: Tool Usage")
print("="*60)
result = graph.invoke(
    {"messages": [HumanMessage(content="What's the weather in Tokyo?")]},
    config={"configurable": {"thread_id": "example-2"}}
)
print(f"Response: {result['messages'][-1].content}")

# Example 3: Conversation with memory
print("\n" + "="*60)
print("Example 3: Conversation Memory")
print("="*60)

# First message
graph.invoke(
    {"messages": [HumanMessage(content="My favorite city is Paris")]},
    config={"configurable": {"thread_id": "example-3"}}
)

# Second message - agent remembers the previous context
result = graph.invoke(
    {"messages": [HumanMessage(content="What's the weather there?")]},
    config={"configurable": {"thread_id": "example-3"}}
)
print(f"Response: {result['messages'][-1].content}")

print("\n" + "="*60)
print("✅ Examples complete!")
print("="*60)
print("\nCheck LangSmith dashboard if you enabled tracing:")
print("https://smith.langchain.com/")
