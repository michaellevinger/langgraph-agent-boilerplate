"""
================================================================================
LangGraph Agent Boilerplate - Complete Guide
================================================================================

WHAT IS LANGGRAPH?
------------------
LangGraph is a framework for building stateful, multi-step AI agents using a
graph-based approach. Think of it like a flowchart where each box (node) is a
step in your AI's reasoning process.

WHY USE LANGGRAPH?
------------------
- Control Flow: Define exactly how your agent makes decisions
- Statefulness: Maintain conversation history and context
- Tool Use: Let your AI call external functions/APIs
- Debugging: See exactly what your agent is doing at each step
- Human-in-the-Loop: Pause for human approval before critical actions

CORE CONCEPTS:
--------------

1. STATE
   - The "memory" of your agent at any point in time
   - In this boilerplate: a list of messages (user questions + AI responses)
   - Think of it like variables that get passed between steps

2. NODES
   - Individual steps in your workflow
   - Examples:
     * Agent Node: Calls the LLM to think/respond
     * Tool Node: Executes external functions (search, calculator, API calls)
   - Each node takes current state → does work → returns updated state

3. EDGES
   - Connections between nodes that define the flow
   - Two types:
     * Normal Edge: Always go from A → B
     * Conditional Edge: Choose next step based on logic (like an if-statement)

4. GRAPH WORKFLOW
   Simple example flow:

   START → Agent (LLM thinks)
           ↓
           Does it need tools?
           ↓             ↓
          YES           NO
           ↓             ↓
        Tool Node      END
           ↓
        Agent (process results)

5. CHECKPOINTING / PERSISTENCE
   - Saves the state after each step
   - Enables conversation history across multiple interactions
   - Like "saving your game" - you can resume where you left off

HOW IT WORKS IN THIS BOILERPLATE:
----------------------------------

1. User sends a message
2. Message gets added to state
3. Agent node calls Claude LLM with all messages
4. LLM responds (either with text or requests to use tools)
5. Conditional edge checks: "Did LLM request tools?"
   - If YES: Route to Tool Node → Execute tools → Back to Agent
   - If NO: End and return response to user
6. State persists for next interaction

COMPARISON TO OTHER APPROACHES:
--------------------------------

Traditional Chatbot:
  User → LLM → Response
  (No tools, no memory between sessions, no complex workflows)

LangChain (without Graph):
  User → Chain of LLM calls → Response
  (Linear, harder to add complex branching logic)

LangGraph (This Boilerplate):
  User → Graph with multiple decision points → Response
  (Full control, can loop, use tools, pause for human input)

================================================================================
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent


# ============================================================================
# 1. STATE MANAGEMENT
# ============================================================================

class AgentState(TypedDict):
    """
    The "memory" of our agent - tracks everything the agent knows.

    In LangGraph, state is passed between nodes. Each node can read from it
    and update it. This creates a shared context across all steps.

    Attributes:
        messages: Complete conversation history including:
                  - HumanMessage: What the user said
                  - AIMessage: What the LLM responded
                  - ToolMessage: Results from tool execution

    WHY THIS MATTERS:
    -----------------
    Without state, the LLM would have no memory. It wouldn't know what you
    asked previously or what tools it already called. State gives it context.

    EXAMPLE STATE:
    {
        "messages": [
            HumanMessage(content="What's the weather in NYC?"),
            AIMessage(content="<calls weather tool>"),
            ToolMessage(content="72°F and sunny"),
            AIMessage(content="It's 72°F and sunny in NYC!")
        ]
    }
    """
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]


# ============================================================================
# 2. LLM SETUP
# ============================================================================

def create_llm(tools: list = None):
    """
    Initialize the Language Model (Claude) that powers the agent.

    WHAT HAPPENS HERE:
    ------------------
    1. Creates a connection to Anthropic's Claude LLM
    2. Configures it with temperature (0 = deterministic, 1 = creative)
    3. If tools are provided, "binds" them to the LLM so it knows they exist

    TOOL BINDING:
    -------------
    When you bind tools, the LLM gets a description of each tool. During
    inference, it can choose to call them by including tool_calls in its
    response. The LLM doesn't execute tools - it just requests them.

    Args:
        tools: List of LangChain Tool objects (optional)
               Each tool is a function the LLM can call

    Returns:
        ChatAnthropic instance ready to generate responses

    EXAMPLE:
        llm = create_llm(tools=[search_tool, calculator_tool])
        # Now Claude knows it can search the web or do math
    """
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",  # Claude Sonnet 4.6 (balanced performance)
        temperature=0,  # 0 = consistent, 1 = creative
        # API key read from ANTHROPIC_API_KEY environment variable
    )

    if tools:
        # "Binding" tools means the LLM gets their schema/description
        # The LLM can then request to call them in its response
        llm = llm.bind_tools(tools)

    return llm


# ============================================================================
# 3. GRAPH COMPONENTS (NODES)
# ============================================================================

def agent_node(state: AgentState, llm, system_prompt: str = None) -> AgentState:
    """
    The "brain" of the agent - where the LLM does its thinking.

    WHAT THIS NODE DOES:
    --------------------
    1. Takes current state (all conversation messages)
    2. Optionally prepends system prompt if provided
    3. Sends them to Claude LLM
    4. Gets back a response (either text or tool requests)
    5. Returns updated state with the new response added

    FLOW:
    -----
    Input State:
        {"messages": [HumanMessage("What is 2+2?")]}

    LLM Processing:
        Claude reads the system prompt (if provided) and question, decides how to respond

    Output State:
        {"messages": [HumanMessage("What is 2+2?"),
                      AIMessage("2+2 equals 4")]}

    WHY RETURN PARTIAL STATE:
    -------------------------
    We return {"messages": messages + [response]} instead of the full state.
    LangGraph merges this with existing state automatically. This is cleaner
    and less error-prone.

    SYSTEM PROMPT:
    --------------
    If provided, the system prompt is prepended to messages for each LLM call.
    It's not stored in state (to avoid duplication) but sent with every request.
    This tells the LLM its role, capabilities, and how to behave.

    Args:
        state: Current agent state with conversation history
        llm: The language model to use for generation
        system_prompt: Optional instructions for the LLM's behavior

    Returns:
        Dictionary with updated messages list
    """
    messages = state["messages"]

    # Prepend system prompt if provided (not stored in state, just sent to LLM)
    if system_prompt:
        messages_with_system = [SystemMessage(content=system_prompt)] + messages
    else:
        messages_with_system = messages

    # Call the LLM with all messages (this is where the magic happens!)
    response = llm.invoke(messages_with_system)

    # Add LLM's response to message history
    # LangGraph will merge this into the full state
    return {"messages": messages + [response]}


# TOOL NODE EXPLANATION:
# ----------------------
# LangGraph provides ToolNode as a prebuilt component. We don't need to
# write it ourselves. Here's what it does:
#
# 1. Looks at the last message from the LLM
# 2. If it contains tool_calls, executes each tool
# 3. Creates ToolMessage objects with results
# 4. Adds them to state
#
# Example:
#   Input: AIMessage(tool_calls=[{"name": "search", "args": {"q": "weather"}}])
#   Tool executes: search(q="weather") → "72°F"
#   Output: ToolMessage(content="72°F", tool_call_id="123")


# ============================================================================
# 4. GRAPH WORKFLOW (EDGES)
# ============================================================================

def should_continue(state: AgentState) -> str:
    """
    Decision function: Determines where to go next in the graph.

    This is a CONDITIONAL EDGE - like an if-statement in your workflow.

    LOGIC:
    ------
    - If LLM wants to use tools → route to "tools" node
    - If LLM gave a final answer → route to "end" (conversation done)

    WHY THIS MATTERS:
    -----------------
    Without this function, the graph wouldn't know when the agent needs to
    call tools vs when it's ready to respond to the user.

    EXAMPLE SCENARIOS:
    ------------------
    Scenario 1: LLM wants to use a tool
        Last message: AIMessage(tool_calls=[{"name": "search", ...}])
        Returns: "tools" → Graph routes to Tool Node

    Scenario 2: LLM has final answer
        Last message: AIMessage(content="The answer is 42")
        Returns: "end" → Graph ends, returns result to user

    Args:
        state: Current agent state

    Returns:
        "tools" if tools should be called, "end" if conversation is complete
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the LLM requested any tool calls
    # tool_calls is a list of tools the LLM wants to use
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # LLM wants to use tools - route to tool execution
        return "tools"

    # LLM provided a final response - we're done
    return "end"


# ============================================================================
# 5. BUILD THE GRAPH
# ============================================================================

def create_agent_graph(tools: list = None, system_prompt: str = None):
    """
    Constructs the complete LangGraph workflow.

    This is where we assemble all pieces into a working agent:
    - Define nodes (agent, tools)
    - Connect them with edges
    - Add decision logic (conditional edges)
    - Enable memory/persistence
    - Configure system prompt for LLM behavior

    GRAPH STRUCTURE (WITH TOOLS):
    ------------------------------

         START
           ↓
       [Agent]  ← Main LLM reasoning (with system prompt)
           ↓
     should_continue()?
           ↓
      ┌────┴────┐
      ↓         ↓
    [Tools]   [END]
      ↓
    [Agent]  ← Process tool results

    GRAPH STRUCTURE (WITHOUT TOOLS):
    ---------------------------------

         START
           ↓
       [Agent]
           ↓
        [END]

    WORKFLOW EXPLANATION:
    ---------------------
    1. User message enters at START
    2. Agent node calls LLM (with system prompt if provided)
    3. Conditional edge checks if tools needed:
       - If YES: Execute tools → back to Agent to process results
       - If NO: End conversation
    4. MemorySaver checkpoints state after each step

    SYSTEM PROMPT:
    --------------
    The system prompt defines the agent's personality, role, and behavior.
    It's sent with every LLM call but not stored in state.

    Example system prompts:
    - "You are a helpful assistant that answers questions concisely."
    - "You are a Python expert. Always provide code examples."
    - "You are a customer service agent. Be polite and professional."

    Args:
        tools: Optional list of tools the agent can use
        system_prompt: Optional instructions for LLM behavior

    Returns:
        Compiled graph (a CompiledGraph object) ready to invoke
    """
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
    )

    # NOTE: System prompts with create_react_agent
    # The system_prompt parameter is kept for API compatibility but not used
    # with the prebuilt agent. To add system prompts, you can:
    # 1. Prepend SystemMessage to the messages list before calling invoke()
    # 2. Or use a custom graph implementation (see git history for examples)
    if system_prompt:
        print(f"Note: System prompt provided but not applied with prebuilt agent.")
        print(f"Consider prepending SystemMessage to your messages list.")

    # Use LangGraph's prebuilt ReAct agent
    # This handles the agent-tool loop correctly for Claude 4
    app = create_react_agent(
        llm,
        tools=tools if tools else [],
        checkpointer=MemorySaver(),  # Enable conversation persistence
    )

    # ========================================================================
    # 7. HUMAN-IN-THE-LOOP (Optional)
    # ========================================================================

    # ADVANCED FEATURE: Pause execution for human approval
    # Useful when you want to review tool calls before they execute
    #
    # Example use cases:
    # - Agent wants to delete data → pause for human confirmation
    # - Agent makes API call that costs money → human approval first
    # - Agent generates content → human reviews before posting
    #
    # To enable, replace the create_react_agent call above with:
    # app = create_react_agent(
    #     llm_with_prompt,
    #     tools=tools if tools else [],
    #     checkpointer=MemorySaver(),
    #     interrupt_before=["tools"],  # Pause before tool execution
    # )
    #
    # Then in your code:
    # 1. Call app.invoke() → execution pauses at tool node
    # 2. Inspect what tools agent wants to call
    # 3. Call app.invoke() again with same config → resumes execution

    return app


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def run_agent_example():
    """
    Example of how to use the agent.

    This demonstrates:
    - Creating the agent
    - Sending messages
    - Handling responses
    - Multi-turn conversations with memory
    """

    # ========================================================================
    # STEP 1: DEFINE TOOLS (if needed)
    # ========================================================================

    # Tools are Python functions decorated with @tool
    # The LLM can call these to perform actions or fetch data
    #
    # Example tool structure:
    #
    # from langchain_core.tools import tool
    #
    # @tool
    # def search_web(query: str) -> str:
    #     """Search the web for information.
    #
    #     Args:
    #         query: The search query
    #
    #     Returns:
    #         Search results as text
    #     """
    #     # Your implementation here
    #     return f"Results for: {query}"
    #
    # @tool
    # def calculator(expression: str) -> str:
    #     """Evaluate a mathematical expression.
    #
    #     Args:
    #         expression: Math expression like "2+2" or "sqrt(16)"
    #
    #     Returns:
    #         The calculated result
    #     """
    #     return str(eval(expression))  # Note: eval is unsafe, just an example
    #
    # tools = [search_web, calculator]

    tools = []  # No tools for this basic example

    # ========================================================================
    # STEP 2: CREATE THE AGENT (with optional system prompt)
    # ========================================================================

    # Optional: Add a system prompt to define agent behavior
    # NOTE: Tools are automatically discovered via bind_tools()
    # This system prompt adds GUIDANCE on how/when to use them
    system_prompt = """You are a helpful AI assistant.
    Keep your responses concise and friendly.
    If you don't know something, say so honestly."""

    agent = create_agent_graph(tools=tools, system_prompt=system_prompt)

    # ========================================================================
    # STEP 3: CONFIGURE CONVERSATION TRACKING
    # ========================================================================

    # thread_id enables conversation memory
    # All messages with same thread_id are part of same conversation
    # Different thread_id = different conversation
    config = {"configurable": {"thread_id": "conversation-1"}}

    # ========================================================================
    # STEP 4: FIRST INTERACTION
    # ========================================================================

    user_input = "Hello! What can you help me with?"

    # Invoke the agent with the user's message
    # This runs the entire graph: START → Agent → END
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config  # Pass config to enable memory
    )

    # Extract and print the agent's response
    final_message = result["messages"][-1]
    print(f"User: {user_input}")
    print(f"Agent: {final_message.content}\n")

    # ========================================================================
    # STEP 5: FOLLOW-UP INTERACTION (demonstrates memory)
    # ========================================================================

    # The agent remembers previous messages because:
    # 1. We use the same thread_id
    # 2. MemorySaver checkpointed the state
    #
    # Uncomment to test:
    #
    # followup = "Can you elaborate on that?"
    # result = agent.invoke(
    #     {"messages": [HumanMessage(content=followup)]},
    #     config=config  # Same thread_id = same conversation
    # )
    # print(f"User: {followup}")
    # print(f"Agent: {result['messages'][-1].content}\n")

    # ========================================================================
    # STEP 6: STREAMING RESPONSES (optional)
    # ========================================================================

    # For real-time output (like ChatGPT typing effect):
    #
    # print("Streaming example:")
    # for chunk in agent.stream(
    #     {"messages": [HumanMessage(content="Tell me a joke")]},
    #     config=config
    # ):
    #     print(chunk)  # Prints each step of the graph execution


def example_with_tools():
    """
    Example showing how to use the agent WITH tools.

    Demonstrates the full agent loop:
    User → Agent → Tool → Agent → Response

    BEST PRACTICE: Hybrid Approach
    -------------------------------
    - Tools are automatically discovered via bind_tools() (clean & simple)
    - System prompt adds usage guidelines and context (control & guidance)
    - This gives you both automatic tool discovery AND custom behavior
    """
    from langchain_core.tools import tool

    # Define a simple calculator tool
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together.

        Args:
            a: First number
            b: Second number

        Returns:
            The product of a and b
        """
        return a * b

    # Create agent with tools and enhanced system prompt
    tools = [multiply]

    # HYBRID APPROACH: Tools are auto-discovered, but we add custom guidance
    system_prompt = """You are a helpful math assistant.

Tool Usage Strategy:
- Use the multiply tool for multiplication operations
- For simple calculations (like 2×3), you can compute directly
- For larger numbers, always use the tool for accuracy
- Explain your reasoning before calling tools

Remember: Always validate that inputs are numbers before using tools."""

    agent = create_agent_graph(tools=tools, system_prompt=system_prompt)

    # Ask a question that requires the tool
    config = {"configurable": {"thread_id": "tool-example"}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is 23 times 47?")]},
        config=config
    )

    # The flow will be:
    # 1. Agent sees question about multiplication
    # 2. Agent requests to use multiply tool
    # 3. Tool executes: multiply(23, 47) = 1081
    # 4. Agent receives tool result
    # 5. Agent formulates final answer: "23 times 47 equals 1081"

    print("Tool Example:")
    print(f"Result: {result['messages'][-1].content}")


# ============================================================================
# DEBUGGING TIPS
# ============================================================================

def debug_agent_state():
    """
    Helper function to inspect agent state at each step.
    Useful for understanding what's happening inside the graph.
    """
    agent = create_agent_graph([])
    config = {"configurable": {"thread_id": "debug"}}

    # Stream shows each node execution
    for step in agent.stream(
        {"messages": [HumanMessage(content="Hello")]},
        config=config
    ):
        print("=== Graph Step ===")
        print(step)
        print()

    # This will show:
    # - agent node execution
    # - State updates after each node
    # - Final state


if __name__ == "__main__":
    print("=" * 80)
    print("LangGraph Agent Boilerplate - Examples")
    print("=" * 80)
    print()

    # Run basic example
    print("EXAMPLE 1: Basic Agent (No Tools)")
    print("-" * 80)
    run_agent_example()

    print()
    print("EXAMPLE 2: Agent with Tools")
    print("-" * 80)
    example_with_tools()

    # Uncomment to see debugging output:
    # print()
    # print("EXAMPLE 3: Debug State")
    # print("-" * 80)
    # debug_agent_state()
