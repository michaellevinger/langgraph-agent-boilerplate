"""
================================================================================
Test Suite for LangGraph Agent Boilerplate
================================================================================

This test suite covers:
1. Unit tests for individual components (state, nodes, edges)
2. Integration tests for the complete graph workflow
3. Mock tests to avoid actual API calls
4. Edge cases and error handling

Run tests with:
    pytest test_langgraph_agent.py -v

Run with coverage:
    pytest test_langgraph_agent.py --cov=langgraph_agent_boilerplate --cov-report=html
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Sequence

# Import components from our boilerplate
from langgraph_agent_boilerplate import (
    AgentState,
    create_llm,
    agent_node,
    should_continue,
    create_agent_graph,
)

# Import LangChain/LangGraph types for mocking
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)


# ============================================================================
# FIXTURES - Reusable test data and mocks
# ============================================================================

@pytest.fixture
def sample_state():
    """
    Creates a basic agent state for testing.

    Returns:
        AgentState with a simple conversation
    """
    return {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi! How can I help?")
        ]
    }


@pytest.fixture
def mock_llm():
    """
    Creates a mock LLM that returns predictable responses.

    This avoids making actual API calls during tests.

    Returns:
        Mock LLM with invoke() method
    """
    llm = Mock()
    llm.invoke = Mock(return_value=AIMessage(content="Mocked response"))
    llm.bind_tools = Mock(return_value=llm)
    return llm


@pytest.fixture
def mock_tool():
    """
    Creates a mock tool for testing tool execution.

    Returns:
        Mock tool that can be called by the agent
    """
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "A test tool"
    tool.invoke = Mock(return_value="Tool result")
    return tool


# ============================================================================
# UNIT TESTS - Testing individual components
# ============================================================================

class TestAgentState:
    """Tests for the AgentState TypedDict."""

    def test_state_structure(self, sample_state):
        """Test that state has correct structure."""
        assert "messages" in sample_state
        assert isinstance(sample_state["messages"], Sequence)

    def test_state_with_human_message(self):
        """Test state with a human message."""
        state = {"messages": [HumanMessage(content="Test")]}
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Test"

    def test_state_with_multiple_messages(self):
        """Test state with multiple messages of different types."""
        state = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Answer"),
                HumanMessage(content="Follow-up"),
            ]
        }
        assert len(state["messages"]) == 3
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)

    def test_empty_state(self):
        """Test state with no messages."""
        state = {"messages": []}
        assert len(state["messages"]) == 0


class TestCreateLLM:
    """Tests for the create_llm function."""

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_create_llm_without_tools(self, mock_chat_anthropic):
        """Test LLM creation without tools."""
        mock_instance = Mock()
        mock_chat_anthropic.return_value = mock_instance

        llm = create_llm()

        # Verify ChatAnthropic was called with correct parameters
        mock_chat_anthropic.assert_called_once()
        call_kwargs = mock_chat_anthropic.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-sonnet-20241022"
        assert call_kwargs["temperature"] == 0

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_create_llm_with_tools(self, mock_chat_anthropic, mock_tool):
        """Test LLM creation with tools bound."""
        mock_instance = Mock()
        mock_instance.bind_tools = Mock(return_value=mock_instance)
        mock_chat_anthropic.return_value = mock_instance

        tools = [mock_tool]
        llm = create_llm(tools=tools)

        # Verify bind_tools was called
        mock_instance.bind_tools.assert_called_once_with(tools)


class TestAgentNode:
    """Tests for the agent_node function."""

    def test_agent_node_adds_response(self, sample_state, mock_llm):
        """Test that agent node adds LLM response to messages."""
        original_message_count = len(sample_state["messages"])

        # Call agent node
        result = agent_node(sample_state, mock_llm)

        # Verify LLM was invoked
        mock_llm.invoke.assert_called_once_with(sample_state["messages"])

        # Verify new message was added
        assert len(result["messages"]) == original_message_count + 1
        assert result["messages"][-1].content == "Mocked response"

    def test_agent_node_preserves_history(self, mock_llm):
        """Test that agent node preserves conversation history."""
        state = {
            "messages": [
                HumanMessage(content="First question"),
                AIMessage(content="First answer"),
                HumanMessage(content="Second question"),
            ]
        }

        result = agent_node(state, mock_llm)

        # Verify all original messages are preserved
        assert len(result["messages"]) == 4
        assert result["messages"][0].content == "First question"
        assert result["messages"][1].content == "First answer"
        assert result["messages"][2].content == "Second question"

    def test_agent_node_with_empty_state(self, mock_llm):
        """Test agent node with no prior messages."""
        state = {"messages": []}

        result = agent_node(state, mock_llm)

        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


class TestShouldContinue:
    """Tests for the should_continue conditional edge function."""

    def test_should_continue_with_tool_calls(self):
        """Test that should_continue routes to tools when tool_calls present."""
        # Create AI message with tool calls
        ai_message = AIMessage(content="Using tool")
        ai_message.tool_calls = [{"name": "test_tool", "args": {"arg": "value"}}]

        state = {"messages": [HumanMessage(content="Question"), ai_message]}

        result = should_continue(state)

        assert result == "tools"

    def test_should_continue_without_tool_calls(self):
        """Test that should_continue routes to end without tool calls."""
        state = {
            "messages": [
                HumanMessage(content="Question"),
                AIMessage(content="Final answer"),
            ]
        }

        result = should_continue(state)

        assert result == "end"

    def test_should_continue_with_empty_tool_calls(self):
        """Test routing with empty tool_calls list."""
        ai_message = AIMessage(content="No tools needed")
        ai_message.tool_calls = []  # Empty list

        state = {"messages": [ai_message]}

        result = should_continue(state)

        assert result == "end"

    def test_should_continue_with_tool_message(self):
        """Test that tool messages don't trigger tool routing."""
        # ToolMessage doesn't have tool_calls attribute
        state = {
            "messages": [
                HumanMessage(content="Question"),
                ToolMessage(content="Tool result", tool_call_id="123"),
            ]
        }

        result = should_continue(state)

        assert result == "end"


# ============================================================================
# INTEGRATION TESTS - Testing complete graph workflows
# ============================================================================

class TestCreateAgentGraph:
    """Tests for the complete graph creation and execution."""

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_create_graph_without_tools(self, mock_chat_anthropic):
        """Test graph creation without tools."""
        mock_llm = Mock()
        mock_chat_anthropic.return_value = mock_llm

        graph = create_agent_graph()

        # Verify graph was created
        assert graph is not None

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_create_graph_with_tools(self, mock_chat_anthropic):
        """Test graph creation with tools."""
        from langchain_core.tools import tool

        # Use real tool instead of mock (ToolNode requires real tools)
        @tool
        def test_tool(query: str) -> str:
            """Test tool."""
            return f"Result: {query}"

        mock_llm = Mock()
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_chat_anthropic.return_value = mock_llm

        tools = [test_tool]
        graph = create_agent_graph(tools=tools)

        # Verify graph was created
        assert graph is not None
        # Verify tools were bound
        mock_llm.bind_tools.assert_called_once_with(tools)

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_graph_execution_simple_response(self, mock_chat_anthropic):
        """Test graph execution with simple response (no tools)."""
        # Mock LLM to return simple response
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=AIMessage(content="Hello!"))
        mock_chat_anthropic.return_value = mock_llm

        graph = create_agent_graph()

        # Execute graph
        config = {"configurable": {"thread_id": "test"}}
        result = graph.invoke(
            {"messages": [HumanMessage(content="Hi")]}, config=config
        )

        # Verify result
        assert "messages" in result
        assert len(result["messages"]) == 2  # Human + AI message
        assert result["messages"][-1].content == "Hello!"

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_graph_preserves_conversation_history(self, mock_chat_anthropic):
        """Test that graph maintains state across multiple invocations."""
        mock_llm = Mock()
        # Return different responses for each call
        mock_llm.invoke = Mock(
            side_effect=[
                AIMessage(content="First response"),
                AIMessage(content="Second response"),
            ]
        )
        mock_chat_anthropic.return_value = mock_llm

        graph = create_agent_graph()
        config = {"configurable": {"thread_id": "test-history"}}

        # First interaction
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="First question")]}, config=config
        )

        # Verify first interaction
        assert len(result1["messages"]) == 2  # 1 human + 1 AI message

        # Get current state to continue conversation
        current_state = graph.get_state(config)

        # Second interaction - append to existing conversation
        result2 = graph.invoke(
            {"messages": current_state.values["messages"] + [HumanMessage(content="Second question")]},
            config=config
        )

        # Verify conversation grew
        assert len(result2["messages"]) == 4  # 2 human + 2 AI messages
        assert result2["messages"][0].content == "First question"
        assert result2["messages"][1].content == "First response"
        assert result2["messages"][2].content == "Second question"
        assert result2["messages"][3].content == "Second response"


class TestGraphWithTools:
    """Tests for graph execution with tool usage."""

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_graph_with_tool_execution(self, mock_chat_anthropic):
        """Test complete flow: Agent → Tool → Agent → Response."""
        mock_llm = Mock()

        # First call: LLM requests tool
        tool_call_message = AIMessage(content="Using tool")
        tool_call_message.tool_calls = [
            {
                "name": "test_tool",
                "args": {"input": "test"},
                "id": "call_123",
            }
        ]

        # Second call: LLM processes tool result
        final_message = AIMessage(content="Based on tool result: answer")

        mock_llm.invoke = Mock(side_effect=[tool_call_message, final_message])
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_chat_anthropic.return_value = mock_llm

        # Create a real tool for testing
        from langchain_core.tools import tool

        @tool
        def test_tool(input: str) -> str:
            """Test tool."""
            return f"Result for: {input}"

        # Create graph with tool
        graph = create_agent_graph(tools=[test_tool])

        # Execute
        config = {"configurable": {"thread_id": "tool-test"}}
        result = graph.invoke(
            {"messages": [HumanMessage(content="Use the tool")]}, config=config
        )

        # Verify the tool was executed and final response is correct
        # Check that we have at least a tool message and final AI message
        has_tool_message = any(isinstance(msg, ToolMessage) for msg in result["messages"])
        assert has_tool_message, "Tool should have been executed"

        # Verify final AI response
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "Based on tool result: answer"

        # Verify tool was called with correct args
        assert mock_llm.invoke.call_count == 2  # Called twice: initial + after tool


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_empty_message_content(self, mock_chat_anthropic):
        """Test handling of empty message content."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value=AIMessage(content=""))
        mock_chat_anthropic.return_value = mock_llm

        graph = create_agent_graph()
        config = {"configurable": {"thread_id": "empty-test"}}

        result = graph.invoke(
            {"messages": [HumanMessage(content="")]}, config=config
        )

        # Should handle gracefully
        assert "messages" in result

    def test_should_continue_with_malformed_message(self):
        """Test should_continue with message lacking expected attributes."""
        # Create message without tool_calls
        state = {"messages": [HumanMessage(content="Test")]}

        # Should not crash
        result = should_continue(state)
        assert result == "end"

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_multiple_tool_calls_in_sequence(self, mock_chat_anthropic):
        """Test handling multiple tool calls from a single LLM response."""
        mock_llm = Mock()

        # LLM requests multiple tools
        tool_call_message = AIMessage(content="Using multiple tools")
        tool_call_message.tool_calls = [
            {"name": "tool1", "args": {}, "id": "call_1"},
            {"name": "tool2", "args": {}, "id": "call_2"},
        ]

        final_message = AIMessage(content="All tools completed")

        mock_llm.invoke = Mock(side_effect=[tool_call_message, final_message])
        mock_llm.bind_tools = Mock(return_value=mock_llm)
        mock_chat_anthropic.return_value = mock_llm

        from langchain_core.tools import tool

        @tool
        def tool1() -> str:
            """First tool."""
            return "Result 1"

        @tool
        def tool2() -> str:
            """Second tool."""
            return "Result 2"

        graph = create_agent_graph(tools=[tool1, tool2])
        config = {"configurable": {"thread_id": "multi-tool"}}

        result = graph.invoke(
            {"messages": [HumanMessage(content="Use multiple tools")]},
            config=config,
        )

        # Should handle both tool calls
        # Minimum: Tool results (2) + AI final response
        assert len(result["messages"]) >= 3
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "All tools completed"


# ============================================================================
# PERFORMANCE AND STATE TESTS
# ============================================================================

class TestStateManagement:
    """Tests for state persistence and management."""

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_different_threads_are_isolated(self, mock_chat_anthropic):
        """Test that different thread_ids maintain separate conversations."""
        mock_llm = Mock()
        mock_llm.invoke = Mock(
            side_effect=[
                AIMessage(content="Thread 1 response"),
                AIMessage(content="Thread 2 response"),
            ]
        )
        mock_chat_anthropic.return_value = mock_llm

        graph = create_agent_graph()

        # Thread 1
        config1 = {"configurable": {"thread_id": "thread-1"}}
        result1 = graph.invoke(
            {"messages": [HumanMessage(content="Message to thread 1")]},
            config=config1,
        )

        # Thread 2
        config2 = {"configurable": {"thread_id": "thread-2"}}
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="Message to thread 2")]},
            config=config2,
        )

        # Each thread should have only 2 messages (their own conversation)
        assert len(result1["messages"]) == 2
        assert len(result2["messages"]) == 2
        assert result1["messages"][0].content == "Message to thread 1"
        assert result2["messages"][0].content == "Message to thread 2"

    @patch("langgraph_agent_boilerplate.ChatAnthropic")
    def test_long_conversation_history(self, mock_chat_anthropic):
        """Test handling of long conversation history."""
        mock_llm = Mock()

        # Return incrementing responses
        responses = [AIMessage(content=f"Response {i}") for i in range(10)]
        mock_llm.invoke = Mock(side_effect=responses)
        mock_chat_anthropic.return_value = mock_llm

        graph = create_agent_graph()
        config = {"configurable": {"thread_id": "long-conversation"}}

        # Simulate 10 turns
        result = None
        for i in range(10):
            # Get current state and append new message
            if i == 0:
                messages = [HumanMessage(content=f"Question {i}")]
            else:
                current_state = graph.get_state(config)
                messages = current_state.values["messages"] + [HumanMessage(content=f"Question {i}")]

            result = graph.invoke(
                {"messages": messages},
                config=config,
            )

        # Should have 20 messages (10 human + 10 AI)
        assert len(result["messages"]) == 20


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    # Run tests when file is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
