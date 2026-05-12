# LangGraph Report Analysis Agent

A multi-node LangGraph agent that analyzes reports using structured outputs and per-node prompts.

## Quick Start

```bash
# Clone
git clone https://github.com/michaellevinger/langgraph-agent-boilerplate.git
cd langgraph-agent-boilerplate

# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure (create .env file)
echo 'ANTHROPIC_API_KEY=your-key-here' > .env
echo 'LANGSMITH_API_KEY=your-langsmith-key' >> .env
echo 'LANGCHAIN_TRACING_V2=true' >> .env
echo 'LANGCHAIN_PROJECT=5x' >> .env

# Run
python example.py
```

## Architecture

```
START
  ↓
[Extract Metrics]   ← prompt: "extract KPIs..."     → ExtractedMetrics
  ↓
[Identify Trends]   ← prompt: "find patterns..."    → TrendAnalysis
  ↓
[Flag Anomalies]    ← prompt: "spot outliers..."     → AnomalyReport
  ↓
[Summarize]         ← prompt: "synthesize..."        → AnalysisSummary
  ↓
END
```

Each node:
1. Has its own **system prompt** (specialized persona)
2. Returns a **Pydantic model** (structured output)
3. Passes structured data to the next node via state

## Key Patterns

### Structured Outputs

```python
from pydantic import BaseModel, Field

class MyOutput(BaseModel):
    result: str = Field(description="The result")
    confidence: str = Field(description="high, medium, or low")

structured_llm = llm.with_structured_output(MyOutput)
response = structured_llm.invoke([...])  # Returns MyOutput instance
```

### Per-Node Prompts

```python
def my_node(state: MyState) -> dict:
    structured_llm = llm.with_structured_output(MyOutput)
    response = structured_llm.invoke([
        SystemMessage(content="You are a specialist in X..."),
        HumanMessage(content=f"Analyze: {state.data}"),
    ])
    return {"my_field": response}
```

### Custom State

```python
class MyState(BaseModel):
    input_data: str = ""
    step_1_result: Step1Output | None = None
    step_2_result: Step2Output | None = None
```

### Building the Graph

```python
graph = StateGraph(MyState)
graph.add_node("step_1", step_1_node)
graph.add_node("step_2", step_2_node)
graph.set_entry_point("step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", END)
app = graph.compile(checkpointer=MemorySaver())
```

## Switching LLM Providers

The example uses Anthropic/Claude. To switch to OpenAI:

```python
# Replace:
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

# With:
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

Set the appropriate env var:
```bash
# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
OPENAI_API_KEY=sk-...
```

## Files

| File | Purpose |
|------|---------|
| `example.py` | Report analysis agent with structured outputs |
| `langgraph_agent_boilerplate.py` | Generic agent boilerplate (ReAct pattern) |
| `test_langgraph_agent.py` | Test suite |
| `requirements.txt` | Dependencies |
| `QUICK_REFERENCE.md` | Cheat sheet for the interview |

## LangSmith

Traces appear at https://smith.langchain.com under the project specified in `LANGCHAIN_PROJECT`.

Each node shows up as a separate span with its prompt, structured output, token usage, and latency.
