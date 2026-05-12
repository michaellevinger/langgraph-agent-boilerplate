# Quick Reference — LangGraph Structured Outputs

## Setup on Any Computer

```bash
git clone https://github.com/michaellevinger/langgraph-agent-boilerplate.git
cd langgraph-agent-boilerplate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_pt_...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=5x
```

Run: `python example.py`

---

## Core Pattern: Structured Output Node

```python
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

class MyOutput(BaseModel):
    answer: str = Field(description="The answer")
    reasoning: list[str] = Field(description="Step-by-step reasoning")

def my_node(state: MyState) -> dict:
    structured_llm = llm.with_structured_output(MyOutput)
    response = structured_llm.invoke([
        SystemMessage(content="You are an expert at X..."),
        HumanMessage(content=f"Analyze: {state.input_data}"),
    ])
    return {"my_output": response}
```

---

## Core Pattern: State

```python
from pydantic import BaseModel, Field

class MyState(BaseModel):
    input_data: str = ""
    step_1: Step1Output | None = None
    step_2: Step2Output | None = None
    final: FinalOutput | None = None
```

---

## Core Pattern: Build Graph

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

graph = StateGraph(MyState)
graph.add_node("step_1", step_1_node)
graph.add_node("step_2", step_2_node)
graph.add_node("final", final_node)

graph.set_entry_point("step_1")
graph.add_edge("step_1", "step_2")
graph.add_edge("step_2", "final")
graph.add_edge("final", END)

app = graph.compile(checkpointer=MemorySaver())
```

---

## Core Pattern: Run It

```python
config = {"configurable": {"thread_id": "session-1"}}
result = app.invoke(MyState(input_data="..."), config=config)
print(result["final"].answer)
```

---

## Conditional Edge (Branching)

```python
def route(state: MyState) -> str:
    if state.step_1.needs_review:
        return "review"
    return "finalize"

graph.add_conditional_edges("step_1", route, {
    "review": "review_node",
    "finalize": "final_node",
})
```

---

## Switch Provider to OpenAI

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# .with_structured_output() works the same way
```

Env var: `OPENAI_API_KEY=sk-...`

---

## Switch Provider to Google

```python
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
```

Env var: `GOOGLE_API_KEY=...`

---

## Imports Cheat Sheet

```python
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic       # or ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
```

---

## Tool-Based Agent (Alternative Pattern)

```python
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def my_tool(param: str) -> str:
    """Description the LLM reads to decide when to call this."""
    return f"Result: {param}"

app = create_react_agent(llm, tools=[my_tool], checkpointer=MemorySaver())
result = app.invoke(
    {"messages": [HumanMessage(content="Do something")]},
    config={"configurable": {"thread_id": "t1"}}
)
```

---

## Debugging

```python
# Print structured output
print(result["my_field"].model_dump_json(indent=2))

# Stream to see each node execute
for step in app.stream(MyState(input_data="..."), config=config):
    print(step)
```

---

## Key Interview Points

- **Why structured outputs?** Guarantees schema compliance. No parsing needed. Downstream nodes get typed data.
- **Why per-node prompts?** Each node is a specialist. Focused prompts = better results than one mega-prompt.
- **Why StateGraph over ReAct?** Deterministic flow. You control the order. Easier to debug and test.
- **Why LangGraph?** State management, checkpointing, conditional routing, streaming, observability via LangSmith.
