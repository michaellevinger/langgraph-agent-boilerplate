"""
Report Analysis Agent with Structured Outputs and Per-Node Prompts.

This demonstrates:
1. Structured outputs (Pydantic models) at each node
2. Per-node prompts — each node has its own system prompt
3. A deterministic graph flow: extract → trends → anomalies → summarize

Run this after setting environment variables (or put them in .env):
    export ANTHROPIC_API_KEY="your-key-here"
    export LANGSMITH_API_KEY="your-key-here"  # Optional
    export LANGCHAIN_TRACING_V2=true  # Optional
"""

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


# =============================================================================
# 1. STRUCTURED OUTPUTS — Define what each node returns
# =============================================================================

class ExtractedMetrics(BaseModel):
    """Structured output from the metrics extraction node."""
    revenue: str = Field(description="Revenue figure and change")
    churn_rate: str = Field(description="Customer churn rate")
    new_signups: str = Field(description="New signup count")
    key_metrics: list[str] = Field(description="Other notable KPIs")


class TrendAnalysis(BaseModel):
    """Structured output from the trend identification node."""
    trends: list[str] = Field(description="List of identified trends")
    direction: str = Field(description="Overall direction: improving, declining, or mixed")
    confidence: str = Field(description="Confidence level: high, medium, or low")


class AnomalyReport(BaseModel):
    """Structured output from the anomaly detection node."""
    warnings: list[str] = Field(description="Items that need attention")
    critical: list[str] = Field(description="Items that need immediate action")
    info: list[str] = Field(description="Informational notes")


class AnalysisSummary(BaseModel):
    """Final structured summary combining all analysis."""
    executive_summary: str = Field(description="2-3 sentence executive summary")
    top_action_items: list[str] = Field(description="Prioritized action items")
    overall_health: str = Field(description="Overall health: healthy, at-risk, or critical")


# =============================================================================
# 2. STATE — Holds report text + structured results from each node
# =============================================================================

class ReportAnalysisState(BaseModel):
    """
    State flows through the graph. Each node reads what it needs and
    writes its structured output back into state.
    """
    messages: list = Field(default_factory=list)
    report_text: str = ""
    metrics: ExtractedMetrics | None = None
    trends: TrendAnalysis | None = None
    anomalies: AnomalyReport | None = None
    summary: AnalysisSummary | None = None


# =============================================================================
# 3. NODES — Each has its own prompt and structured output
# =============================================================================

# Shared LLM instance
llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def extract_metrics_node(state: ReportAnalysisState) -> dict:
    """
    NODE 1: Extract Metrics
    PROMPT: Focused on pulling numbers and KPIs from raw text.
    OUTPUT: ExtractedMetrics (structured)
    """
    structured_llm = llm.with_structured_output(ExtractedMetrics)

    response = structured_llm.invoke([
        SystemMessage(content=(
            "You are a metrics extraction specialist. "
            "Given a report, extract all key numerical metrics. "
            "Focus on: revenue, growth rates, churn, signups, and any other KPIs. "
            "Be precise — include percentages and comparisons where available."
        )),
        HumanMessage(content=f"Extract metrics from this report:\n\n{state.report_text}"),
    ])

    return {"metrics": response}


def identify_trends_node(state: ReportAnalysisState) -> dict:
    """
    NODE 2: Identify Trends
    PROMPT: Focused on pattern recognition across the extracted metrics.
    OUTPUT: TrendAnalysis (structured)
    """
    structured_llm = llm.with_structured_output(TrendAnalysis)

    response = structured_llm.invoke([
        SystemMessage(content=(
            "You are a trend analysis specialist. "
            "Given extracted metrics from a report, identify patterns and trends. "
            "Consider: quarter-over-quarter changes, acceleration/deceleration, "
            "correlations between metrics. State your confidence level."
        )),
        HumanMessage(content=(
            f"Report:\n{state.report_text}\n\n"
            f"Extracted metrics:\n{state.metrics.model_dump_json(indent=2)}"
        )),
    ])

    return {"trends": response}


def flag_anomalies_node(state: ReportAnalysisState) -> dict:
    """
    NODE 3: Flag Anomalies
    PROMPT: Focused on spotting outliers and concerns.
    OUTPUT: AnomalyReport (structured)
    """
    structured_llm = llm.with_structured_output(AnomalyReport)

    response = structured_llm.invoke([
        SystemMessage(content=(
            "You are an anomaly detection specialist. "
            "Given a report and its metrics, flag anything unusual: "
            "spending mismatches, sudden spikes/drops, targets missed, "
            "or metrics that don't add up. Categorize as critical, warning, or info."
        )),
        HumanMessage(content=(
            f"Report:\n{state.report_text}\n\n"
            f"Metrics:\n{state.metrics.model_dump_json(indent=2)}\n\n"
            f"Trends:\n{state.trends.model_dump_json(indent=2)}"
        )),
    ])

    return {"anomalies": response}


def summarize_node(state: ReportAnalysisState) -> dict:
    """
    NODE 4: Summarize
    PROMPT: Synthesizes all prior analysis into actionable summary.
    OUTPUT: AnalysisSummary (structured)
    """
    structured_llm = llm.with_structured_output(AnalysisSummary)

    response = structured_llm.invoke([
        SystemMessage(content=(
            "You are an executive briefing specialist. "
            "Synthesize the metrics, trends, and anomalies into a concise "
            "executive summary. Prioritize action items by urgency. "
            "Assess overall health as: healthy, at-risk, or critical."
        )),
        HumanMessage(content=(
            f"Metrics:\n{state.metrics.model_dump_json(indent=2)}\n\n"
            f"Trends:\n{state.trends.model_dump_json(indent=2)}\n\n"
            f"Anomalies:\n{state.anomalies.model_dump_json(indent=2)}"
        )),
    ])

    return {"summary": response}


# =============================================================================
# 4. BUILD THE GRAPH — Deterministic flow with per-node prompts
# =============================================================================
#
#   START
#     ↓
#   [Extract Metrics]   ← prompt: "extract KPIs..."
#     ↓
#   [Identify Trends]   ← prompt: "find patterns..."
#     ↓
#   [Flag Anomalies]    ← prompt: "spot outliers..."
#     ↓
#   [Summarize]         ← prompt: "synthesize..."
#     ↓
#    END
#

def create_report_analysis_graph():
    """Build the report analysis graph with structured outputs at each node."""
    graph = StateGraph(ReportAnalysisState)

    # Add nodes
    graph.add_node("extract_metrics", extract_metrics_node)
    graph.add_node("identify_trends", identify_trends_node)
    graph.add_node("flag_anomalies", flag_anomalies_node)
    graph.add_node("summarize", summarize_node)

    # Define edges — deterministic linear flow
    graph.set_entry_point("extract_metrics")
    graph.add_edge("extract_metrics", "identify_trends")
    graph.add_edge("identify_trends", "flag_anomalies")
    graph.add_edge("flag_anomalies", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile(checkpointer=MemorySaver())


# =============================================================================
# 5. RUN IT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Report Analysis Agent — Structured Outputs")
    print("=" * 60)

    app = create_report_analysis_graph()

    quarterly_report = """
    Q3 2025 Performance Report - Acme SaaS

    Revenue hit $2.4M this quarter, up from $2.14M last quarter. We closed 3 new
    enterprise deals but saw some softness in mid-market. Customer churn dropped
    to 3.1% from 4.2% after the onboarding revamp. We added 1,847 new signups.
    Marketing spend was $480K (up from $340K). Support tickets doubled in week 6
    due to the billing migration. NPS is at 72, continuing its upward trajectory.
    """

    config = {"configurable": {"thread_id": "report-1"}}

    # Run the full analysis pipeline
    result = app.invoke(
        ReportAnalysisState(report_text=quarterly_report.strip()),
        config=config,
    )

    # Print structured results from each node
    print("\n--- Extracted Metrics ---")
    print(f"  Revenue: {result['metrics'].revenue}")
    print(f"  Churn: {result['metrics'].churn_rate}")
    print(f"  Signups: {result['metrics'].new_signups}")
    print(f"  Other KPIs: {result['metrics'].key_metrics}")

    print("\n--- Trend Analysis ---")
    print(f"  Direction: {result['trends'].direction}")
    print(f"  Confidence: {result['trends'].confidence}")
    for trend in result["trends"].trends:
        print(f"  • {trend}")

    print("\n--- Anomalies ---")
    for item in result["anomalies"].critical:
        print(f"  🔴 CRITICAL: {item}")
    for item in result["anomalies"].warnings:
        print(f"  🟡 WARNING: {item}")
    for item in result["anomalies"].info:
        print(f"  🔵 INFO: {item}")

    print("\n--- Executive Summary ---")
    print(f"  Health: {result['summary'].overall_health}")
    print(f"  Summary: {result['summary'].executive_summary}")
    print(f"  Action items:")
    for item in result["summary"].top_action_items:
        print(f"    → {item}")
