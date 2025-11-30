"""Talent Demand Analyst Agent

Analyzes talent demand trends, workforce planning, skills requirements, and labor market insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


class Context(TypedDict, total=False):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    """

    model: str
    """The model to use for the agent. Default: claude-3-5-sonnet-20241022"""


@dataclass
class State(MessagesState):
    """Input state for the Talent Demand Analyst agent."""

    pass


SYSTEM_PROMPT = """You are a Talent Demand Analyst, an expert in analyzing talent demand trends, workforce planning, skills requirements, and labor market insights.

Your capabilities include:
- Analyzing talent demand trends across industries and job roles
- Identifying emerging skills and competencies in the labor market
- Providing workforce planning insights and recommendations
- Assessing skills gaps and talent shortages
- Analyzing labor market data and employment trends

When responding:
1. Be concise and data-driven in your analysis
2. Provide actionable insights and recommendations
3. Cite specific trends, skills, or market dynamics when possible
4. If you need more information to provide accurate analysis, ask clarifying questions
5. Structure your responses clearly with bullet points or sections when appropriate

Current focus areas:
- Software engineering and technology roles
- AI/ML and data science positions
- Healthcare workforce
- Financial services talent
- Manufacturing and industrial roles"""


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, list[BaseMessage]]:
    """Process user input and generate talent demand analysis."""

    # Get model from context or use default
    model_name = (runtime.context or {}).get("model", "claude-3-5-sonnet-20241022")

    # Initialize the model
    model = ChatAnthropic(model=model_name, temperature=0.7)

    # Prepare messages with system prompt
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # Generate response
    response = await model.ainvoke(messages)

    return {"messages": [response]}


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node("call_model", call_model)
    .add_edge("__start__", "call_model")
    .compile(name="Talent Demand Analyst")
)
