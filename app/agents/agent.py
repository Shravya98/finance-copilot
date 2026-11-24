from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.services.llm_service import LLM


class AgentState(TypedDict):
    """The data that flows through the graph."""

    # add_messages = "append new messages to the list instead of replacing it"
    messages: Annotated[list[BaseMessage], add_messages]


class FinanceAgent:
    """A LangGraph agent that loops: think -> use tools -> think -> answer."""

    def __init__(self, llm_service: LLM, tools: list):
        self.chain = llm_service.tool_chain(tools)   # prompt | llm.bind_tools(tools)
        self.graph = self._build_graph(tools)

    async def _call_model(self, state: AgentState) -> dict:
        """The 'agent' node: ask the model what to do next."""
        response = await self.chain.ainvoke({"messages": state["messages"]})
        return {"messages": [response]}

    def _build_graph(self, tools: list):
        graph = StateGraph(AgentState)

        graph.add_node("agent", self._call_model)
        graph.add_node("tools", ToolNode(tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
        graph.add_edge("tools", "agent")                       # loop back

        return graph.compile()

    async def run(self, history: list[BaseMessage], question: str) -> str:
        result = await self.graph.ainvoke(
            {"messages": history + [HumanMessage(content=question)]},
            config={"recursion_limit": 10},
        )
        return result["messages"][-1].content








    