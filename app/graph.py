from langgraph.graph import StateGraph, END
from app.nodes.memory import memory_node
from app.state import AgentState
from app.nodes.context import context_node
from app.nodes.parser import parser_node
from app.nodes.safety import safety_node
from app.nodes.router import router_node
from app.nodes.retrieve import retrieve_node
from app.nodes.compare import compare_node
from app.nodes.completion import completion_node
from app.nodes.formatter import formatter_node
def clarify_node(state):
    state["reply"] = "What role and seniority are you hiring for?"
    state["recommendations"] = []
    state["end_of_conversation"] = False
    return state


def refuse_node(state):
    state["reply"] = (
        "I can only help with SHL Individual Test Solutions assessment selection."
    )
    state["recommendations"] = []
    state["end_of_conversation"] = False
    return state





# def retrieve_node(state):
#     state["reply"] = "Retriever not connected yet."
#     state["recommendations"] = []
#     state["end_of_conversation"] = False
#     return state

builder = StateGraph(AgentState)

builder.add_node("context", context_node)
builder.add_node("parser", parser_node)
builder.add_node("safety", safety_node)
builder.add_node("router", router_node)
builder.add_node("memory", memory_node)
builder.add_node("clarify", clarify_node)
builder.add_node("refuse", refuse_node)
builder.add_node("compare", compare_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("formatter", formatter_node)
builder.add_node("completion", completion_node)

builder.set_entry_point("context")
builder.add_edge("formatter", "completion")
builder.add_edge("completion", END)
builder.add_edge("context", "parser")
builder.add_edge("parser", "memory")
builder.add_edge("memory", "safety")
builder.add_edge("safety", "router")
builder.add_edge("retrieve", "formatter")

def route(state):
    intent = state["intent"]

    if intent == "clarify":
        return "clarify"

    elif intent == "refuse":
        return "refuse"

    elif intent == "compare":
        return "compare"

    elif intent == "retrieve":
        return "retrieve"

    return "clarify"

builder.add_conditional_edges(
    "router",
    route,
    {
        "clarify": "clarify",
        "refuse": "refuse",
        "compare": "compare",
        "retrieve": "retrieve",
    }
)

builder.add_edge("clarify", END)
builder.add_edge("refuse", END)
builder.add_edge("compare", END)


graph = builder.compile()