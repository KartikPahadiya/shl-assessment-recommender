from fastapi import FastAPI
from .schemas import ChatRequest, ChatResponse, Recommendation
from .graph import graph
from .mappings import build_recommendation

app = FastAPI(
    title="SHL Assessment Agent",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "SHL Assessment Agent API"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    state = {
        "messages": [],
        "user_query": "",
        "intent": "",
        "constraints": None,
        "new_constraints": {},
        "removed_constraints": {},
        "retrieved_docs": [],
        "recommendations": [],
        "reply": "",
        "end_of_conversation": False,
        "safety_violation": False,
        "turn_count": 0,
        "ready_to_recommend": False,
    }

    # Replay conversation turn-by-turn
    for msg in request.messages:
        state["messages"].append({
            "role": msg.role,
            "content": msg.content
        })

        # Only invoke the graph for user messages.
        # Assistant messages are part of history but should not trigger
        # new parsing / routing / retrieval.
        if msg.role == "user":
            state["turn_count"] = state.get("turn_count", 0) + 1
            state = graph.invoke(state)

    recommendations = []

    for rec in state["recommendations"]:
        recommendations.append(build_recommendation(rec))

    return {
        "reply": state["reply"],
        "recommendations": recommendations,
        "end_of_conversation": state["end_of_conversation"]
    }
