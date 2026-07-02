from typing import TypedDict, List, Dict, Optional

class AgentState(TypedDict):
    messages: List[dict]
    user_query: str
    intent: str
    constraints: Dict
    new_constraints: Dict
    removed_constraints: Dict
    retrieved_docs: List[dict]
    recommendations: List[dict]
    reply: str
    end_of_conversation: bool
    safety_violation: bool
    turn_count: int
    ready_to_recommend: bool
