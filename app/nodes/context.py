def context_node(state):
    if (
        "constraints" not in state
        or state["constraints"] is None
        or not isinstance(state["constraints"], dict)
        or "role" not in state["constraints"]
    ):
        state["constraints"] = {
            "role": None,
            "seniority": None,
            "language": None,
            "duration": None,
            "remote": None,
            "adaptive": None,
            "skills": []
        }

    if "removed_constraints" not in state or state["removed_constraints"] is None:
        state["removed_constraints"] = {}

    if "turn_count" not in state or state["turn_count"] is None:
        state["turn_count"] = 0

    if "ready_to_recommend" not in state or state["ready_to_recommend"] is None:
        state["ready_to_recommend"] = False

    return state
