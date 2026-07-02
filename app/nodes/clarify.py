def clarify_node(state):
    constraints = state["constraints"]
    missing = []

    if not constraints["role"]:
        missing.append("role")

    if not constraints["seniority"]:
        missing.append("seniority")

    if not constraints["duration"]:
        missing.append("preferred assessment duration")

    if len(missing) == 0:
        state["reply"] = "Could you refine your requirements?"
        return state
    
    question = ", ".join(missing)

    state["reply"] = (
        f"I need a bit more information. "
        f"Please specify: {question}."
    )

    state["end_of_conversation"] = False
    return state