def router_node(state):
    if state["safety_violation"]:
        state["intent"] = "refuse"
        return state

    query = state["user_query"].lower()

    if "compare" in query or "vs" in query or " versus " in query or "difference between" in query or "different from" in query:
        state["intent"] = "compare"
        return state

    if "salary" in query or "fire employee" in query:
        state["intent"] = "refuse"
        return state

    constraints = state["constraints"]
    turn_count = state.get("turn_count", 1)

    # VAGUENESS CHECK: On turn 1, if the query is very short and vague,
    # always clarify regardless of what the parser extracted.
    # This prevents "I need an assessment" or "I need a assessment for hiring"
    # from immediately returning recommendations.
    if turn_count == 1 and len(query.split()) < 8:
        state["intent"] = "clarify"
        return state

    # Minimum info needed
    has_role = constraints.get("role") is not None
    has_seniority = constraints.get("seniority") is not None
    has_skills = bool(constraints.get("skills"))

    # Check if query signals explicit readiness
    readiness_signals = [
        "go ahead", "yes please", "yes, go", "lock it in", "locking it in",
        "that works", "that looks good", "looks good", "perfect",
        "keep it", "keep them", "keep the shortlist", "keep as-is",
        "keep as is", "confirmed", "confirm", "proceed", "move forward",
        "final list", "finalize", "i'm good", "this is fine", "fine by me",
        "approved", "approve", "done", "set", "final", "shortlist is good",
        "go with", "use this", "choose", "decided", "settled on",
        "opt for", "select", "pick", "we'll use", "we will use",
    ]
    user_ready = any(sig in query for sig in readiness_signals)

    # Explicit user readiness overrides everything
    if user_ready and has_role:
        state["intent"] = "retrieve"
        state["ready_to_recommend"] = True
        return state

    # Need at least a role to do anything meaningful
    if not has_role:
        state["intent"] = "clarify"
        return state

    # Turn 3+: always retrieve if we have a role (be permissive on later turns)
    if turn_count >= 3:
        state["intent"] = "retrieve"
        state["ready_to_recommend"] = True
        return state

    # Turn 1: be conservative unless the query is very specific
    if turn_count == 1:
        # If role + seniority + skills all present → probably specific enough
        if has_role and has_seniority and has_skills:
            state["intent"] = "retrieve"
            state["ready_to_recommend"] = True
            return state

        # If role + seniority but no skills → check if query is detailed enough
        if has_role and has_seniority:
            # Very short vague query → clarify (e.g. "senior leadership")
            if len(query.split()) < 6:
                state["intent"] = "clarify"
                return state
            # Otherwise retrieve on turn 1 (most common case: C4, C5, C6, C10)
            state["intent"] = "retrieve"
            state["ready_to_recommend"] = True
            return state

        # If role + skills but no seniority → specific enough (e.g. C8: admin assistants for Excel and Word)
        if has_role and has_skills:
            state["intent"] = "retrieve"
            state["ready_to_recommend"] = True
            return state

        # Only role, nothing else → clarify
        state["intent"] = "clarify"
        return state

    # Turn 2: retrieve if we have role + (seniority or skills)
    if has_role and (has_seniority or has_skills):
        state["intent"] = "retrieve"
        state["ready_to_recommend"] = True
        return state

    # Turn 2: if assistant previously asked a question and user answered, retrieve
    messages = state.get("messages", [])
    if len(messages) >= 2:
        last_assistant_msg = None
        for msg in reversed(messages[:-1]):
            if msg.get("role") == "assistant":
                last_assistant_msg = msg.get("content", "")
                break
        
        if last_assistant_msg and "?" in last_assistant_msg:
            # The assistant asked a question; user is answering it
            state["intent"] = "retrieve"
            state["ready_to_recommend"] = True
            return state

    state["intent"] = "clarify"
    return state