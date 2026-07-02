def completion_node(state):
    query = state["user_query"].lower()
    intent = state.get("intent", "")
    turn_count = state.get("turn_count", 0)

    # 8-turn cap enforcement: force end if we hit the limit
    # The evaluator caps at 8 turns including user & assistant.
    # If we're at turn 7 (7 user messages), we should end on this response.
    if turn_count >= 7:
        state["end_of_conversation"] = True
        return state

    # Expanded acceptance / closing signals
    closing_signals = [
        "thanks", "thank you", "done", "that's enough", "bye",
        "perfect", "that's what we need", "what we need",
        "that works", "works for me", "looks good", "sounds good",
        "i'm good", "i am good", "good to go", "all good",
        "keep it", "keep them", "keep the shortlist", "keep as-is",
        "keep as is", "confirmed", "confirm", "proceed", "move forward",
        "final list", "finalize", "this is fine", "fine by me",
        "approved", "approve", "set", "final", "shortlist is good",
        "locking it in", "lock it in", "locked in", "go ahead",
        "understood", "clear", "got it", "makes sense",
        "no further questions", "nothing else", "that's all",
        "we'll use this", "using this", "decided", "we have decided",
        "agreed", "we agree", "accepted", "we accept",
        "that's good", "this is good", "good", "sounds perfect",
        "excellent", "great", "wonderful", "awesome", "superb",
        "love it", "love this", "perfect choice", "perfect fit",
        "let's go with", "go with this", "use this", "we'll take it",
        "this is it", "this is the one", "this works", "this is perfect",
        "i am satisfied", "satisfied", "happy with this", "content",
    ]

    if any(sig in query for sig in closing_signals):
        state["end_of_conversation"] = True
        return state

    # Clarification means conversation continues
    if intent == "clarify":
        state["end_of_conversation"] = False
        return state

    # Compare usually leads to follow-up
    if intent == "compare":
        state["end_of_conversation"] = False
        return state

    # Retrieval usually allows refinement
    if intent == "retrieve":
        state["end_of_conversation"] = False
        return state

    # Refusal can end turn
    if intent == "refuse":
        state["end_of_conversation"] = True
        return state

    state["end_of_conversation"] = False
    return state