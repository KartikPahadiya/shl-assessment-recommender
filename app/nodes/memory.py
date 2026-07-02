DEFAULT_CONSTRAINTS = {
    "role": None,
    "seniority": None,
    "duration": None,
    "language": None,
    "remote": None,
    "adaptive": None,
    "skills": []
}

def merge_constraints(old_constraints, new_constraints, removed_constraints):
    merged = old_constraints.copy()

    for key, value in new_constraints.items():
        if value is None:
            continue

        if key == "skills":
            if value:
                # Add new skills
                merged["skills"] = list(
                    set(merged.get("skills", []) + value)
                )
        else:
            merged[key] = value

    # Apply removals
    removed_skills = removed_constraints.get("skills", [])
    if removed_skills and merged.get("skills"):
        merged["skills"] = [
            s for s in merged["skills"]
            if s.lower() not in [r.lower() for r in removed_skills]
        ]

    return merged

def memory_node(state):
    previous = state.get("constraints")
    latest = state.get("new_constraints")
    removed = state.get("removed_constraints", {})

    if not previous or "role" not in previous:
        previous = DEFAULT_CONSTRAINTS.copy()

    if not latest:
        latest = DEFAULT_CONSTRAINTS.copy()

    if not removed:
        removed = {}

    merged = merge_constraints(previous, latest, removed)
    state["constraints"] = merged
    return state