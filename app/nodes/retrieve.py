from app.rag.retriever import retrieve


def _normalize_to_str(value):
    """Ensure a constraint value is a string; join lists with spaces."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v)
    if value is None:
        return ""
    return str(value)


def retrieve_node(state):
    constraints = state["constraints"]
    removed = state.get("removed_constraints", {})

    query_parts = []

    role = _normalize_to_str(constraints.get("role"))
    if role:
        query_parts.append(role)

    seniority = _normalize_to_str(constraints.get("seniority"))
    if seniority:
        query_parts.append(seniority)

    language = _normalize_to_str(constraints.get("language"))
    if language:
        query_parts.append(language)

    # ADD SKILLS TO QUERY (critical fix)
    skills = constraints.get("skills", [])
    if isinstance(skills, list):
        query_parts.extend(str(s) for s in skills if s)
    elif isinstance(skills, str):
        query_parts.append(skills)

    query = " ".join(query_parts)

    if not query.strip():
        state["reply"] = (
            "I need more information before recommending assessments. "
            "Please specify role and seniority."
        )
        state["recommendations"] = []
        state["retrieved_docs"] = []
        state["end_of_conversation"] = False
        return state

    docs = retrieve(query, constraints=constraints, k=10)

    # Filter out removed items/skills by name
    removed_skills = [r.lower() for r in removed.get("skills", [])]
    removed_items = [r.lower() for r in removed.get("items", [])]

    filtered_docs = []
    for doc in docs:
        doc_name_lower = doc.get("name", "").lower()
        # Skip if explicitly removed by name
        if any(ri in doc_name_lower for ri in removed_items):
            continue
        # Skip if removed skill appears in doc name
        if any(rs in doc_name_lower for rs in removed_skills):
            continue
        filtered_docs.append(doc)

    docs = filtered_docs

    if len(docs) == 0:
        state["reply"] = (
            "I couldn't find matching SHL assessments. "
            "Try relaxing constraints like duration or language."
        )
        state["recommendations"] = []
        state["retrieved_docs"] = []
        state["end_of_conversation"] = False
        return state

    docs = docs[:10]

    recommendations = []

    for doc in docs:
        recommendations.append({
            "name": doc["name"],
            "url": doc["link"],
            "duration": doc["duration"],
            "remote": doc["remote"],
            "adaptive": doc["adaptive"],
            "keys": doc.get("keys", [])
        })

    state["retrieved_docs"] = docs
    state["recommendations"] = recommendations
    state["end_of_conversation"] = False

    return state
