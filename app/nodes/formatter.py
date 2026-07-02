from app.llm import llm


def build_prompt(query, docs):
    # Use only top 5 docs to keep the prompt short and fast
    docs = docs[:5]
    rec_text = ""

    for i, item in enumerate(docs, 1):
        rec_text += f"""
Candidate {i}:
Name: {item['name']}
Description: {item['description'][:200]}...
Duration: {item['duration']}
Remote: {item['remote']}
Adaptive: {item['adaptive']}
Languages: {', '.join(item['languages'][:5])}
URL: {item['link']}
"""

    prompt = f"""
You are a friendly, concise SHL assessment recommendation assistant chatting with a hiring manager.

User request:
{query}

Retrieved candidate assessments from the SHL catalog:
{rec_text}

Your job:
- Pick the 1–10 best-matching assessments from the list above.
- Write a short, natural, conversational reply (2–4 sentences max).
- Briefly mention the top 1–3 picks by name and why they fit.
- Do NOT write structured sections like "User Requirement Summary" or "Recommended Assessments".
- Do NOT write bullet points or numbered lists.
- Do NOT dump all candidate details.
- If helpful, end with one short follow-up question (e.g., "Would you like shorter options?" or "Should I filter for remote-only tests?").

Rules:
- Recommend ONLY from the provided candidate list.
- NEVER invent assessments or URLs.
- Keep it chatty and human.
"""
    return prompt

def formatter_node(state):
    query = state["user_query"]
    docs = state["retrieved_docs"]

    prompt = build_prompt(query, docs)
    try:
        response = llm.invoke(prompt)
        state["reply"] = response.content
    except Exception:
        state["reply"] = (
            "Here are the relevant assessments from the SHL catalog. "
            "Let me know if you need further refinement."
        )
    return state