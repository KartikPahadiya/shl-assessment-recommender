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
You are an SHL assessment recommendation assistant.

User request:
{query}

Retrieved candidate assessments from SHL catalog:
{rec_text}

Your tasks:
1. Rerank candidates by relevance to the user query
2. Remove irrelevant/noisy assessments
3. Select best 1 to 10 assessments only
4. Recommend only from provided candidates

Important ranking criteria:
- Role match
- Seniority match
- Skill match
- Language constraints
- Duration constraints
- Remote/adaptive requirements

Rules:
- Recommend ONLY SHL assessments from candidate list
- NEVER invent assessments
- NEVER invent URLs
- Every URL must be copied exactly from candidate list
- Ignore irrelevant assessments even if retrieved

Response format:
1. Brief summary of user requirement
2. Recommended assessments (best first)
3. Why each fits
4. End with a follow-up question
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