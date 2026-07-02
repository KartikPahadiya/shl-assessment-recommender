from app.rag.retriever import retrieve
from app.llm import llm
import re


def find_assessment(name):
    results = retrieve(name, k=1)

    if len(results) == 0:
        return None

    return results[0]


def build_compare_prompt(doc1, doc2):
    return f"""
You are an SHL assessment comparison assistant.

Compare these two SHL assessments.

Assessment 1:
Name: {doc1['name']}
Description: {doc1['description']}
Duration: {doc1['duration']}
Remote: {doc1['remote']}
Languages: {', '.join(doc1['languages'])}

Assessment 2:
Name: {doc2['name']}
Description: {doc2['description']}
Duration: {doc2['duration']}
Remote: {doc2['remote']}
Languages: {', '.join(doc2['languages'])}

Explain clearly:
1. Main differences
2. Which role each suits
3. Which one to choose in what scenario
4. Final recommendation
"""


def compare_node(state):
    query = state["user_query"].lower()

    separators = [" vs ", " versus ", " vs. "]
    split_result = None

    for sep in separators:
        if sep in query:
            split_result = query.split(sep, 1)
            break

    if split_result is None:
        state["reply"] = "Please specify two assessments using 'vs' or 'versus'."
        state["end_of_conversation"] = False
        return state

    left = split_result[0].strip()
    right = split_result[1].strip()

    left = re.sub(r"compare\s+", "", left).strip()

    doc1 = find_assessment(left)
    doc2 = find_assessment(right)

    if not doc1 or not doc2:
        state["reply"] = "Could not identify one or both assessments."
        state["end_of_conversation"] = False
        return state

    prompt = build_compare_prompt(doc1, doc2)
    try:
        response = llm.invoke(prompt)
        state["reply"] = response.content
    except Exception:
        state["reply"] = (
            f"Both assessments are valid SHL solutions from the catalog. "
            f"{doc1['name']} and {doc2['name']} differ in focus and duration. "
            f"Please review the catalog details for the best fit."
        )

    # Keep existing recommendations rather than replacing with just the two compared items
    # This matches expected behavior where compare turn still shows the full shortlist
    state["end_of_conversation"] = False

    return state