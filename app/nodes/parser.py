import json
import re
from app.llm import llm
from app.prompts import CONSTRAINT_EXTRACTION_PROMPT

DEFAULT_CONSTRAINTS = {
    "role": None,
    "seniority": None,
    "duration": None,
    "language": None,
    "remote": None,
    "adaptive": None,
    "skills": []
}

PROGRAMMING_LANGUAGES = {
    "java", "python", "c++", "javascript",
    "typescript", "go", "rust", "scala"
}


# Keywords that signal removal of skills or items.
# Capture until sentence terminator; "and" inside is handled by post-processing split.
# NOTE: the regex uses a negative lookahead to avoid crossing action-verb boundaries
# so "Drop REST and add AWS" captures only "REST".
REMOVAL_KEYWORDS = [
    r"drop\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"remove\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"exclude\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"without\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"no\s+(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"take out\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"get rid of\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"skip\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"omit\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"leave out\s+(?:the\s+)?(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"replace\s+(?:the\s+)?(.+?)\s+with\s+(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
    r"instead of\s+(?:the\s+)?(.+?),?\s*(?:use|add|try)\s+(.+?)(?:\s+and\s+(?:add|use|try|include|go|replace|with|put|switch|also|then|plus|as|using|consider|maybe|perhaps|can|could|would|should|will|want)\s+|,|;|\.|$)",
]

# Action verbs that indicate a new clause, not a removal item
ACTION_VERBS = {
    "add", "use", "try", "include", "go", "replace", "with", "put", "switch",
    "also", "then", "plus", "as", "using", "consider", "maybe", "perhaps",
    "can", "could", "would", "should", "will", "want",
}

def _is_valid_removal_item(text):
    """Check if a split text part is a valid removal item (not an action verb)."""
    text = text.strip().strip(".,;")
    if not text:
        return False
    first_word = text.split()[0].lower()
    if first_word in ACTION_VERBS:
        return False
    return True


def extract_removals(query):
    """Extract removed skills/items from user query via regex."""
    removed_skills = []
    removed_items = []
    replaced = {}  # old -> new (for future use)

    q_lower = query.lower()

    for pattern in REMOVAL_KEYWORDS:
        matches = re.finditer(pattern, q_lower, re.IGNORECASE)
        for match in matches:
            groups = match.groups()
            if "replace" in match.group(0) or "instead of" in match.group(0):
                if len(groups) >= 2:
                    old_thing = groups[0].strip()
                    new_thing = groups[1].strip()
                    removed_skills.append(old_thing)
                    replaced[old_thing] = new_thing
            else:
                thing = groups[0].strip()
                # Heuristic: if it's a single word or short phrase, it's likely a skill
                if len(thing.split()) <= 3:
                    removed_skills.append(thing)
                else:
                    removed_items.append(thing)

    # Split multi-item removals on commas and 'and', filtering out action-verb clauses
    expanded_skills = []
    for raw in removed_skills:
        parts = re.split(r",\s*|\s+and\s+|\s+&\s+", raw)
        for p in parts:
            p = p.strip().strip(".,;")
            if p and _is_valid_removal_item(p) and p not in expanded_skills:
                expanded_skills.append(p)

    expanded_items = []
    for raw in removed_items:
        parts = re.split(r",\s*|\s+and\s+|\s+&\s+", raw)
        for p in parts:
            p = p.strip().strip(".,;")
            if p and _is_valid_removal_item(p) and p not in expanded_items:
                expanded_items.append(p)

    return {
        "skills": list(set(expanded_skills)),
        "items": list(set(expanded_items)),
        "replaced": replaced,
    }


def parser_node(state):
    query = state["messages"][-1]["content"]

    prompt = f"""
{CONSTRAINT_EXTRACTION_PROMPT}

User Query:
{query}
"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
    except Exception:
        # Graceful fallback: if LLM is unavailable, use empty constraints
        raw = ""

    # Remove markdown code fences if LLM returns ```json ... ```
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        constraints = json.loads(raw)
    except json.JSONDecodeError:
        constraints = DEFAULT_CONSTRAINTS.copy()

    # Ensure dict
    if not isinstance(constraints, dict):
        constraints = DEFAULT_CONSTRAINTS.copy()

    # Fill missing keys
    for key, default_value in DEFAULT_CONSTRAINTS.items():
        if key not in constraints:
            constraints[key] = default_value

    # Force skills to always be list
    if constraints["skills"] is None:
        constraints["skills"] = []

    if not isinstance(constraints["skills"], list):
        constraints["skills"] = []

    # Regex override for duration
    duration_match = re.search(
        r"(under|below|less than)\s+(\d+)",
        query.lower()
    )

    if duration_match:
        constraints["duration"] = int(duration_match.group(2))

    # Fix Java/Python mistakenly classified as human language
    lang = constraints.get("language")
    if isinstance(lang, str) and lang.lower() in PROGRAMMING_LANGUAGES:
        constraints["language"] = None

        existing_skills = [s.lower() for s in constraints["skills"]]
        if lang.lower() not in existing_skills:
            constraints["skills"].append(lang)

    # Normalize seniority
    if isinstance(constraints["seniority"], str):
        constraints["seniority"] = constraints["seniority"].lower()

    # Remove seniority words from role if duplicated
    if isinstance(constraints["role"], str):
        role = constraints["role"]
        role = re.sub(
            r"\b(senior|junior|lead|principal|mid)\b",
            "",
            role,
            flags=re.IGNORECASE
        )
        constraints["role"] = " ".join(role.split())

    # Extract negative constraints (removals)
    removals = extract_removals(query)
    state["removed_constraints"] = removals

    state["user_query"] = query
    state["new_constraints"] = constraints

    return state