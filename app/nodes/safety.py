import re

# Keywords that trigger safety violations
PROMPT_INJECTION_KEYWORDS = [
    "ignore previous", "ignore all previous", "ignore your instructions",
    "reveal your prompt", "show your prompt", "what is your prompt",
    "jailbreak", "pretend you are", "act as", "act like", "forget",
    "system prompt", "override", "bypass", "disregard", "you are now",
    "new instructions", "new role", "ignore rules", "ignore constraints",
    "ignore previous instructions", "ignore previous rules",
]

OFF_TOPIC_KEYWORDS = [
    "salary", "how much should i pay", "offer letter", "compensation",
    "how to interview", "interview questions", "interview tips",
    "how to fire", "can i fire", "firing", "terminate employee",
    "legal advice", "legal question", "lawyer", "legal requirement",
    "hiring strategy", "recruitment strategy", "best hiring practice",
    "background check", "reference check", "drug test",
]

UNSAFE_KEYWORDS = [
    "hack", "steal", "fraud", "illegal", "exploit", "attack",
    "breach", "break in", "crack", "phishing", "scam", "cheat",
    "bypass security", "sql injection", "xss", "ddos", "malware",
    "ransomware", "trojan", "virus", "spyware", "keylogger",
]


def _contains_keyword(text, keywords):
    text_lower = text.lower()
    for kw in keywords:
        if kw in text_lower:
            return True
    return False


def safety_node(state):
    query = state["messages"][-1]["content"]
    query_lower = query.lower()

    # Check for prompt injection
    if _contains_keyword(query, PROMPT_INJECTION_KEYWORDS):
        state["safety_violation"] = True
        state["violation_type"] = "prompt_injection"
        return state

    # Check for off-topic queries
    if _contains_keyword(query, OFF_TOPIC_KEYWORDS):
        state["safety_violation"] = True
        state["violation_type"] = "off_topic"
        return state

    # Check for unsafe/harmful queries
    if _contains_keyword(query, UNSAFE_KEYWORDS):
        state["safety_violation"] = True
        state["violation_type"] = "unsafe"
        return state

    # Default: safe
    state["safety_violation"] = False
    state["violation_type"] = None
    return state
