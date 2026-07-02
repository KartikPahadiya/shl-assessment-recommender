"""Mapping utilities for SHL catalog data to API schema."""

# Mapping from catalog 'keys' (categories) to test_type letter codes
KEY_TO_TEST_TYPE = {
    "Personality & Behavior": "P",
    "Knowledge & Skills": "K",
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Simulations": "S",
    "Assessment Exercises": "E",
}


def map_keys_to_test_type(keys):
    """Map a list of catalog keys to a comma-separated test_type string.

    Returns joined codes like "K,S" or "P". Unknown keys are skipped.
    """
    if not keys:
        return ""
    codes = []
    for key in keys:
        key_stripped = key.strip()
        code = KEY_TO_TEST_TYPE.get(key_stripped)
        if code:
            codes.append(code)
    return ",".join(codes) if codes else ""


def build_recommendation(item):
    """Build a schema-compliant recommendation dict from a catalog item."""
    # retrieve_node stores URL as 'url', but raw catalog items have 'link'
    url = item.get("url") or item.get("link", "")
    return {
        "name": item.get("name", ""),
        "url": url,
        "test_type": map_keys_to_test_type(item.get("keys", [])),
        "duration": item.get("duration") or None,
        "remote": item.get("remote") or None,
        "adaptive": item.get("adaptive") or None,
    }
