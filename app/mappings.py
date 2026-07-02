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

    def _bool(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.lower().strip()
            if lower in ("yes", "true", "1", "y"):
                return True
            if lower in ("no", "false", "0", "n"):
                return False
        return None

    return {
        "name": item.get("name", ""),
        "url": url,
        "test_type": map_keys_to_test_type(item.get("keys", [])),
        "duration": item.get("duration") or None,
        "remote": _bool(item.get("remote")),
        "adaptive": _bool(item.get("adaptive")),
    }
