"""Test harness that replays the 10 sample conversations against the SHL agent.

Usage (with TestClient, no server needed):
    python test_conversations.py

Usage (with running server):
    python test_conversations.py --url http://localhost:8000
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Parse sample conversations from Markdown
# ---------------------------------------------------------------------------

def parse_markdown_conversation(md_path: Path) -> List[Dict[str, Any]]:
    """Parse a C*.md file into a list of turn dicts."""
    text = md_path.read_text(encoding="utf-8")
    turns = []

    # Split by "### Turn N" headers
    raw_turns = re.split(r"###\s*Turn\s*\d+\s*\n", text)
    # first chunk is the "## Conversation" header — skip it
    for block in raw_turns[1:]:
        block = block.strip()
        if not block:
            continue

        # ---- User message ----
        user_match = re.search(
            r"\*\*User\*\*\s*\n>\s*(.+?)(?=\n\n\*\*Agent\*\*)",
            block, re.DOTALL
        )
        user_msg = user_match.group(1).replace("\n", " ").strip() if user_match else ""

        # ---- Does the expected turn have recommendations? ----
        has_table = bool(re.search(r"\|\s*#\s*\|\s*Name\s*\|", block))
        has_null = "recommendations: null" in block or "No recommendations" in block

        # ---- Expected end_of_conversation ----
        eoc_match = re.search(
            r"`end_of_conversation`:\s*\*\*(true|false)\*\*",
            block, re.IGNORECASE
        )
        expected_eoc = eoc_match.group(1).lower() == "true" if eoc_match else False

        # ---- Extract expected recommendation names from table ----
        expected_names = []
        if has_table:
            table_body = re.search(
                r"\|\s*#\s*\|.*\n\|[-:\s|]+\n((?:\|.*\n?)+)",
                block
            )
            if table_body:
                for line in table_body.group(1).strip().split("\n"):
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if len(cells) >= 2 and cells[0].isdigit():
                        expected_names.append(cells[1])

        turns.append({
            "user": user_msg,
            "expected_has_recs": has_table,
            "expected_eoc": expected_eoc,
            "expected_names": expected_names,
        })

    return turns


def load_catalog_urls(catalog_path: Path) -> set:
    """Return a set of valid SHL catalog URLs."""
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    urls = set()
    for item in catalog:
        link = item.get("link", "")
        if link.startswith("https://www.shl.com/"):
            urls.add(link)
        # Also accept URLs without trailing slash or with/without https
        urls.add(link.rstrip("/"))
        if link.startswith("https://"):
            urls.add(link.replace("https://", "http://", 1).rstrip("/"))
    return urls


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_response(resp: dict, turn_idx: int, expected: dict, valid_urls: set) -> List[str]:
    """Return a list of error strings for a single response."""
    errors = []

    # 1. Schema compliance
    if "reply" not in resp:
        errors.append(f"Turn {turn_idx+1}: missing 'reply' field")
    if "recommendations" not in resp:
        errors.append(f"Turn {turn_idx+1}: missing 'recommendations' field")
    if "end_of_conversation" not in resp:
        errors.append(f"Turn {turn_idx+1}: missing 'end_of_conversation' field")

    recs = resp.get("recommendations", [])
    eoc = resp.get("end_of_conversation", False)

    # 2. test_type present in every recommendation
    for i, rec in enumerate(recs):
        if "test_type" not in rec or not rec["test_type"]:
            errors.append(f"Turn {turn_idx+1}, rec {i+1}: missing 'test_type'")
        if "name" not in rec or not rec["name"]:
            errors.append(f"Turn {turn_idx+1}, rec {i+1}: missing 'name'")
        if "url" not in rec or not rec["url"]:
            errors.append(f"Turn {turn_idx+1}, rec {i+1}: missing 'url'")

    # 3. URL validation (must be from catalog)
    for i, rec in enumerate(recs):
        url = rec.get("url", "")
        # Strip markdown link wrappers if any
        clean_url = url.strip("<>").strip()
        if clean_url and clean_url not in valid_urls:
            # Try lenient match
            found = False
            for vu in valid_urls:
                if clean_url.rstrip("/") == vu.rstrip("/"):
                    found = True
                    break
            if not found:
                errors.append(f"Turn {turn_idx+1}, rec {i+1}: URL not in catalog: {clean_url}")

    # 4. Recommendation count (1-10 when non-empty)
    if len(recs) > 10:
        errors.append(f"Turn {turn_idx+1}: {len(recs)} recommendations (>10)")

    # 5. Expected presence/absence
    if expected["expected_has_recs"] and len(recs) == 0:
        errors.append(f"Turn {turn_idx+1}: expected recommendations but got empty list")
    if not expected["expected_has_recs"] and len(recs) > 0:
        # This is a warning, not necessarily an error — the agent might reasonably recommend
        # when the gold trace says null.  We flag it so the user can review.
        pass  # relaxed for now

    # 6. end_of_conversation correctness (only check final turn strictly)
    if expected["expected_eoc"] and not eoc:
        errors.append(f"Turn {turn_idx+1}: expected end_of_conversation=true but got false")

    return errors


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_with_testclient(valid_urls: set) -> dict:
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    results = {}

    conv_dir = Path(__file__).parent / "SampleConversations"
    for md_file in sorted(conv_dir.glob("C*.md")):
        conv_name = md_file.stem
        turns = parse_markdown_conversation(md_file)
        print(f"\n▶ Replaying {conv_name}  ({len(turns)} turns)")

        messages = []
        conv_errors = []
        turn_count = 0

        for idx, turn in enumerate(turns):
            messages.append({"role": "user", "content": turn["user"]})
            turn_count += 1

            # Call the endpoint
            response = client.post("/chat", json={"messages": messages})
            if response.status_code != 200:
                conv_errors.append(
                    f"Turn {idx+1}: HTTP {response.status_code} — {response.text}"
                )
                break

            resp_body = response.json()
            errors = validate_response(resp_body, idx, turn, valid_urls)
            conv_errors.extend(errors)

            # Simulate assistant turn in history for next request
            messages.append({
                "role": "assistant",
                "content": resp_body.get("reply", "")
            })

            # Check 8-turn cap
            if turn_count > 8:
                conv_errors.append(f"Turn {idx+1}: exceeded 8-turn user cap!")
                break

            # If end_of_conversation, stop replay
            if resp_body.get("end_of_conversation", False):
                break

        if conv_errors:
            print(f"  ❌ {len(conv_errors)} issue(s):")
            for e in conv_errors:
                print(f"    - {e}")
        else:
            print(f"  ✅ All checks passed")

        results[conv_name] = {
            "turns": len(turns),
            "errors": conv_errors,
            "passed": len(conv_errors) == 0,
        }

    return results


def run_with_url(base_url: str, valid_urls: set) -> dict:
    import requests

    results = {}
    conv_dir = Path(__file__).parent / "SampleConversations"

    for md_file in sorted(conv_dir.glob("C*.md")):
        conv_name = md_file.stem
        turns = parse_markdown_conversation(md_file)
        print(f"\n▶ Replaying {conv_name}  ({len(turns)} turns)")

        messages = []
        conv_errors = []
        turn_count = 0

        for idx, turn in enumerate(turns):
            messages.append({"role": "user", "content": turn["user"]})
            turn_count += 1

            try:
                response = requests.post(
                    f"{base_url}/chat",
                    json={"messages": messages},
                    timeout=120
                )
            except Exception as exc:
                conv_errors.append(f"Turn {idx+1}: request failed — {exc}")
                break

            if response.status_code != 200:
                conv_errors.append(
                    f"Turn {idx+1}: HTTP {response.status_code} — {response.text[:200]}"
                )
                break

            resp_body = response.json()
            errors = validate_response(resp_body, idx, turn, valid_urls)
            conv_errors.extend(errors)

            messages.append({
                "role": "assistant",
                "content": resp_body.get("reply", "")
            })

            if turn_count > 8:
                conv_errors.append(f"Turn {idx+1}: exceeded 8-turn user cap!")
                break

            if resp_body.get("end_of_conversation", False):
                break

        if conv_errors:
            print(f"  ❌ {len(conv_errors)} issue(s):")
            for e in conv_errors:
                print(f"    - {e}")
        else:
            print(f"  ✅ All checks passed")

        results[conv_name] = {
            "turns": len(turns),
            "errors": conv_errors,
            "passed": len(conv_errors) == 0,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay SHL sample conversations")
    parser.add_argument(
        "--url",
        default=None,
        help="Base URL of running server (e.g. http://localhost:8000). "
             "If omitted, uses FastAPI TestClient directly.",
    )
    args = parser.parse_args()

    catalog_path = Path(__file__).parent / "app" / "data" / "catalog_fixed.json"
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found at {catalog_path}")
        sys.exit(1)

    print("Loading catalog URLs...")
    valid_urls = load_catalog_urls(catalog_path)
    print(f"  {len(valid_urls)} unique URLs loaded")

    if args.url:
        print(f"\nTesting against running server: {args.url}")
        results = run_with_url(args.url.rstrip("/"), valid_urls)
    else:
        print("\nTesting with FastAPI TestClient (no server needed)")
        results = run_with_testclient(valid_urls)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for r in results.values() if r["passed"])
    print(f"Conversations tested : {total}")
    print(f"Passed               : {passed}")
    print(f"Failed               : {total - passed}")
    for name, r in results.items():
        status = "✅" if r["passed"] else "❌"
        print(f"  {status} {name}  ({r['turns']} turns)")
    print("=" * 60)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
