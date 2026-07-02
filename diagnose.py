import json, requests

body = {
    "messages": [
        {"role": "user", "content": "I need an assessment for a senior Java developer"}
    ]
}

resp = requests.post("http://localhost:8000/chat", json=body, timeout=30)
print(f"Status: {resp.status_code}")
print(f"Response body:")
print(json.dumps(resp.json(), indent=2))
