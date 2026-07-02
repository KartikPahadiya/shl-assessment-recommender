---
title: SHL Assessment Recommender API
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# SHL Assessment Recommender API

A conversational AI agent that recommends SHL assessments based on job roles, required skills, seniority level, and other constraints. Built for the SHL GenAI Intern Hiring Assessment.

## API Endpoint

### `POST /chat`

Accepts a conversation history and returns a reply with assessment recommendations.

**Request body:**
```json
{
  "messages": [
    {"role": "user", "content": "I need an assessment for a senior Python developer"},
    {"role": "assistant", "content": "What specific skills should the assessment cover?"},
    {"role": "user", "content": "Python, Django, and system design skills."}
  ]
}
```

**Response:**
```json
{
  "reply": "Here are 3 recommended assessments for a senior Python developer with Django and system design focus...",
  "recommendations": [
    {
      "name": "Python (Coding): Entry-level Algorithms",
      "url": "https://www.shl.com/solutions/products/product-catalog/python-coding-entry-level-algorithms/",
      "test_type": "K",
      "duration": "35 min",
      "remote": true,
      "adaptive": false
    }
  ],
  "end_of_conversation": false
}
```

## Features

- **Conversational Interface**: Multi-turn dialogue that refines recommendations through natural conversation
- **Hybrid Retrieval**: Combines FAISS vector similarity + BM25 keyword matching for accurate assessment matching
- **Constraint Refinement**: Users can add, remove, or compare constraints dynamically
- **Schema-Compliant Output**: Returns structured recommendations with `name`, `url`, `test_type`, `duration`, `remote`, and `adaptive` fields
- **Safety Checks**: Built-in prompt injection and off-topic detection

## Architecture

- **FastAPI** backend with Pydantic schemas
- **LangGraph** state machine for conversation flow (parser → router → retrieve → formatter → completion)
- **Groq LLM** (Llama 3.1 8B Instant) for fast, free inference
- **Local Embeddings**: BAAI/bge-small-en-v1.5 running on CPU
- **FAISS** vector index + **BM25** sparse retrieval for hybrid search

## Setup

### Secrets (Required)

Set the following secret in your HuggingFace Space settings:

- `GROQ_API_KEY` — Get a free API key at [https://console.groq.com/keys](https://console.groq.com/keys)

No HuggingFace token is needed for the embedding model (public).

## Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Test

```bash
python test_conversations.py
```

This replays 10 sample conversations and validates schema compliance.
