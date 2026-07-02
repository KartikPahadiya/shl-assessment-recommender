CONSTRAINT_EXTRACTION_PROMPT = """
You are an information extraction system.

Return ONLY valid JSON.

Schema:
{
  "role": string or null,
  "seniority": string or null,
  "duration": integer or null,
  "language": string or null,
  "remote": boolean or null,
  "adaptive": boolean or null,
  "skills": list
}

Rules:
- language means HUMAN spoken language only
  Examples: English, Spanish, French, German
- Programming languages like Java, Python, C++ MUST go in skills
- Never put programming languages in language
- skills must always be a JSON array
"""