from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Groq offers a generous free tier with fast inference.
# Get your free API key at: https://console.groq.com/keys
# Set GROQ_API_KEY in your .env file or environment.

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set.\n"
        "1. Get a free key at https://console.groq.com/keys\n"
        "2. Add it to your .env file: GROQ_API_KEY=gsk_...\n"
        "3. Restart the server."
    )

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=512,
    api_key=GROQ_API_KEY,
)