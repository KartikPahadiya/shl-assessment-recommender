import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

model = SentenceTransformer(
    MODEL_NAME,
    token=HF_TOKEN
)


def embed_text(texts, show_progress=False):
    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress
    )

    return embeddings