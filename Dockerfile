FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for FAISS and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (avoids 2GB+ CUDA packages)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's cached in the image
# This avoids downloading at runtime and speeds up cold start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy application code
COPY . .

# Expose HuggingFace Spaces default port
EXPOSE 7860

# Start the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
