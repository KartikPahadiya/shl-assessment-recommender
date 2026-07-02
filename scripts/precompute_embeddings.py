import json
import pickle
import faiss
import numpy as np

from app.rag.embeddings import embed_text

with open("app/data/catalog_fixed.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

documents = []

for item in catalog:
    text = f"""
Assessment Name: {item['name']}

Description: {item['description']}

Job Levels: {', '.join(item['job_levels'])}

Duration: {item['duration']}

Remote: {item['remote']}

Adaptive: {item['adaptive']}

Categories: {', '.join(item['keys'])}
"""
    documents.append(text)

embeddings = embed_text(documents)
embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "app/data/faiss.index")

with open("app/data/metadata.pkl", "wb") as f:
    pickle.dump(catalog, f)

print(f"Indexed {len(catalog)} assessments")