import faiss
import pickle

index = faiss.read_index("app/data/faiss.index")

with open("app/data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Add index position to each doc for FAISS score lookup
for i, item in enumerate(metadata):
    item["_idx"] = i

def get_doc(idx):
    return metadata[idx]