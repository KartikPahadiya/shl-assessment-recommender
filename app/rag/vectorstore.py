import faiss
import pickle

index = faiss.read_index("app/data/faiss.index")

with open("app/data/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def get_doc(idx):
    return metadata[idx]