from sentence_transformers import SentenceTransformer
import os

def generate_embeddings(documents, model_name="all-MiniLM-L6-v2", model_dir="models"):

    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_name}...")
        embedder = SentenceTransformer(model_name)
        embedder.save(model_path)
        print(f"Model downloaded and saved to {model_path}")
    else:
        print(f"Using locally saved model from {model_path}...")
    
    embedder = SentenceTransformer(model_path)
    
    document_embeddings = embedder.encode(documents, convert_to_tensor=True)
    
    return document_embeddings