import torch
from sentence_transformers import SentenceTransformer
import os
import fitz 
from huggingface_hub import InferenceClient
from package.idea.config import HUGGINGFACE_API_KEY

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

def retrieve_top_k_documents(document_embeddings, documents, top_k=1):
    
    cosine_similarities = torch.nn.functional.cosine_similarity(
        document_embeddings[0].unsqueeze(0), document_embeddings
    )
    
    top_indices = torch.argsort(cosine_similarities, descending=True)[:top_k]
    
    retrieved_documents = [documents[i] for i in top_indices]
    return retrieved_documents

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

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)  
    text = ""
    for page in doc:
        text += page.get_text()  
    return text

def generate_answer(retrieved_text, model_name="meta-llama/Llama-3.2-3B-Instruct", max_length=150):

    prompt = (
        f"Using the following document, summarize it in 50 words:\n\n"
        f"Document: {retrieved_text}\n\n"
        f"Answer:"
    )
    
    completion = client.text_generation(
        model=model_name,
        prompt=prompt,
        temperature=0.7,  
        max_new_tokens=max_length, 
        repetition_penalty=1.1  
    )
    
    return completion[0] if isinstance(completion, list) else completion