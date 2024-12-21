import torch

def retrieve_top_k_documents(document_embeddings, documents, top_k=1):
    
    cosine_similarities = torch.nn.functional.cosine_similarity(
        document_embeddings[0].unsqueeze(0), document_embeddings
    )
    
    top_indices = torch.argsort(cosine_similarities, descending=True)[:top_k]
    
    retrieved_documents = [documents[i] for i in top_indices]
    return retrieved_documents