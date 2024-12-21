from huggingface_hub import InferenceClient
from config import HUGGINGFACE_API_KEY

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

def generate_answer(retrieved_text, model_name="meta-llama/Llama-3.2-3B-Instruct", max_length=150):

    prompt = (
        f"Using the following document, summarize it in 100 words:\n\n"
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