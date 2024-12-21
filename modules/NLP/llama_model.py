from huggingface_hub import InferenceClient
from config import HUGGINGFACE_API_KEY

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

def ai_generate_problem_statement(chosen_keywords):
    prompt = (
        f"Generate a single 100-word problem statement based on the following keywords: {', '.join(chosen_keywords)}. "
        f"It should begin with one of the following terms: 'To classify', 'To regress'"
        f"Ensure the statement is clear, concise, and relevant to current industry or societal needs. "
        f"Be sure to use formal, professional language in your response. JUST GENERATE THE PROBLEM STATEMENT!"
    )
    completion = client.text_generation(
        model="meta-llama/Llama-3.2-3B-Instruct",
        prompt=prompt,
        temperature=0.7,
        max_new_tokens=150,
        repetition_penalty=1.1
    )
    return completion[0] if isinstance(completion, list) else completion