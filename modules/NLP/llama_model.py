from huggingface_hub import InferenceClient
from config import HUGGINGFACE_API_KEY

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

def ai_generate_problem_statement(chosen_keywords, task):
    prompt = (
        f"Generate a single 100-word problem statement based on the following keywords: {', '.join(chosen_keywords)}. "
        f"It should begin with one of the following terms: 'To classify', 'To regress' based on the task {task}. Be very sure to begin with this task only! "
        f"Additionally, clearly define the dependent variable (target column) and independent variables (dataset columns) for the task. "
        f"The dataset column names must align with the keywords provided and give users a clear idea for creating the dataset. "
        f"The statement must be creative, informative, and relevant for new users starting with machine learning. JUST GENERATE THE OUTPUT!"
    )
    completion = client.text_generation(
        model="meta-llama/Llama-3.2-3B-Instruct",
        prompt=prompt,
        temperature=0.5,
        repetition_penalty=1.1,
        max_new_tokens=200 
    )
    return completion[0] if isinstance(completion, list) else completion