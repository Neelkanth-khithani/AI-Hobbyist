from huggingface_hub import InferenceClient
from package.idea.config import HUGGINGFACE_API_KEY

client = InferenceClient(api_key=HUGGINGFACE_API_KEY)

def nlg_generate_problem_statement(chosen_words, task):
    dependent_variable = chosen_words[0] if chosen_words else "Target"
    independent_variables = [f"- {word}" for word in chosen_words[1:]]
    
    problem_statement = (
        f"To {task}, we aim to analyze the given dataset to solve challenges such as "
        f"{', '.join(chosen_words)}. The goal is to understand the relationship between "
        f"the dependent variable '{dependent_variable}' and the independent variables.\n\n"
        
        f"Dependent Variable: {dependent_variable}\n"
        f"Independent Variables:\n"
        f"{chr(10).join(independent_variables)}\n\n"
        
        f"Dataset Columns:\n"
    )
    
    for word in chosen_words:
        problem_statement += (
            f"- {word}\n"
        )
    
    return problem_statement

def ai_generate_problem_statement(chosen_keywords, task):
    prompt = (
        f"Generate a single 100-word problem statement based on the following keywords: {', '.join(chosen_keywords)}. "
        f"It should begin with one of the following terms: 'To classify', 'To regress' based on the task {task}. Be very sure to begin with this task only! "
        f"Additionally, clearly define the dependent variable (target column) and independent variables (dataset columns) for the task. "
        f"The dataset column names must align with the keywords provided and give users a clear idea for creating the dataset. "
        f"The statement must be creative, informative, and relevant for new users starting with machine learning. JUST GENERATE THE OUTPUT!"
    )
    try:
        completion = client.text_generation(
            model="meta-llama/Llama-3.2-3B-Instruct",
            prompt=prompt,
            temperature=0.7,
            repetition_penalty=1.1,
            max_new_tokens=200
        )
        if isinstance(completion, list) and completion:
            return completion[0]
        elif isinstance(completion, str):
            return completion
        else:
            print("Unexpected response format:", completion)
            return "Problem statement generation failed. Please check the input."
    except Exception as e:
        print("Error during problem statement generation:", e)
        return "Problem statement generation failed due to an error."