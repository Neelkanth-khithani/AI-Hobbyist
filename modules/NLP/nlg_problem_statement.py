def generate_problem_statement(chosen_words, task):
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