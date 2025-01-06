from modules.NLP.llama_model import ai_generate_problem_statement
from modules.NLP.nlg_problem_statement import generate_problem_statement

def generate_problem_statement_choice(selected_keywords, task):
    print("\nHow would you like to generate the problem statement?")
    print("a. Normal NLG")
    print("b. AI-Generated")
    ps_choice = input("Enter your choice (a/b): ").strip().lower()

    if ps_choice == "a":
        problem_statement = generate_problem_statement(selected_keywords, task)
        print("\nGenerated Problem Statement:")
        print(problem_statement)
    elif ps_choice == "b":
        problem_statement = ai_generate_problem_statement(selected_keywords, task)
        print("\nGenerated AI Problem Statement by Llama:")
        print(problem_statement)
        print("--------------------------------------------")
    else:
        print("Invalid choice. Exiting program.")