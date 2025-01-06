from modules.MAIN.get_choice import get_choice
from modules.MAIN.handle_choice import (
    handle_story,
    handle_news,
    handle_document
)
from modules.MAIN.select_task import select_task
from modules.MAIN.select_keywords import select_keywords
from modules.MAIN.generate_problem_statement import generate_problem_statement_choice
from modules.NLP.nltk_resources import download_nltk_resources
from modules.PRE.dataset_preprocessing import dataset_preprocessing
from modules.EDA.load_dataset import load_dataset
from modules.EDA.perform_eda import perform_eda
from modules.TRN.classification import classification_task
from modules.TRN.regression import regression_task

def main():
    download_nltk_resources()

    choice = get_choice()
    word_count = {}

    if choice == "a":
        word_count = handle_story()
    elif choice == "b":
        word_count = handle_news()
    elif choice == "c":
        word_count = handle_document()
    else:
        print("Invalid choice. Exiting program.")
        return

    if not word_count:
        print("No keywords generated. Exiting program.")
        return

    task = select_task()
    
    selected_keywords = select_keywords(word_count)

    if not selected_keywords:
        print("No keywords selected. Exiting program.")
        return

    generate_problem_statement_choice(selected_keywords, task)

    print("\nProblem statement generated successfully!")
    print("\nNext, we will proceed with dataset preprocessing.")
    
    dataset_preprocessing()

    print("\nNext, we will proceed with Exploratory Data Analysis.")

    dataset_1 = load_dataset()

    perform_eda(dataset_1)

    dataset_2 = load_dataset()

    if dataset_2 is not None:
        print("\nChoose a task:")
        print("1. Classification")
        print("2. Regression")
        task_choice = input("Enter the number corresponding to your choice: ")

        if task_choice == "1":
            classification_task(dataset_2)
        elif task_choice == "2":
            regression_task(dataset_2)
        else:
            print("Invalid choice. Exiting program.")
    
if __name__ == "__main__":
    main()