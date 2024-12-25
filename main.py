import pandas as pd
from modules.EDA.dataset_visual import (
    regression_visualization,
    classification_visualization,
)
from modules.NLP.fetch_news import fetch_sustainability_news
from modules.NLP.llama_model import ai_generate_problem_statement
from modules.NLP.nlg_problem_statement import generate_problem_statement
from modules.NLP.text_processing import clean, extract_pos
from modules.NLP.word_cloud import bag_of_words_menu
from modules.NLP.nltk_resources import download_nltk_resources
from modules.EDA.models import train_regression_model, train_classification_model
from modules.RAG.pdf_process import extract_text_from_pdf
from modules.RAG.embeddings import generate_embeddings
from modules.RAG.doc_retrieval import retrieve_top_k_documents
from modules.RAG.rag_model import generate_answer
from modules.MTR.pycaret_visual import classification, Regression

def get_choice():
    print("\n=== Welcome to AI-Hobbyist ===\n")
    print("For your idea generation, do you:")
    print("   a. Have a story,")
    print("   b. Want to check the news,")
    print("   c. Want to upload a document?")
    return input("Enter your choice (a/b/c): ").strip().lower()

def handle_story():
    story = input("Unveil your story to us: ").strip()
    cleaned_story = clean(story)
    print("\nCrafting the Bag of Words from your story...")
    words = extract_pos(cleaned_story, pos_types=("NN", "JJ"))
    word_count = {word: cleaned_story.split().count(word) for word in words}
    bag_of_words_menu(word_count)
    return word_count

def handle_news():
    print("\nGathering the latest sustainability news for you...\n")
    news_list = fetch_sustainability_news()
    if not news_list:
        print("No news available at the moment. Please try again later.")
        return
    print("\nSelect a news article:")
    for idx, (title, description) in enumerate(news_list, 1):
        print(f"{idx}. {title}")
    try:
        news_choice = int(input("Enter the news number: ").strip())
        if news_choice < 1 or news_choice > len(news_list):
            print("Invalid selection. Exiting program.")
            return
    except ValueError:
        print("Invalid input. Exiting program.")
        return
    selected_news = news_list[news_choice - 1][1]
    cleaned_news = clean(selected_news)
    print("\nTransforming the selected news into a rich Bag of Words...")
    words = extract_pos(cleaned_news, pos_types=("NN", "JJ"))
    word_count = {word: cleaned_news.split().count(word) for word in words}
    bag_of_words_menu(word_count)
    return word_count

def handle_document():
    print("\nPlease upload your document (PDF file):")
    pdf_file = input("Enter the file path: ").strip()
    try:
        document_text = extract_text_from_pdf(pdf_file)
        print("\nDocument extracted successfully. Summarizing...\n")
        documents = [document_text]
        document_embeddings = generate_embeddings(documents)
        retrieved_document = retrieve_top_k_documents(document_embeddings, documents)[0]
        summary = generate_answer(retrieved_document)
        print("\nDocument Summary:")
        print(summary)
        cleaned_document = clean(summary)
        print("\nGenerating Bag of Words from the document summary...")
        words = extract_pos(cleaned_document, pos_types=("NN", "JJ"))
        word_count = {word: cleaned_document.split().count(word) for word in words}
        bag_of_words_menu(word_count)
    except Exception as e:
        print(f"Error in document processing: {e}")
    return word_count

def select_task():
    while True:
        print("\nChoose how would you like your problem statement to begin with?")
        print("1. To Classify...")
        print("2. To Regress...")
        task_choice = input("Enter your choice (1/2): ").strip()
        task_mapping = {"1": "classify", "2": "regress"}
        task = task_mapping.get(task_choice, None)
        if task:
            return task
        else:
            print("Invalid choice. Please select 1 or 2.")

def select_keywords(word_count):
    try:
        selected_words = input("Enter the words (numbers) you want to include in your problem statement: ").strip()
        selected_indices = list(map(int, selected_words.split(",")))
        selected = [word for idx, word in enumerate(word_count.keys(), start=1) if idx in selected_indices]
        return selected
    except ValueError:
        print("Invalid input for keywords. Please use comma-separated numbers corresponding to the indices.")
        return []

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
        print("\nGenerated AI Problem Statement:")
        print(problem_statement)
    else:
        print("Invalid choice. Exiting program.")


def load_dataset(task):
    dataset_files = {
        "classify": "datasets/classification.csv",
        "regress": "datasets/regression.csv",
    }
    dataset_file = dataset_files[task]
    try:
        df = pd.read_csv(dataset_file)
        print("\nHere is the dataset based on the problem statement for analysis:")
        print("\n*Please observe it carefully*\n")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Dataset file for {task} not found. Exiting program.")
        return None

def perform_eda_or_train_model(task):
    task_name = "classification" if task == "classify" else "regression"  # Determine the task name
    
    choice = input(f"\nWould you like to perform EDA with the pre-made dataset or train a model with your own dataset based on *{task_name}*?\nEnter 'EDA' for EDA or 'train' to train a model: ").strip().lower()

    if choice == 'eda':
        df = load_dataset(task)
        if df is None:
            return
        perform_eda(df, task)
    elif choice == 'train':
        dataset_path = input("Enter the path to your dataset: ").strip()
        target = input("Enter the target variable: ").strip()

        if task == "classify":
            classification(dataset_path, target)
        elif task == "regress":
            Regression(dataset_path, target)
    else:
        print("Invalid choice. Exiting program.")

def perform_eda(df, task):
    while True:
        print("\nBased on your observation, what analysis would you like to perform?")
        print("1. Classification Analysis")
        print("2. Regression Analysis")
        analysis_choice = input("Enter your choice (1/2): ").strip()

        if analysis_choice == "1" and task == "classify":
            model, y_test, y_pred, accuracy, precision, recall, f1 = train_classification_model(df)
            print(f"Classification Model Accuracy: {accuracy * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")
            classification_visualization(df, y_test, y_pred)
            break
        elif analysis_choice == "2" and task == "regress":
            model, y_test, y_pred, mse, r2 = train_regression_model(df)
            print(f"Regression Model Mean Squared Error: {mse:.2f}")
            print(f"R-squared: {r2:.2f}")
            regression_visualization(df, y_test, y_pred)
            break
        else:
            print("Incorrect analysis choice for the task. Please try again.")

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

    perform_eda_or_train_model(task)

if __name__ == "__main__":
    main()