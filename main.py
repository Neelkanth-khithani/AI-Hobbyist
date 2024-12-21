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

def get_choice():
    print("\n=== Welcome to AI-Hobbyist ===\n")
    print("1. Do you have a story, or would you like to explore news?")
    print("   a. I have a story.")
    print("   b. I want to check the news.")
    print("   c. I want to upload a document.")
    return input("Enter your choice (a/b/c): ").strip().lower()


def handle_story():
    story = input("Enter your story: ").strip()
    cleaned_story = clean(story)
    print("\nGenerating Bag of Words from your story...")
    words = extract_pos(cleaned_story, pos_types=("NN", "JJ"))
    word_count = {word: cleaned_story.split().count(word) for word in words}
    bag_of_words_menu(word_count)


def handle_news():
    print("Fetching sustainability-related news...\n")
    news_list = fetch_sustainability_news()
    if not news_list:
        print("No news available at the moment. Please try again later.")
        return
    print("Select a news article to analyze:")
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
    print("\nGenerating Bag of Words from the selected news...")
    words = extract_pos(cleaned_news, pos_types=("NN", "JJ"))
    word_count = {word: cleaned_news.split().count(word) for word in words}
    bag_of_words_menu(word_count)


def handle_document():
    print("Please upload your document (PDF file):")
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
    selected_words = input(
        "Enter the numbers corresponding to the keywords you want to use (comma-separated): "
    ).strip()
    return [
        word
        for idx, word in enumerate(word_count.keys(), 1)
        if str(idx) in selected_words.split(",")
    ]


def generate_problem_statement_choice(selected_keywords, task):
    print("\nHow would you like to generate the problem statement?")
    print("a. Normal NLG")
    print("b. AI-Generated (LLaMA Model)")
    ps_choice = input("Enter your choice (a/b): ").strip().lower()
    if ps_choice == "a":
        problem_statement = generate_problem_statement(selected_keywords, task)
        print("\nGenerated Problem Statement:")
        print(problem_statement)
    elif ps_choice == "b":
        problem_statement = ai_generate_problem_statement(selected_keywords)
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
        print(f"\nLoaded dataset for {task}: {dataset_file}")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Dataset file for {task} not found. Exiting program.")
        return None


def perform_eda(df, task):
    while True:
        print("\nWhat analysis would you like to perform?")
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

    if choice == "a":
        handle_story()
    elif choice == "b":
        handle_news()
    elif choice == "c":
        handle_document()
    else:
        print("Invalid choice. Exiting program.")
        return

    task = select_task()

    word_count = {}  
    selected_keywords = select_keywords(word_count)

    generate_problem_statement_choice(selected_keywords, task)

    df = load_dataset(task)
    if df is None:
        return

    perform_eda(df, task)


if __name__ == "__main__":
    main()