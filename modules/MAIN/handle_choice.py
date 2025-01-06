from modules.NLP.text_processing import clean, extract_pos
from modules.NLP.word_cloud import bag_of_words_menu
from modules.NLP.fetch_news import fetch_sustainability_news
from modules.RAG.pdf_process import extract_text_from_pdf
from modules.RAG.embeddings import generate_embeddings
from modules.RAG.doc_retrieval import retrieve_top_k_documents
from modules.RAG.rag_model import generate_answer

def handle_story():
    story = input("Share your problem story with us: ").strip()
    cleaned_story = clean(story)
    print("\nCrafting the Bag of Words from your story...")
    words = extract_pos(cleaned_story, pos_types=("NN"))
    word_count = {word: cleaned_story.split().count(word) for word in words}
    bag_of_words_menu(word_count)
    return word_count

def handle_news():
    print("\nGathering the latest news for you...\n")
    news_list = fetch_sustainability_news()
    if not news_list:
        print("No news available at the moment. Please try again later.")
        return
    print("\nSelect a news article:")
    for idx, (title) in enumerate(news_list, 1):
        print(f"{idx}.  {title}", "\n")
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
    words = extract_pos(cleaned_news, pos_types=("NN"))
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
        words = extract_pos(cleaned_document, pos_types=("NN"))
        word_count = {word: cleaned_document.split().count(word) for word in words}
        bag_of_words_menu(word_count)
    except Exception as e:
        print(f"Error in document processing: {e}")
    return word_count