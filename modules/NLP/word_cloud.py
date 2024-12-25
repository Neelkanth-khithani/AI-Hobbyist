import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_word_cloud(text_data):

    wordcloud = WordCloud(
        width=800, height=400, background_color="white",
        collocations=False, min_font_size=10
    ).generate(text_data)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def bag_of_words_menu(word_count):

    text_data = " ".join([f"{word} " * count for word, count in word_count.items()])

    generate_word_cloud(text_data)

    for idx, word in enumerate(word_count.keys(), 1):
        print(f"{idx}. {word}")