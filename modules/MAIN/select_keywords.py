def select_keywords(word_count):
    try:
        selected_words = input("Dear Hobbyist, please enter the words, problems, or issues for which you want to generate a problem statement.: ").strip()
        selected_indices = list(map(int, selected_words.split(",")))
        selected = [word for idx, word in enumerate(word_count.keys(), start=1) if idx in selected_indices]
        return selected
    except ValueError:
        print("Invalid input for keywords. Please use comma-separated numbers corresponding to the indices.")
        return []