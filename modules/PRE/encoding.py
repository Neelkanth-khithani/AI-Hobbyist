import pandas as pd

def handle_encoding(df):
    print("1. One-hot encode")
    print("2. Label encoding")
    choice = input("Choose an option (1-2): ")

    if choice == '1':
        columns = input("Enter column(s) to one-hot encode (comma-separated): ").split(',')
        df = pd.get_dummies(df, columns=columns)
    elif choice == '2':
        columns = input("Enter column(s) to label encode (comma-separated): ").split(',')
        for column in columns:
            df[column] = df[column].astype('category').cat.codes
    else:
        print("Invalid choice.")
    return df