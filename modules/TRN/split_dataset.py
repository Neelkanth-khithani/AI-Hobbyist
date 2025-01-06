from sklearn.model_selection import train_test_split

def split_dataset(X, y):
    while True:
        try:
            split_ratio = float(input("Enter the train-test split ratio (e.g., 0.8 for 80% training): "))
            if 0 < split_ratio < 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio), random_state=42)
                return X_train, X_test, y_train, y_test
            else:
                print("Split ratio must be a value between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a numeric value.")