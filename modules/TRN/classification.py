from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .save_model import save_model
from .split_dataset import split_dataset
import pandas as pd

def classification_task(dataset):
    target_column = input("Enter the name of the target column for classification: ")
    if target_column not in dataset.columns:
        print("Invalid target column.")
        return

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nChoose a classification model:")
    print("1. Logistic Regression")
    print("2. Decision Tree Classification")
    print("3. Random Forest Classification")
    model_choice = input("Enter the number corresponding to your choice: ")

    if model_choice == "1":
        model = LogisticRegression()
    elif model_choice == "2":
        model = DecisionTreeClassifier()
    elif model_choice == "3":
        model = RandomForestClassifier()
    else:
        print("Invalid choice. Exiting classification task.")
        return

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel trained successfully! Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    additional_vis = input("Do you want to generate visualizations for predicted vs actual values? (yes/no): ").lower()
    if additional_vis == "yes":
        # Create a DataFrame for comparison
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        print("\nSample comparison of actual and predicted values:")
        print(results_df.head())

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=results_df.index, y="Actual", data=results_df, label="Actual", color="blue")
        sns.scatterplot(x=results_df.index, y="Predicted", data=results_df, label="Predicted", color="orange")
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Sample Index")
        plt.ylabel("Class")
        plt.legend()
        plt.show()

        # Count plot to compare distributions
        plt.figure(figsize=(10, 6))
        sns.countplot(x="value", hue="variable", data=pd.melt(results_df.reset_index(), id_vars=["index"]))
        plt.title("Distribution of Actual vs Predicted Classes")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.legend(title="Category")
        plt.show()

    save_model(model)