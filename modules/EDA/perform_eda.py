import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(dataset):
    while True:
        print("\nChoose a visualization or analysis option:")
        print("1. Univariate Analysis (Histogram)")
        print("2. Univariate Analysis (Boxplot)")
        print("3. Bivariate Analysis (Scatter Plot)")
        print("4. Bivariate Analysis (Line Plot)")
        print("5. Multivariate Analysis (Correlation Heatmap)")
        print("6. Multivariate Analysis (Pairplot)")
        print("7. Value Counts for a Column")
        print("8. Quit EDA")
        choice = input("Enter the number corresponding to your choice: ")

        if choice == "1":
            column = input("Enter the column name for the histogram: ")
            if column in dataset.columns:
                print("\nGenerating Histogram...\n")
                dataset[column].hist(bins=30, figsize=(8, 6))
                plt.title(f"Histogram for {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                plt.show()
            else:
                print("Invalid column name.")
        elif choice == "2":
            column = input("Enter the column name for the boxplot: ")
            if column in dataset.columns:
                print("\nGenerating Boxplot...\n")
                sns.boxplot(data=dataset[column])
                plt.title(f"Boxplot for {column}")
                plt.show()
            else:
                print("Invalid column name.")
        elif choice == "3":
            x_column = input("Enter the column name for the x-axis: ")
            y_column = input("Enter the column name for the y-axis: ")
            if x_column in dataset.columns and y_column in dataset.columns:
                print("\nGenerating Scatter Plot...\n")
                plt.figure(figsize=(8, 6))
                plt.scatter(dataset[x_column], dataset[y_column], alpha=0.7)
                plt.title(f"Scatter Plot: {x_column} vs {y_column}")
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.show()
            else:
                print("Invalid column names.")
        elif choice == "4":
            x_column = input("Enter the column name for the x-axis: ")
            y_column = input("Enter the column name for the y-axis: ")
            if x_column in dataset.columns and y_column in dataset.columns:
                print("\nGenerating Line Plot...\n")
                plt.figure(figsize=(8, 6))
                plt.plot(dataset[x_column], dataset[y_column], marker='o')
                plt.title(f"Line Plot: {x_column} vs {y_column}")
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.show()
            else:
                print("Invalid column names.")
        elif choice == "5":
            print("\nGenerating Correlation Heatmap...\n")
            plt.figure(figsize=(10, 8))
            sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.show()
        elif choice == "6":
            print("\nGenerating Pairplot...\n")
            sns.pairplot(dataset)
            plt.show()
        elif choice == "7":
            column = input("Enter the column name to view value counts: ")
            if column in dataset.columns:
                print("\nValue Counts:\n")
                print(dataset[column].value_counts())
            else:
                print("Invalid column name.")
        elif choice == "8":
            print("Exiting EDA.")
            break
        else:
            print("Invalid choice. Please try again.")
