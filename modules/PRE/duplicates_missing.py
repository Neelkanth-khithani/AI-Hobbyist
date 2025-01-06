from sklearn.impute import KNNImputer
import pandas as pd

def handle_duplicates_and_missing(df):
    print("1. Drop duplicate rows")
    print("2. Drop missing values")
    print("3. Fill missing values")
    print("4. Find and replace values")
    print("5. Fill missing values using KNN Imputer")  # New option for KNN Imputer
    choice = input("Choose an option (1-5): ")

    if choice == '1':
        columns = input("Enter column(s) to check duplicates (comma-separated): ").split(',')
        columns = [col.strip() for col in columns]
        df.drop_duplicates(subset=columns, inplace=True)
    elif choice == '2':
        columns = input("Enter column(s) to drop missing values (comma-separated): ").split(',')
        columns = [col.strip() for col in columns]
        df.dropna(subset=columns, inplace=True)
    elif choice == '3':
        print("1. Fill with a specific value")
        print("2. Fill with mean")
        print("3. Fill with median")
        print("4. Fill with mode")
        fill_choice = input("Choose a fill option (1-4): ")

        columns = input("Enter column(s) to fill missing values (comma-separated): ").split(',')
        columns = [col.strip() for col in columns]

        if fill_choice == '1':
            fill_value = input("Enter value to fill missing entries: ")

            for column in columns:
                if df[column].dtype == 'object':
                    df[column] = df[column].fillna(str(fill_value))
                else:
                    df[column] = df[column].fillna(fill_value)
        elif fill_choice == '2':
            for column in columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    mean_value = df[column].mean()
                    df[column] = df[column].fillna(mean_value)
                else:
                    print(f"Column '{column}' is not numeric, skipping mean fill.")
        elif fill_choice == '3':
            for column in columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                else:
                    print(f"Column '{column}' is not numeric, skipping median fill.")
        elif fill_choice == '4':
            for column in columns:
                try:
                    mode_value = df[column].mode()[0]
                    df[column] = df[column].fillna(mode_value)
                except IndexError:
                    print(f"Column '{column}' has no mode value, skipping.")
        else:
            print("Invalid choice.")
    elif choice == '4':
        column = input("Enter column to find and replace: ").strip()
        old_value = input("Enter value to find: ")
        new_value = input("Enter value to replace with: ")
        df[column] = df[column].replace(old_value, new_value)
    elif choice == '5':  
        print("Using KNN Imputer for missing value handling.")
        try:
            n_neighbors = int(input("Enter the number of neighbors (k): "))
            columns = input("Enter column(s) to apply KNN Imputer (comma-separated): ").split(',')
            columns = [col.strip() for col in columns]

            numeric_df = df[columns].select_dtypes(include=[float, int])
            if numeric_df.isnull().sum().sum() == 0:
                print("No missing values found in the selected columns.")
            elif numeric_df.empty:
                print("No numeric columns selected. Please choose numeric columns for KNN Imputation.")
            else:
                knn_imputer = KNNImputer(n_neighbors=n_neighbors)
                imputed_data = knn_imputer.fit_transform(numeric_df)
                df[columns] = imputed_data
        except ValueError as e:
            print(f"Error: {e}")
        except KeyError:
            print("One or more columns entered do not exist in the DataFrame.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    else:
        print("Invalid choice.")
    return df