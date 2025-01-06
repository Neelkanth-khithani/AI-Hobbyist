import pandas as pd

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def preview_data(df):
    print(df)

def export_data(df):
    file_path = input("Enter location to save the cleaned dataset (CSV): ")
    try:
        df.to_csv(file_path, index=False)
        print(f"Data exported to {file_path}")
    except Exception as e:
        print(f"Error exporting data: {e}")

def dataset_summary(df):
    print("1. Column summary")
    print("2. Row summary")
    print("3. General info about the dataset")
    summary_choice = input("Choose an option (1-3): ")

    if summary_choice == '1':
        print(df.describe(include='all'))
    elif summary_choice == '2':
        print(df.T) 
    elif summary_choice == '3':
        print("\nDataset info:")
        print(df.info())

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nUnique Values in Columns:")
        print(df.nunique())

    else:
        print("Invalid choice.")