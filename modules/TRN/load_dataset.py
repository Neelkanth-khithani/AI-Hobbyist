import pandas as pd

def load_dataset():
    file_path = input("Enter the path to your cleaned dataset (CSV format): ")
    try:
        dataset = pd.read_csv(file_path)
        print("\nDataset successfully loaded!")
        print(f"\nColumns in the dataset: {list(dataset.columns)}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None