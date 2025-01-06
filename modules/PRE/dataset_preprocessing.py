from .dataset_operations import (
    dataset_summary,
    load_data,
    export_data,
    preview_data
)
from .column_operations import handle_column_operations
from .duplicates_missing import handle_duplicates_and_missing
from .encoding import handle_encoding
from .formatting import handle_formatting
from .scaling import handle_scaling
from .remove_outliers import remove_outliers

def dataset_preprocessing():

    print("Now upload your dataset for performing pre-processing")

    file_path = input("\nEnter the path to your CSV file: ")
    df = load_data(file_path)
    if df is None:
        return

    while True:
        print("\nData Cleaning Operations:")
        print("1. Duplicate & Missing")
        print("2. Format")
        print("3. Encoding")
        print("4. Scaling")
        print("5. Column Operation")
        print("6. Outlier Removal") 
        print("7. Export Data")
        print("8. Preview Data")
        print("9. Data Summary")
        print("10. Exit")

        choice = input("Choose an option (1-10): ")

        if choice == '1':
            df = handle_duplicates_and_missing(df)
        elif choice == '2':
            df = handle_formatting(df)
        elif choice == '3':
            df = handle_encoding(df)
        elif choice == '4':
            df = handle_scaling(df)
        elif choice == '5':
            df = handle_column_operations(df)
        elif choice == '6':  
            df = remove_outliers(df)
        elif choice == '7':
            export_data(df)
        elif choice == '8':
            preview_data(df)
        elif choice == '9':
            dataset_summary(df)
        elif choice == '10':
            break
        else:
            print("Invalid choice.")