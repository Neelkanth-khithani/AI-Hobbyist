import numpy as np

def handle_scaling(df):
    print("1. Round")
    print("2. Round down")
    print("3. Round up")
    print("4. Scale by min-max values")
    print("5. Normalize")
    print("6. Standardize")
    choice = input("Choose an option (1-6): ")

    if choice == '1':
        columns = input("Enter column(s) to round (comma-separated): ").split(',')
        df[columns] = df[columns].round()
    elif choice == '2':
        columns = input("Enter column(s) to round down (comma-separated): ").split(',')
        df[columns] = df[columns].apply(np.floor)
    elif choice == '3':
        columns = input("Enter column(s) to round up (comma-separated): ").split(',')
        df[columns] = df[columns].apply(np.ceil)
    elif choice == '4':
        columns = input("Enter column(s) to scale (comma-separated): ").split(',')
        for col in columns:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
    elif choice == '5':
        columns = input("Enter column(s) to normalize (comma-separated): ").split(',')
        for col in columns:
            norm = np.linalg.norm(df[col])
            if norm == 0:
                print(f"Column {col} cannot be normalized (zero norm).")
            else:
                df[col] = df[col] / norm
    elif choice == '6':
        columns = input("Enter column(s) to standardize (comma-separated): ").split(',')
        for col in columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val == 0:
                print(f"Column {col} cannot be standardized (zero standard deviation).")
            else:
                df[col] = (df[col] - mean_val) / std_val
    else:
        print("Invalid choice.")
    return df
