def handle_column_operations(df):
    print("1. Change column type")
    print("2. Clone column")
    print("3. Drop column")
    print("4. Rename column")
    choice = input("Choose an option (1-4): ")

    if choice == '1':
        column = input("Enter column to change type: ")
        new_type = input("Enter new type (e.g., int, float, str): ")
        df[column] = df[column].astype(new_type)
    elif choice == '2':
        column = input("Enter column to clone: ")
        new_column_name = input("Enter new column name: ")
        df[new_column_name] = df[column]
    elif choice == '3':
        column = input("Enter column to drop: ")
        df.drop(columns=[column], inplace=True)
    elif choice == '4':
        column = input("Enter column to rename: ")
        new_name = input("Enter new column name: ")
        df.rename(columns={column: new_name}, inplace=True)
    else:
        print("Invalid choice.")
    return df