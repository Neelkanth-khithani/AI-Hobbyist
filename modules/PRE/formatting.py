def handle_formatting(df):
    print("1. Capitalize first character")
    print("2. Convert text to lowercase")
    print("3. Convert text to uppercase")
    print("4. Strip white space")
    choice = input("Choose an option (1-4): ")

    if choice == '1':
        columns = input("Enter column(s) to capitalize (comma-separated): ").split(',')
        df[columns] = df[columns].apply(lambda x: x.str.capitalize())
    elif choice == '2':
        columns = input("Enter column(s) to convert to lowercase (comma-separated): ").split(',')
        df[columns] = df[columns].apply(lambda x: x.str.lower())
    elif choice == '3':
        columns = input("Enter column(s) to convert to uppercase (comma-separated): ").split(',')
        df[columns] = df[columns].apply(lambda x: x.str.upper())
    elif choice == '4':
        columns = input("Enter column(s) to strip white space (comma-separated): ").split(',')
        df[columns] = df[columns].apply(lambda x: x.str.strip())
    else:
        print("Invalid choice.")
    return df
