def select_task():
    while True:
        print("\nChoose how would you like your problem statement to begin with?")
        print("1. To Classify...")
        print("2. To Regress...")
        task_choice = input("Enter your choice (1/2): ").strip()
        task_mapping = {"1": "classify", "2": "regress"}
        task = task_mapping.get(task_choice, None)
        if task:
            return task
        else:
            print("Invalid choice. Please select 1 or 2.")