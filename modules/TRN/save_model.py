import joblib
import os

def save_model(model):
    save_option = input("Do you want to save the trained model? (yes/no): ").lower()
    if save_option == "yes":
        folder_path = input("Enter the folder path where you want to save the model: ")
        model_name = input("Enter the name you want for your model (without extension): ").strip()
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            model_path = os.path.join(folder_path, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved successfully at {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")