from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from .save_model import save_model
from .split_dataset import split_dataset

def regression_task(dataset):
    target_column = input("Enter the name of the target column for regression: ")
    if target_column not in dataset.columns:
        print("Invalid target column.")
        return

    X = dataset.drop(columns=[target_column])
    y = dataset[target_column]

    scale_features = input("Would you like to scale the features? (yes/no): ").lower()
    if scale_features == 'yes':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("\nChoose a regression model:")
    print("1. Simple Linear Regression")
    print("2. Multiple Linear Regression")
    print("3. Decision Tree Regression")
    print("4. Random Forest Regression")
    model_choice = input("Enter the number corresponding to your choice: ")

    if model_choice == "1":
        model = LinearRegression()
    elif model_choice == "2":
        model = LinearRegression()  
    elif model_choice == "3":
        model = DecisionTreeRegressor()
    elif model_choice == "4":
        model = RandomForestRegressor()
    else:
        print("Invalid choice. Exiting regression task.")
        return

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = model.score(X_test, y_test)
        print(f"\nModel trained successfully! MSE: {mse:.2f}, R²: {r2:.2f}")

        plt.figure(figsize=(8, 6))
        residuals = y_test - y_pred
        plt.scatter(y_test, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title("Residual Plot")
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.title("Actual vs Predicted Values")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.show()

        save_model(model)
    except Exception as e:
        print(f"Error during model training or prediction: {e}")