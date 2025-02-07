import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    confusion_matrix, f1_score, recall_score, precision_score, 
    classification_report, mean_absolute_percentage_error
)
from django.shortcuts import render
from django.http import FileResponse, Http404
from io import StringIO
import os
import joblib

def save_model(model, model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)

    filename = f"{model_name}.joblib"
    filepath = os.path.join(models_dir, filename)
    joblib.dump(model, filepath)
    return filename

def download_model(request, filename):
    models_dir = os.path.join(os.getcwd(), 'models')
    filepath = os.path.join(models_dir, filename)
    
    if not os.path.exists(filepath):
        raise Http404("Model file not found")
        
    return FileResponse(open(filepath, 'rb'), as_attachment=True, filename=filename)

def process_data(dataset_str, target_variable):
    dataset = pd.read_csv(StringIO(dataset_str))
    
    le = LabelEncoder()
    for col in dataset.select_dtypes(include=['object']).columns:
        if col != target_variable:
            dataset[col] = le.fit_transform(dataset[col].astype(str))
    
    if dataset[target_variable].dtype == 'object':
        dataset[target_variable] = le.fit_transform(dataset[target_variable])
    
    feature_names = [col for col in dataset.columns if col != target_variable]
    X = dataset.drop(target_variable, axis=1)
    y = dataset[target_variable]
    
    return X, y, dataset, feature_names

def train_and_evaluate_regression(preprocessor, X_train, X_test, y_train, y_test, model_type):
    models = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1)
    }
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', models[model_type])
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mape': mean_absolute_percentage_error(y_test, y_pred),
        'cv_score': np.mean(cross_val_score(model, X_train, y_train, cv=5))
    }
    
    return model, metrics

def train_and_evaluate_classification(preprocessor, X_train, X_test, y_train, y_test, model_type):
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'svm': SVC(probability=True)
    }
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', models[model_type])
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return model, metrics

def get_feature_importances(model, feature_names):
    try:
        importances = model.named_steps['regressor'].feature_importances_
        return sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    except:
        return None