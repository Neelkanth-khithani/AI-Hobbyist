from pycaret.classification import *
from pycaret.regression import *
import pandas as pd

def classification(dataset_path, target):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()
    s = setup(data, target=target, session_id=123)
    best = compare_models()
    pred_holdout = predict_model(best)
    new_data = data.copy().drop(target, axis=1)
    predictions = predict_model(best, data=new_data)
    save_model(best, 'classification_model')

def Regression(dataset_path, target):
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.str.strip()
    s = setup(data, target=target, session_id=123)
    best = compare_models()
    print(best)
    evaluate_model(best)
    pred_holdout = predict_model(best)
    new_data = data.copy().drop(target, axis=1)
    predictions = predict_model(best, data=new_data)
    save_model(best, 'regression_model')