# eda.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from io import StringIO

def detect_delimiter(text):
    first_line = text.splitlines()[0] 
    possible_delimiters = [',', ';', '\t', '|', ':', '#', ' ']
    delimiter_count = {delim: first_line.count(delim) for delim in possible_delimiters}
    return max(delimiter_count, key=delimiter_count.get)

def convert_text_to_dataframe(input_text, header=True):
    input_text = input_text.replace('\r\n', '\n')  
    
    delimiter = detect_delimiter(input_text)
    print(f"Detected delimiter: '{delimiter}'")
    
    df = pd.read_csv(StringIO(input_text), delimiter=delimiter)
    
    if not header:
        df.columns = [f"Column{i}" for i in range(1, len(df.columns) + 1)]
    
    return df

def handle_missing_values(df, placeholder_values=None):
    if placeholder_values is None:
        placeholder_values = ['Unknown', 'Not Provided', 'N/A', 'na', 'none', 'null', '']

    notes = {}  
    for col in df.columns:
        df[col].replace(placeholder_values, pd.NA, inplace=True)

        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object': 
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                notes[col] = f"Filled missing values with mode: '{mode_value}'"
            else:  
                skewness = df[col].skew()
                if abs(skewness) > 1:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    notes[col] = f"Filled missing values with median: {median_value}"
                else:
                    mean_value = df[col].mean()
                    df[col].fillna(mean_value, inplace=True)
                    notes[col] = f"Filled missing values with mean: {mean_value}"

    return df, notes

def handle_missing_data(dataset: pd.DataFrame, placeholder_values=None):

    if placeholder_values is None:
        placeholder_values = ['Unknown', 'Not Provided', 'N/A', 'na', 'none', 'null', '']

    null_values = dataset.isnull().sum()

    placeholder_data = dataset.isin(placeholder_values).sum()

    missing_percentage = (null_values + placeholder_data) / len(dataset) * 100

    missing_data_summary = {
        'null_values': null_values,
        'placeholder_data': placeholder_data,
        'missing_percentage': missing_percentage
    }

    missing_columns = dataset.columns[missing_percentage > 0]
    
    return missing_data_summary, missing_columns


def handle_categorical_encoding(df, target_column=None):
    categorical_summary = {}
    encoded_df = df.copy()
    columns_to_encode = []

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            columns_to_encode.append(col)

    for col in columns_to_encode:
        unique_values = df[col].nunique()
        unique_items = df[col].unique().tolist()

        if col == target_column:
            encoder = LabelEncoder()
            encoded_df[col] = encoder.fit_transform(df[col].astype(str))
            encoding_method = 'label encoding (target column)'
            note = "Performed label encoding as this column was marked as the target"
        elif unique_values <= 3:
            temp_df = pd.get_dummies(encoded_df[[col]], columns=[col], prefix=col, dtype='int32')
            encoded_df = encoded_df.drop(columns=[col])
            encoded_df = pd.concat([encoded_df, temp_df], axis=1)
            encoding_method = 'one-hot encoding'
            note = f"Performed one-hot encoding as column had {unique_values} unique values"
        else:
            encoder = LabelEncoder()
            encoded_df[col] = encoder.fit_transform(df[col].astype(str))
            encoding_method = 'label encoding'
            note = f"Performed label encoding as column had {unique_values} unique values"

        categorical_summary[col] = {
            'unique_values': unique_values,
            'unique_items': unique_items,
            'encoding_method': encoding_method,
            'note': note
        }

    return encoded_df, categorical_summary

def is_contiguous_sequence(series):
 
    clean_series = series.dropna()
    
    if len(clean_series) <= 1:
        return False
        
    numeric_series = pd.to_numeric(clean_series, errors='coerce')
    
    numeric_series = numeric_series.dropna()
    
    sorted_values = numeric_series.sort_values().reset_index(drop=True)
    
    differences = sorted_values.diff()
    
    differences = differences.dropna()
    
    return len(differences) > 0 and all(differences == 1)

def remove_contiguous_columns(dataset):

    if dataset is None or dataset.empty:
        return dataset
        
    removed_columns = []
    
    df = dataset.copy()
    
    for col in df.columns:
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            
            if numeric_series.notna().mean() > 0.5:
                if is_contiguous_sequence(numeric_series):
                    removed_columns.append(col)
                    df = df.drop(columns=[col])
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            continue
    
    if removed_columns:
        print(f"Removed contiguous columns: {removed_columns}")
    
    return df

def handle_outliers_auto(df, params={}):
    visualization_details = {}

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        skewness = df[col].skew()
        print(f"Skewness of {col}: {skewness}")

        method = "iqr" if abs(skewness) > 0.5 else "zscore"
        print(f"Method chosen for {col}: {method}")

        if method == "zscore":
            threshold = params.get("threshold", 3) 
            df['Z-Score'] = (df[col] - df[col].mean()) / df[col].std()

            outliers = df[np.abs(df['Z-Score']) > threshold]
            visualization_details[col] = (df.drop(columns=['Z-Score']), outliers, "Z-Score Method")

            df = df[np.abs(df['Z-Score']) <= threshold]
            df.drop(columns=['Z-Score'], inplace=True)

        elif method == "iqr":
            k = params.get("k", 1.5)  
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            visualization_details[col] = (df, outliers, "IQR Method")

            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df, visualization_details


def handle_scaling_auto(df, params={}):

    scaling_details = {}

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        column_range = df[col].max() - df[col].min()
        method = "StandardScaler (Z-Score)" if column_range > 100 else "Min-Max Scaler"

        if method == "StandardScaler (Z-Score)":
            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scaling_details[col] = {
                "range": column_range,
                "scaling_method": "StandardScaler (Z-Score)"
            }

        elif method == "Min-Max Scaler":
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scaling_details[col] = {
                "range": column_range,
                "scaling_method": "Min-Max Scaler"
            }

    return df, scaling_details