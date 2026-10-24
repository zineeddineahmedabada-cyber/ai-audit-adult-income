import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import urllib.request

# URLs for the Adult Income dataset
URL_TRAIN = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
URL_TEST = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# Column names based on the dataset documentation
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]

def download_and_load_data():
    print("Downloading dataset...")
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    train_path = 'data/raw/adult.data'
    test_path = 'data/raw/adult.test'
    
    # Download if not exists
    if not os.path.exists(train_path):
        urllib.request.urlretrieve(URL_TRAIN, train_path)
    if not os.path.exists(test_path):
        urllib.request.urlretrieve(URL_TEST, test_path)

    # Load data (skipinitialspace removes the leading spaces in the CSV)
    df_train = pd.read_csv(train_path, names=COLUMNS, skipinitialspace=True)
    df_test = pd.read_csv(test_path, names=COLUMNS, skipinitialspace=True, skiprows=1) # test file has a header row to skip
    
    return df_train, df_test

def clean_data(df):
    # Replace '?' with NaN and drop missing values
    df = df.replace('?', np.nan).dropna()
    
    # Clean the target variable (remove trailing dots in test set and spaces)
    df['income'] = df['income'].str.replace('.', '', regex=False).str.strip()
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    
    return df

def preprocess_features(df_train, df_test):
    # Separate features and target
    X_train = df_train.drop('income', axis=1)
    y_train = df_train['income']
    
    X_test = df_test.drop('income', axis=1)
    y_test = df_test['income']
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numerical_cols = X_train.select_dtypes(exclude=['object']).columns
    
    # We will use LabelEncoder for simplicity (Random Forest handles this well)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Fit on train, transform both (using a trick to handle unseen labels in test)
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        
        # Handle test set (map unseen labels to -1)
        X_test[col] = X_test[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
        label_encoders[col] = le
        
    return X_train, y_train, X_test, y_test, label_encoders

def main():
    print("Starting data preparation...")
    df_train_raw, df_test_raw = download_and_load_data()
    
    print(f"Initial train shape: {df_train_raw.shape}")
    
    df_train_clean = clean_data(df_train_raw)
    df_test_clean = clean_data(df_test_raw)
    
    print(f"Clean train shape: {df_train_clean.shape} (rows dropped due to missing values)")
    
    X_train, y_train, X_test, y_test, encoders = preprocess_features(df_train_clean, df_test_clean)
    
    # Save processed data for the modeling step
    os.makedirs('data/processed', exist_ok=True)
    train_processed = X_train.copy()
    train_processed['income'] = y_train
    train_processed.to_csv('data/processed/train_processed.csv', index=False)
    
    test_processed = X_test.copy()
    test_processed['income'] = y_test
    test_processed.to_csv('data/processed/test_processed.csv', index=False)
    
    print("Data preparation complete. Files saved to data/processed/")

if __name__ == "__main__":
    main()