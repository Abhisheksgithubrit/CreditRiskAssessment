import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess(df):
    df = df.copy()

    # Drop missing values
    df.dropna(inplace=True)

    # Encode categorical features
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y
