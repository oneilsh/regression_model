# twinsight_model/preprocessing.py

from sklearn.model_selection import train_test_split

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
