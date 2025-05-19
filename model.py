import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

class TwinsightModel:
    def __init__(self):
        self.model = LogisticRegression()
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of positive class
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    def load(self, path):
        self.model = joblib.load(path)
