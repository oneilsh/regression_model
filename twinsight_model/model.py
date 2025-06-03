import logging
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score
)

logging.basicConfig(level=logging.INFO)

def train_model(X_train, y_train, **kwargs):
    """
    Train a logistic regression model with configurable parameters.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    **kwargs: Additional parameters for LogisticRegression.

    Returns:
    LogisticRegression: Trained logistic regression model.
    """
    logging.info("Initializing LogisticRegression model...")
    model = LogisticRegression(max_iter=1000, **kwargs)
    try:
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using several metrics.

    Parameters:
        model: Trained model with predict and predict_proba methods.
        X_test (array-like): Test data features.
        y_test (array-like): True test labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1, and ROC AUC scores.
    """
    preds = model.predict(X_test)
    try:
        probas = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probas)
    except Exception:
        probas = None
        roc_auc = None
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, average="binary"),
        "recall": recall_score(y_test, preds, average="binary"),
        "f1_score": f1_score(y_test, preds, average="binary"),
        "roc_auc": roc_auc
    }
    return metrics

def save_model(model, path):
    """
    Save the trained model to a file.

    Parameters:
        model: Trained model object.
        path (str): File path to save the model.
    """
    try:
        joblib.dump(model, path)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Failed to save model: {e}")
        raise

def load_model(path):
    """
    Load a model from a file.

    Parameters:
        path (str): File path to load the model from.

    Returns:
        The loaded model object.
    """
    try:
        model = joblib.load(path)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
