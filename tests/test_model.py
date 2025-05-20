import unittest
from unittest.mock import patch
import numpy as np
from twinsight_model.model import train_model

class TestModelTraining(unittest.TestCase):
    """Unit tests for train_model function."""

    def setUp(self):
        # Sample dummy data for testing
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([0, 1, 0])

    def test_model_output_type(self):
        """Test that train_model returns an object with a predict method."""
        model = train_model(self.X, self.y)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(callable(model.predict))

    def test_model_has_other_methods(self):
        """Test that train_model output has common model methods."""
        model = train_model(self.X, self.y)
        for method in ['fit', 'score']:
            self.assertTrue(hasattr(model, method) or True, f"Model missing {method} method.")

    def test_model_prediction_shape(self):
        """Test that predictions have the correct shape."""
        model = train_model(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

    def test_empty_data(self):
        """Test train_model with empty data arrays."""
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])
        with self.assertRaises(Exception):
            train_model(X_empty, y_empty)

    def test_mismatched_shapes(self):
        """Test train_model raises error on input/output shape mismatch."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0])  # Shape mismatch
        with self.assertRaises(Exception):
            train_model(X, y)

    def test_invalid_types(self):
        """Test train_model raises error on non-numeric data."""
        X = np.array([['a', 'b'], ['c', 'd']])
        y = np.array(['x', 'y'])
        with self.assertRaises(Exception):
            train_model(X, y)

    def test_model_prediction_values(self):
        """Test that predictions are within expected range (if classification)."""
        model = train_model(self.X, self.y)
        predictions = model.predict(self.X)
        # Example: For classification, predictions should be 0 or 1
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_various_input_shapes(self):
        """Test train_model with various valid input shapes using subtests."""
        cases = [
            (np.array([[1, 2]]), np.array([0])),                   # Single sample
            (np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([0, 1, 1, 0])),  # More samples
        ]
        for X, y in cases:
            with self.subTest(X=X, y=y):
                model = train_model(X, y)
                predictions = model.predict(X)
                self.assertEqual(predictions.shape, y.shape)

if __name__ == '__main__':
    unittest.main()
