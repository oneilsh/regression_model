import unittest
import numpy as np
from twinsight_model.model import train_model

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Sample dummy data for testing
        self.X = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([0, 1, 0])

    def test_model_output_type(self):
        model = train_model(self.X, self.y)
        # Assert that model has a predict method
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_prediction_shape(self):
        model = train_model(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, self.y.shape)

if __name__ == '__main__':
    unittest.main()
