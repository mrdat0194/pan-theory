import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock keras before importing data_helper
sys.modules['keras'] = MagicMock()
sys.modules['keras.utils'] = MagicMock()

import numpy as np
import pandas as pd
from MLModel.data_pipeline.data_helper import get_unique, data_pipeline, imbalance_solve_v2, get_data, get_data_test

class TestDataHelper(unittest.TestCase):

    def test_get_unique(self):
        # Normal case with duplicates
        X = [[1, 2], [3, 4], [1, 2]]
        Y = [1, 2, 1]
        
        X_unique, Y_unique = get_unique(X, Y)
        
        self.assertEqual(len(X_unique), 2)
        self.assertEqual(len(Y_unique), 2)
        
        # Check that unique pairs are present
        pairs = set(zip([tuple(x) for x in X_unique], Y_unique))
        self.assertIn(((1, 2), 1), pairs)
        self.assertIn(((3, 4), 2), pairs)
        
    def test_get_unique_empty(self):
        X = []
        Y = []
        X_unique, Y_unique = get_unique(X, Y)
        self.assertEqual(len(X_unique), 0)
        self.assertEqual(len(Y_unique), 0)

    def test_data_pipeline(self):
        X = np.array([[i] for i in range(10)])
        Y = np.array([i % 2 for i in range(10)])
        
        X_train, X_test, Y_train, Y_test = data_pipeline(X, Y)
        
        self.assertEqual(len(X_train) + len(X_test), 10)
        self.assertEqual(len(Y_train) + len(Y_test), 10)
        self.assertEqual(len(X_test), 2)  # test_size=0.2

    @patch('pandas.read_csv')
    def test_get_data(self, mock_read_csv):
        # Create a mock DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [1.1, 2.2, 3.3],
            'label': [0, 1, 0]
        })
        mock_read_csv.return_value = df
        
        X, Y = get_data('dummy_link.csv')
        
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(Y.shape, (3,))
        self.assertTrue((X[0] == [10, 1.1]).all())
        self.assertEqual(Y[0], 0)

    @patch('pandas.read_csv')
    def test_get_data_test(self, mock_read_csv):
        # Create a mock DataFrame without label
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'feature1': [10, 20, 30],
            'feature2': [1.1, 2.2, 3.3]
        })
        mock_read_csv.return_value = df
        
        X, ID = get_data_test('dummy_link.csv')
        
        self.assertEqual(X.shape, (3, 2))
        self.assertEqual(ID.shape, (3,))
        self.assertTrue((X[0] == [10, 1.1]).all())
        self.assertEqual(ID[0], 1)

    def test_imbalance_solve_v2(self):
        # X: array of arrays. Need at least 2 features because of X_row[1]
        X = np.array([[1, 10], [2, 20], [3, 30]])
        Y = np.array([0, 0, 1])
        X_aug1 = np.array([[4, 40]])
        Y_aug1 = np.array([1])
        X_aug2 = np.array([[5, 50]])
        Y_aug2 = np.array([0])
        
        # Initial labels: 0: 3, 1: 2
        # Ratio len_label0 / len(X_label1) = 3 / 2 = 1.5 -> int(1.5) = 1
        # range(1, 1) is empty. So no augmentation is done in imbalance_solve_v2.
        
        X_final, Y_final = imbalance_solve_v2(X, Y, X_aug1, Y_aug1, X_aug2, Y_aug2)
        
        self.assertEqual(len(X_final), 5)
        self.assertEqual(len(Y_final), 5)
        
        # Test with more imbalance
        X2 = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        Y2 = np.array([0, 0, 0, 0, 0])
        X_aug1_2 = np.array([[6, 60]])
        Y_aug1_2 = np.array([1])
        X_aug2_2 = np.empty((0, 2))
        Y_aug2_2 = np.empty((0,))
        
        # Initial labels: 0: 5, 1: 1
        # Ratio len_label0 / len(X_label1) = 5 / 1 = 5
        # range(1, 5) -> ages 1, 2, 3, 4
        # for each X_label1 (which is [[6, 60]]), 4 augments
        
        X_final2, Y_final2 = imbalance_solve_v2(X2, Y2, X_aug1_2, Y_aug1_2, X_aug2_2, Y_aug2_2)
        
        self.assertEqual(len(X_final2), 10)  # 6 original + 4 augmented
        self.assertEqual(len(Y_final2), 10)
        self.assertEqual(list(Y_final2).count(1), 5) # 1 original + 4 augmented

if __name__ == '__main__':
    unittest.main()
