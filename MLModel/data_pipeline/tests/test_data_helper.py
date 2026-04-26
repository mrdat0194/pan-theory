import unittest
import numpy as np
from unittest.mock import patch

from MLModel.data_pipeline.data_helper import get_unique, data_pipeline, data_pipeline_nn, imbalance_solve_v2, imbalance_solve

class TestDataHelper(unittest.TestCase):

    def test_get_unique_normal(self):
        x = [[1, 2, 3], [4, 5, 6], [1, 2, 3]]
        y = [1, 2, 1]
        x_unique, y_unique = get_unique(x, y)
        self.assertEqual(len(x_unique), 2)
        self.assertEqual(len(y_unique), 2)
        self.assertEqual(set(tuple(i) for i in x_unique), {(1, 2, 3), (4, 5, 6)})
        self.assertEqual(set(y_unique), {1, 2})

    def test_get_unique_empty(self):
        x = []
        y = []
        x_unique, y_unique = get_unique(x, y)
        self.assertEqual(len(x_unique), 0)
        self.assertEqual(len(y_unique), 0)

    @patch('MLModel.data_pipeline.data_helper.train_test_split')
    def test_data_pipeline(self, mock_split):
        mock_split.return_value = (np.array([[1, 2]]), np.array([[3, 4]]), np.array([0]), np.array([1]))
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([0, 1])
        X_train, X_test, Y_train, Y_test = data_pipeline(X, Y)
        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(X_test), 1)

    @patch('MLModel.data_pipeline.data_helper.train_test_split')
    @patch('MLModel.data_pipeline.data_helper.to_categorical')
    def test_data_pipeline_nn(self, mock_to_cat, mock_split):
        mock_split.side_effect = [
            (np.array([[1, 2], [5, 6]]), np.array([[3, 4]]), np.array([0, 0]), np.array([1])), # First split
            (np.array([[1, 2]]), np.array([[5, 6]]), np.array([0]), np.array([0])) # Second split
        ]
        mock_to_cat.side_effect = lambda x, num_classes: x
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        Y = np.array([0, 1, 0])
        X_train, X_test, X_val, Y_train, Y_test, Y_val = data_pipeline_nn(X, Y)
        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(X_val), 1)
        
    def test_imbalance_solve_v2(self):
        X = np.array([[0, 0], [1, 1]])
        Y = np.array([0, 1])
        X_aug1 = np.array([[2, 2]])
        Y_aug1 = np.array([0])
        X_aug2 = np.array([[3, 3]])
        Y_aug2 = np.array([0])
        
        X_final, Y_final = imbalance_solve_v2(X, Y, X_aug1, Y_aug1, X_aug2, Y_aug2)
        self.assertEqual(len(Y_final), 6)
        self.assertEqual(list(Y_final).count(1), 3)
        self.assertEqual(list(Y_final).count(0), 3)

    def test_imbalance_solve(self):
        X = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        Y = np.array([0, 1])
        X_aug1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        Y_aug1 = np.array([0])
        X_aug2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        Y_aug2 = np.array([0])
        
        X_final, Y_final = imbalance_solve(X, Y, X_aug1, Y_aug1, X_aug2, Y_aug2, rm_values=0, rm_thres=0.5)
        
        self.assertEqual(len(Y_final), 4)
        self.assertEqual(list(Y_final).count(1), 3)
        self.assertEqual(list(Y_final).count(0), 1)

if __name__ == '__main__':
    unittest.main()
