import unittest
import sys
from unittest.mock import patch, MagicMock

# `tf.Session` is called at the module level in MLModel.model.neural_network.
# We mock tensorflow entirely for this test file to avoid import errors and side-effects.

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.mock_tf = MagicMock()
        self.mock_keras = MagicMock()
        self.modules_patcher = patch.dict('sys.modules', {
            'tensorflow': self.mock_tf,
            'keras': self.mock_keras,
            'tensorflow.python.util': MagicMock(),
            'tensorflow.python.util.deprecation': MagicMock()
        })
        self.modules_patcher.start()
        
        import MLModel.model.neural_network as nn_model
        self.nn_model = nn_model

    def tearDown(self):
        self.modules_patcher.stop()
        if 'MLModel.model.neural_network' in sys.modules:
            del sys.modules['MLModel.model.neural_network']

    def test_model_nn_initialization(self):
        # Setup mocks
        mock_model = MagicMock()
        self.mock_keras.models.Sequential.return_value = mock_model
        
        # Call the function to initialize model
        input_shape = (10,)
        n_classes = 2
        model = self.nn_model.model_nn(input_shape, n_classes)
        
        # Verify sequential model was created
        self.mock_keras.models.Sequential.assert_called_once()
        
        # Verify model.add was called twice (two dense layers)
        self.assertEqual(mock_model.add.call_count, 2)
        
        # Verify model.compile was called
        mock_model.compile.assert_called_once()
        
        args, kwargs = mock_model.compile.call_args
        self.assertEqual(kwargs['loss'], 'binary_crossentropy')
        self.assertIn('accuracy', kwargs['metrics'])

if __name__ == '__main__':
    unittest.main()
