import unittest
from unittest.mock import patch, mock_open
import bcrypt

from my_functions.Ongoing.Complete_Function import get_config

class TestCompleteFunction(unittest.TestCase):
    @patch('builtins.input', side_effect=['test_user', 'secret', 'test_key'])
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('my_functions.Ongoing.Complete_Function.MAIN_DIR', '/tmp')
    def test_get_config_success(self, mock_json_load, mock_file, mock_input):
        # Create a real hashed password for 'secret'
        hashed_pw = bcrypt.hashpw(b'secret', bcrypt.gensalt()).decode('utf-8')
        mock_json_load.return_value = {
            'test_user': hashed_pw,
            'test_key': 'test_value'
        }

        result = get_config()
        self.assertEqual(result, 'test_value')

    @patch('builtins.input', side_effect=['test_user', 'wrong_secret', 'test_key'])
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('my_functions.Ongoing.Complete_Function.MAIN_DIR', '/tmp')
    @patch('builtins.print')
    def test_get_config_wrong_password(self, mock_print, mock_json_load, mock_file, mock_input):
        hashed_pw = bcrypt.hashpw(b'secret', bcrypt.gensalt()).decode('utf-8')
        mock_json_load.return_value = {
            'test_user': hashed_pw,
            'test_key': 'test_value'
        }

        result = get_config()
        self.assertIsNone(result)
        mock_print.assert_called_with("Please let it go!")

    @patch('builtins.input', side_effect=['wrong_user', 'secret', 'test_key'])
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('my_functions.Ongoing.Complete_Function.MAIN_DIR', '/tmp')
    @patch('builtins.print')
    def test_get_config_wrong_user(self, mock_print, mock_json_load, mock_file, mock_input):
        hashed_pw = bcrypt.hashpw(b'secret', bcrypt.gensalt()).decode('utf-8')
        mock_json_load.return_value = {
            'test_user': hashed_pw,
            'test_key': 'test_value'
        }

        result = get_config()
        self.assertIsNone(result)
        mock_print.assert_called_with("Please let it go!")

    @patch('builtins.input', side_effect=['test_user', 'secret', 'test_key'])
    @patch('builtins.open', side_effect=IOError)
    @patch('my_functions.Ongoing.Complete_Function.MAIN_DIR', '/tmp')
    @patch('builtins.print')
    def test_get_config_ioerror(self, mock_print, mock_file, mock_input):
        with self.assertRaises(UnboundLocalError):
            # In the function, if IOError occurs, users_dict is not defined,
            # and it will crash at: hashed = users_dict.get(username)
            get_config()

if __name__ == '__main__':
    unittest.main()
