import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock dependencies to handle environments where they might be missing
mock_unidecode = MagicMock()
sys.modules['unidecode'] = mock_unidecode

mock_fuzz = MagicMock()
mock_fuzzywuzzy = MagicMock()
mock_fuzzywuzzy.fuzz = mock_fuzz
sys.modules['fuzzywuzzy'] = mock_fuzzywuzzy

# Now import the functions to test
from my_functions.text_similarity.text_similarity import string_reformat, get_token_set_ratio

class TestTextSimilarity(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test
        mock_unidecode.unidecode.reset_mock()
        mock_fuzz.token_set_ratio.reset_mock()

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_string_reformat_happy_path(self, mock_uni):
        """Test simple lowercasing and standard text."""
        mock_uni.side_effect = lambda x: x
        self.assertEqual(string_reformat("Hello"), "hello")
        self.assertEqual(string_reformat("WORLD"), "world")
        self.assertEqual(string_reformat("test case"), "test case")

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_string_reformat_accents(self, mock_uni):
        """Test removing accents from characters by verifying call to unidecode."""
        mock_uni.side_effect = lambda x: x.replace('é', 'e').replace('ü', 'u')
        self.assertEqual(string_reformat("café"), "cafe")
        self.assertEqual(string_reformat("über"), "uber")
        mock_uni.assert_any_call("café")
        mock_uni.assert_any_call("über")

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_string_reformat_special_characters(self, mock_uni):
        """Test replacing redundant special characters with spaces."""
        mock_uni.side_effect = lambda x: x
        self.assertEqual(string_reformat("hello-world"), "hello world")
        self.assertEqual(string_reformat("a@b.c"), "a b c")
        self.assertEqual(string_reformat("100% discount!"), "100 discount")

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_string_reformat_whitespace(self, mock_uni):
        """Test normalizing extra whitespaces."""
        mock_uni.side_effect = lambda x: x
        self.assertEqual(string_reformat("  extra   spaces  "), "extra spaces")
        self.assertEqual(string_reformat("tab\tseparated"), "tab separated")

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_string_reformat_quotations(self, mock_uni):
        """Test removing quotation marks and apostrophes."""
        mock_uni.side_effect = lambda x: x
        self.assertEqual(string_reformat("don't"), "dont")
        self.assertEqual(string_reformat('"smart"'), "smart")
        self.assertEqual(string_reformat("“smart”"), "smart")
        self.assertEqual(string_reformat("«French»"), "french")
        self.assertEqual(string_reformat("‘single’"), "single")

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_string_reformat_empty_string(self, mock_uni):
        """Test empty string and string with only special characters."""
        mock_uni.side_effect = lambda x: x
        self.assertEqual(string_reformat(""), "")
        self.assertEqual(string_reformat("   "), "")
        self.assertEqual(string_reformat("!@#$"), "")

    @patch('my_functions.text_similarity.text_similarity.string_reformat')
    def test_get_token_set_ratio_calls_fuzz(self, mock_reformat):
        """Test that get_token_set_ratio calls fuzzywuzzy with reformatted strings."""
        mock_reformat.side_effect = ["ref1", "ref2"]
        mock_fuzz.token_set_ratio.return_value = 100
        
        result = get_token_set_ratio("string 1", "string 2")
        
        self.assertEqual(mock_reformat.call_count, 2)
        mock_fuzz.token_set_ratio.assert_called_once_with("ref1", "ref2")
        self.assertEqual(result, 100)

    @patch('my_functions.text_similarity.text_similarity.string_reformat')
    def test_get_token_set_ratio_empty_strings(self, mock_reformat):
        """Test comparing strings that become empty."""
        mock_reformat.return_value = ""
        mock_fuzz.token_set_ratio.return_value = 0
        
        result = get_token_set_ratio("!!!", "???")
        
        mock_fuzz.token_set_ratio.assert_called_once_with("", "")
        self.assertEqual(result, 0)

    @patch('my_functions.text_similarity.text_similarity.unidecode.unidecode')
    def test_get_token_set_ratio_integration(self, mock_uni):
        """Test get_token_set_ratio with actual unmocked string_reformat functionality."""
        mock_uni.side_effect = lambda x: x
        mock_fuzz.token_set_ratio.return_value = 85

        result = get_token_set_ratio("Hello World!", "hello world")

        mock_fuzz.token_set_ratio.assert_called_once_with("hello world", "hello world")
        self.assertEqual(result, 85)

if __name__ == '__main__':
    unittest.main()
