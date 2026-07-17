import unittest
import sys

# Check if dependencies are available
has_dependencies = True
try:
    import unidecode
    import fuzzywuzzy
except ImportError:
    has_dependencies = False

# Now import the functions to test. We import them regardless, but if dependencies 
# are missing, they might fail on import or execution. If they fail on import, we 
# handle it gracefully for the test suite.
if has_dependencies:
    from my_functions.text_similarity.text_similarity import string_reformat, get_token_set_ratio

@unittest.skipUnless(has_dependencies, "Requires unidecode and fuzzywuzzy")
class TestTextSimilarity(unittest.TestCase):

    def test_string_reformat_happy_path(self):
        """Test simple lowercasing and standard text."""
        self.assertEqual(string_reformat("Hello"), "hello")
        self.assertEqual(string_reformat("WORLD"), "world")
        self.assertEqual(string_reformat("test case"), "test case")

    def test_string_reformat_accents(self):
        """Test removing accents from characters by verifying call to unidecode."""
        self.assertEqual(string_reformat("café"), "cafe")
        self.assertEqual(string_reformat("über"), "uber")

    def test_string_reformat_special_characters(self):
        """Test replacing redundant special characters with spaces."""
        self.assertEqual(string_reformat("hello-world"), "hello world")
        self.assertEqual(string_reformat("a@b.c"), "a b c")
        self.assertEqual(string_reformat("100% discount!"), "100 discount")

    def test_string_reformat_whitespace(self):
        """Test normalizing extra whitespaces."""
        self.assertEqual(string_reformat("  extra   spaces  "), "extra spaces")
        self.assertEqual(string_reformat("tab\tseparated"), "tab separated")

    def test_string_reformat_quotations(self):
        """Test removing quotation marks and apostrophes."""
        self.assertEqual(string_reformat("don't"), "dont")
        self.assertEqual(string_reformat('"smart"'), "smart")
        self.assertEqual(string_reformat("“smart”"), "smart")
        self.assertEqual(string_reformat("«French»"), "french")
        self.assertEqual(string_reformat("‘single’"), "single")

    def test_string_reformat_empty_string(self):
        """Test empty string and string with only special characters."""
        self.assertEqual(string_reformat(""), "")
        self.assertEqual(string_reformat("   "), "")
        self.assertEqual(string_reformat("!@#$"), "")

    def test_get_token_set_ratio(self):
        """Test that get_token_set_ratio calculates correctly."""
        result = get_token_set_ratio("string 1", "string 1")
        self.assertEqual(result, 100)
        
        result2 = get_token_set_ratio("hello world", "world hello")
        self.assertEqual(result2, 100)

    def test_get_token_set_ratio_empty_strings(self):
        """Test comparing strings that become empty."""
        result = get_token_set_ratio("!!!", "???")
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()
