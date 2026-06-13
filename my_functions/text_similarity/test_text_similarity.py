import unittest
from my_functions.text_similarity.text_similarity import string_reformat, get_token_set_ratio

class TestTextSimilarity(unittest.TestCase):

    def test_string_reformat_happy_path(self):
        """Test simple lowercasing and standard text."""
        self.assertEqual(string_reformat("Hello"), "hello")
        self.assertEqual(string_reformat("WORLD"), "world")
        self.assertEqual(string_reformat("test case"), "test case")

    def test_string_reformat_accents(self):
        """Test removing accents from characters."""
        self.assertEqual(string_reformat("café"), "cafe")
        self.assertEqual(string_reformat("über"), "uber")
        self.assertEqual(string_reformat("façade"), "facade")
        self.assertEqual(string_reformat("niño"), "nino")
        self.assertEqual(string_reformat("résumé"), "resume")

    def test_string_reformat_special_characters(self):
        """Test replacing redundant special characters with spaces."""
        self.assertEqual(string_reformat("hello-world"), "hello world")
        self.assertEqual(string_reformat("a@b.c"), "a b c")
        self.assertEqual(string_reformat("test_case_1"), "test case 1")
        self.assertEqual(string_reformat("100% discount!"), "100 discount")
        self.assertEqual(string_reformat("bracket(test)"), "bracket test")

    def test_string_reformat_whitespace(self):
        """Test normalizing extra whitespaces."""
        self.assertEqual(string_reformat("  extra   spaces  "), "extra spaces")
        self.assertEqual(string_reformat("tab\tseparated"), "tab separated")
        self.assertEqual(string_reformat("newline\ntext"), "newline text")

    def test_string_reformat_quotations(self):
        """Test removing quotation marks and apostrophes."""
        self.assertEqual(string_reformat("don't"), "dont")
        self.assertEqual(string_reformat("it's"), "its")
        self.assertEqual(string_reformat('"smart"'), "smart")
        self.assertEqual(string_reformat("“smart”"), "smart")
        self.assertEqual(string_reformat("«French»"), "french")
        self.assertEqual(string_reformat("‘single’"), "single")

    def test_string_reformat_empty_string(self):
        """Test empty string and string with only special characters."""
        self.assertEqual(string_reformat(""), "")
        self.assertEqual(string_reformat("   "), "")
        self.assertEqual(string_reformat("!@#$"), "")
        self.assertEqual(string_reformat("'''"), "")

    def test_get_token_set_ratio_exact_match(self):
        """Test identical strings or identical after formatting."""
        self.assertEqual(get_token_set_ratio("hello", "hello"), 100)
        self.assertEqual(get_token_set_ratio("Hello World!", "hello world"), 100)

    def test_get_token_set_ratio_subset_match(self):
        """Test subset strings using token set ratio."""
        self.assertEqual(get_token_set_ratio("fox", "the quick brown fox"), 100)


        # 'honey bee' vs 'gloria gaynor honeybee' does not tokenize exactly due to missing space in honeybee
        score = get_token_set_ratio("Honey Bee", "Gloria Gaynor - Honeybee")
        self.assertTrue(30 <= score <= 50) # Expected around 39

    def test_get_token_set_ratio_no_match(self):
        """Test completely different strings."""
        score = get_token_set_ratio("apple", "banana")
        self.assertTrue(score < 50)

    def test_get_token_set_ratio_case_and_accents(self):
        """Test formatting inside the ratio calculation."""
        self.assertEqual(get_token_set_ratio("Café", "cafe"), 100)
        self.assertEqual(get_token_set_ratio("Über", "uber"), 100)

    def test_get_token_set_ratio_empty_strings(self):
        """Test comparing empty strings or strings that become empty."""
        self.assertEqual(get_token_set_ratio("", ""), 0)
        self.assertEqual(get_token_set_ratio("!@#", "   "), 0)
        # Empty against non-empty
        self.assertEqual(get_token_set_ratio("", "hello"), 0)

if __name__ == '__main__':
    unittest.main()
