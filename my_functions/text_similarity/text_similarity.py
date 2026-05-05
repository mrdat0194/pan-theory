import unidecode
import re
from fuzzywuzzy import fuzz

'''
Sources: https://www.datacamp.com/community/tutorials/fuzzy-string-python
https://github.com/seatgeek/fuzzywuzzy
'''

def string_reformat(string: str) -> str:
    """
    Reformats a string by removing accents, special characters, and normalizing whitespace.
    
    Args:
        string (str): Input string to be reformatted
        
    Returns:
        str: Reformatted string with accents removed, special characters replaced with spaces,
             and all text converted to lowercase
    """
    # Compile regex patterns once for better performance
    pat_redundant_chars = re.compile(r"[\s`~!@#$%^&*()\-_+={\}\[\]\\|:;<>,./?]+")
    pat_quotations = re.compile(r'[\"«»''‚‛""„‟‹›❛❜❝❞❮❯〝〞〟＂⹂“”‘’]+')

    # Process string in steps
    str_remove_accent = unidecode.unidecode(string).lower()
    str_remove_quatation = str_remove_accent.replace("'", "")
    str_quotation = pat_quotations.sub("", str_remove_quatation)
    str_reformat_result = pat_redundant_chars.sub(" ", str_quotation).strip()

    return str_reformat_result


def get_token_set_ratio(str1: str, str2: str) -> int:
    """
    Calculates the token set ratio between two strings using fuzzy string matching.
    
    Args:
        str1 (str): First string to compare
        str2 (str): Second string to compare
        
    Returns:
        int: Token set ratio between 0 and 100, where 100 indicates perfect match
    """
    string_reformat1 = string_reformat(str1)
    string_reformat2 = string_reformat(str2)
    return fuzz.token_set_ratio(string_reformat1, string_reformat2)


if __name__ == "__main__":
    str1 = 'Honey Bee'
    str2 = "Gloria Gaynor - Honeybee"
    # joy = get_token_set_ratio(str2, str1)
    print(string_reformat(str1))
    print(string_reformat(str2))
    print(get_token_set_ratio(str2, str1))
    print(get_token_set_ratio(str1, str2))
