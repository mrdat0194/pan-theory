import re

def get_ads_from_url(url):
  """
  Extracts "ads" from the beginning of a URL path segment using a regular expression.

  Args:
      url: The string to search.

  Returns:
      The captured "ads" string or None if not found at the beginning.
  """
  # regex = r"/\/(.*?)(?=\/|$)/"
  regex = r"/(.*?)(?=/)"
  match = re.search(regex, url)
  print(match)
  if match:
    return match.group(0)[1:]  # Extract "ads" without leading slash
  else:
    return None

# Example usage
url = "/ads/c/Véhicules/"

ads_part = get_ads_from_url(url)
if ads_part:
  print(f"Extracted 'ads': {ads_part}")
else:
  print("'ads' not found at the beginning of the string")