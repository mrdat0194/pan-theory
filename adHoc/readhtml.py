import urllib3
from bs4 import BeautifulSoup

# Define the URL and request parameters
url = 'https://marketingplatform.google.com/home/orgs/pglM-mkLRGqEmfMghwdjHQ/usage?period=2024-04&orderType=13&authuser=1'

# Create a urllib3 PoolManager instance
http = urllib3.PoolManager()

try:
  # Send a GET request with urllib3
  response = http.request('GET', url)
  print(response.status)
  # Check for successful response (status code 200)
  if response.status == 200:
    # Read the response content
    html_doc = response.read().decode('utf-8')

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_doc, 'html.parser')

    # Format and prettify the parsed HTML
    strhtm = soup.prettify()

    # Print the first few characters
    print(strhtm[:100])  # Limit output to first 100 characters for brevity
  else:
    print(f"Error: HTTP response status {response.status}")
except urllib3.exceptions.MaxRetryError as e:
  print(f"Connection error: {e}")
except urllib3.exceptions.RequestError as e:
  print(f"Request error: {e}")