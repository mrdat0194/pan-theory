import requests
import datetime

api_key = "xx6nnEoT0ziwKoydsNbLVrYISR1WpfmG"

# Extract base_currency and symbols from request parameters (if provided)
# base_currency = requests.args.get("base_currency", "USD")
# symbols = requests.args.get("symbols", "VND")

base_currency = 'USD'
symbols = 'VND'

# Construct the API URL
url = f"https://api.apilayer.com/exchangerates_data/latest?symbols={symbols}&base={base_currency}"

# Make the HTTP GET request with headers
headers = {"apikey": api_key}
response = requests.get(url, headers=headers)

# Check for successful response
if response.status_code == 200:
  data = response.json()
  print(data)

  # Extract and format exchange rate for BigQuery
  exchange_rate = data.get("rates", {}).get(symbols, None)
  if exchange_rate is None:
      print( f"Error: Currency '{symbols}' not found in response.")