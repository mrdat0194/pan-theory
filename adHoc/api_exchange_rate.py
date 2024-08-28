import datetime
import requests

def get_exchange_rate(base_currency, symbols, date):
    api_key = "jNC2YcPZXWCYJIAqDQJyvxMW23VgzK3x"
    headers = {"apikey": api_key}
    url = f"https://api.apilayer.com/exchangerates_data/{date}?symbols={symbols}&base={base_currency}"
    response = requests.get(url, headers=headers)
    data = response.json()
    print(data['rates']['VND'])
    return data['rates']['VND']


def generate_date_list(start_month=5, start_year=2024):
    """
    Generates a list of dates from the specified month and year to today's date.
    """
    today = datetime.date(2024, 7, 31)
    date_list = []
    current_date = datetime.date(start_year, start_month, 1)  # Start from 1st of the month
    while current_date <= today:
        date_list.append(current_date.strftime("%Y-%m-%d"))  # Format date in YYYY-MM-DD
        current_date += datetime.timedelta(days=1)  # Increment by 1 day
    return date_list

# Get the list of dates from May 2024 to today
dates = generate_date_list()

# Calculate the average exchange rate
total_rate = 0
for date in dates:
    rate = get_exchange_rate('USD', 'VND', date)
    total_rate += rate

average_rate = total_rate / len(dates)
print(average_rate)