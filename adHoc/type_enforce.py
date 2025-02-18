def user_facing_api(text: str, amount: float) -> bool:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(amount, float):
        raise TypeError("amount must be a float")
    return True
    # ... rest of your function logic

# Example usage:
text = "Hello"
# text = 5
amount = 10.5

result = user_facing_api(text, amount)
print(result)