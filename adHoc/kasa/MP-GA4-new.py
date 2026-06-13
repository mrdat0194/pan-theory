from ga4mp import GtagMP
from ga4mp.store import DictStore
import json
# Create a DictStore
# my_store = DictStore(data=<DICTIONARY>)

# Create an instance of GA4 object using gtag, including the new DictStore.
# # Example 1: purchase
# with open("credentials/credentials.json", 'r') as json_file:
#     client_config = json.load(json_file)
#     measurement_id = client_config['MEASUREMENT_ID']
#     api_secret = client_config['API_SECRET']
#     client_id = client_config['CLIENT_ID']
#     gtag_tracker = GtagMP(api_secret=api_secret, measurement_id=measurement_id, client_id=client_id)
#
#     # Create a new event for purchase.
#     purchase_event = gtag_tracker.create_new_event(name="purchase")
#
#     # Set transaction_id, value, and currency.
#     purchase_event.set_event_param(name="transaction_id", value="t2003")
#     purchase_event.set_event_param(name="currency", value="VND")
#     purchase_event.set_event_param(name="value", value=30000)
#
#     # # Create an item and add details about the item.
#     # shirt = purchase_event.create_new_item(item_id="SKU_12345", item_name="Stan and Friends Tee")
#     # shirt.set_parameter("price", 9.99)
#     # shirt.set_parameter("quantity", 1)
#     # shirt.set_parameter("item_category", "Apparel")
#     #
#     # # Add the item to the purchase event.
#     # purchase_event.add_item_to_event(shirt)
#     #
#     # # Add a user property based on what you know about the user.
#     # gtag_tracker.store.set_user_property(name="shirt_wearer", value="yes")
#
#     # Send the event to GA4 immediately.
#     event_list = [purchase_event]
#     gtag_tracker.send(events=event_list)

# Example 2: refund
with open("credentials/credentials.json", 'r') as json_file:
    client_config = json.load(json_file)
    measurement_id = client_config['MEASUREMENT_ID']
    api_secret = client_config['API_SECRET']
    client_id = client_config['CLIENT_ID']
    gtag_tracker = GtagMP(api_secret=api_secret, measurement_id=measurement_id, client_id=client_id)

    # Create a new event for purchase.
    purchase_event = gtag_tracker.create_new_event(name="refund")

    # Set transaction_id, value, and currency.
    purchase_event.set_event_param(name="transaction_id", value="t2003")
    purchase_event.set_event_param(name="currency", value="VND")
    purchase_event.set_event_param(name="value", value=30000)

    # Send the event to GA4 immediately.
    event_list = [purchase_event]
    gtag_tracker.send(events=event_list)