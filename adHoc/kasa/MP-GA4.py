from ga4mp import Ga4mp
import json

with open("credentials.json", 'r') as json_file:
    client_config = json.load(json_file)
    measurement_id = client_config['MEASUREMENT_ID']
    api_secret = client_config['API_SECRET']
    client_id = client_config['CLIENT_ID']

# Create an instance of GA4 object
ga = Ga4mp(measurement_id , api_secret , client_id)

# Specify event type and parameters
event_type = 'refund'

event_parameters = { "transaction_id": "t20002" , "value": 20000, "currency": "VND"}

event = {'name': event_type, 'params': event_parameters }
events = [event]
ga.send(events)
