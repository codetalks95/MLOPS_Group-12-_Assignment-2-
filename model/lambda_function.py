import json
import pickle

import pandas as pd

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


def lambda_handler(event, context):
    features = event['features']
    input_data = pd.DataFrame([features])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    return {
        'statusCode': 200,
        'body': json.dumps({'prediction': int(prediction[0])})
    }
