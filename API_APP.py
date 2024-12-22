import json
import os
import random
from datetime import datetime, timedelta

import requests
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Register custom loss function if 'mse' is a custom function
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# Load the model with custom objects
try:
    model = tf.keras.models.load_model('energy_disaggregation_model.h5', custom_objects={'mse': mse})
    print("Model loaded successfully")
except Exception as e:
    print("Failed to load the model. Error:", e)

# Load the scaler
try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully")
except Exception as e:
    print("Failed to load the scaler. Error:", e)

# Define the window size
window_size = 60  # Modify this according to your model's requirements

app = Flask(__name__)

# Define the list of appliances
appliances = [
    'laptop computer', 'television', 'light', 'HTPC', 'food processor', 'toasted sandwich maker',
    'toaster', 'microwave', 'computer monitor', 'audio system', 'audio amplifier', 'broadband router',
    'ethernet switch', 'USB hub', 'tablet computer charger', 'radio', 'wireless phone charger', 'mobile phone charger',
    'coffee maker', 'computer', 'external hard disk', 'desktop computer', 'printer', 'immersion heater',
    'security alarm', 'projector', 'server computer', 'running machine', 'network attached storage', 'fridge',
    'air conditioner'
]

# IPFS URL (Adjust according to your IPFS node configuration)
IPFS_API_URL = 'http://127.0.0.1:5001/api/v0/add'


def predict_appliance_consumption(aggregate_window, scaler, window_size):
    # Standardize the input features
    aggregate_window = np.array(aggregate_window).reshape(-1, 1)
    aggregate_window = scaler.transform(aggregate_window)
    aggregate_window = aggregate_window.reshape((1, window_size, 1))

    # Predict using the trained model
    predictions = model.predict(aggregate_window)
    # Clip negative values to zero
    predictions = np.clip(predictions, 0, None)
    return predictions


def upload_to_ipfs(filepath):
    with open(filepath, 'rb') as file:
        response = requests.post(IPFS_API_URL, files={'file': file})
        ipfs_hash = response.json()['Hash']
        return ipfs_hash


@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for making predictions."""
    data = request.json
    if not data or 'input' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    try:
        # Ensure the input is a list of 60 aggregate power readings
        input_data = data['input']
        if len(input_data) != window_size:
            return jsonify({"error": f"Input data must be of length {window_size}"}), 400

        # Get predictions
        predictions = predict_appliance_consumption(input_data, scaler, window_size)

        # Convert predictions to standard Python float
        predictions = predictions[0].astype(float).tolist()

        # Map appliance names to predictions
        predicted_consumption = dict(zip(appliances, predictions))

        # Load existing JSON data
        with open('appliance_consumption.json', 'r') as jsonfile:
            data = json.load(jsonfile)

        # Get the latest entry (most recent day)
        latest_entry = data[-1]
        current_date = latest_entry["date"]

        # Update the latest entry with new predictions for the next hour
        for appliance, prediction in predicted_consumption.items():
            latest_entry['consumptionPerHour'][appliance].append(round(prediction, 3))
            latest_entry['consumptionPerHour'][appliance] = latest_entry['consumptionPerHour'][appliance][-24:]

        # Calculate the new total consumption for the day
        latest_entry['total'] = round(sum(sum(latest_entry['consumptionPerHour'][appliance]) for appliance in appliances), 3)

        # Save updated data back to JSON file
        with open('appliance_consumption.json', 'w') as jsonfile:
            json.dump(data, jsonfile, indent=4)

        # Upload updated JSON file to IPFS
        ipfs_hash = upload_to_ipfs('appliance_consumption.json')

        return jsonify({"predictions": predicted_consumption, "ipfs_hash": ipfs_hash})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
