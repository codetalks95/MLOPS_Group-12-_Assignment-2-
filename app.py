import logging
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.metrics import classification_report, confusion_matrix

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model
best_model = None
try:
    with open('notebooks/pkl/model.pkl', 'rb') as file:
        best_model = pickle.load(file)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", str(e))

# Load the saved scaler
scaler = None
try:
    with open('notebooks/pkl/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error("Error loading scaler: %s", str(e))

# Check if the model and scaler are loaded correctly
if best_model is None:
    logging.error("Model is None after loading.")
if scaler is None:
    logging.error("Scaler is None after loading.")


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(-1, 4)
    scaled_features = scaler.transform(features)

    logging.info("Input features: %s", features)
    logging.info("Scaled features: %s", scaled_features)

    predictions = best_model.predict(scaled_features)
    logging.info("Raw predictions: %s", predictions)

    # Check if all predictions are zero
    if np.all(predictions == 0):
        logging.warning("All predictions are zero for input: %s", features)
        return jsonify({
            'message': 'All predictions are 0. Please consider adjusting your input values for better results.'
        }), 200

    return jsonify({
        'predictions': predictions.tolist(),
        'message': 'Predictions generated successfully.'
    })


@app.route('/evaluate', methods=['POST'])
def evaluate():
    data = request.get_json()
    logging.info("Received evaluation data: %s", data)

    try:
        features = np.array(data['features'])
        true_labels = np.array(data['true_labels'])

        scaled_features = scaler.transform(features)
        predictions = best_model.predict(scaled_features)

        # Evaluate the model
        cm = confusion_matrix(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)

        # Format the response
        response = {
            'classification_report': {
                str(label): {
                    "precision": float(report[str(label)]['precision']),
                    "recall": float(report[str(label)]['recall']),
                    "f1-score": float(report[str(label)]['f1-score']),
                    "support": int(report[str(label)]['support'])
                } for label in report if label.isdigit()
            },
            'accuracy': float(report['accuracy']),
            'macro avg': {
                "precision": float(report['macro avg']['precision']),
                "recall": float(report['macro avg']['recall']),
                "f1-score": float(report['macro avg']['f1-score']),
                "support": int(report['macro avg']['support'])
            },
            'weighted avg': {
                "precision": float(report['weighted avg']['precision']),
                "recall": float(report['weighted avg']['recall']),
                "f1-score": float(report['weighted avg']['f1-score']),
                "support": int(report['weighted avg']['support'])
            },
            'confusion_matrix': cm.tolist()
        }

        return jsonify(response)

    except Exception as e:
        logging.error("Error during evaluation: %s", str(e))
        return jsonify({'error': 'Invalid input data'}), 400


if __name__ == "__main__":
    app.run(debug=True, port=9001)
