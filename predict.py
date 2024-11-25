import pickle
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model file
model_file = 'laptop_price_rf.bin'

# Load the model and preprocessor
try:
    with open(model_file, 'rb') as f_in:
        preprocessor, model = pickle.load(f_in)
    logging.info('Preprocessor and model loaded successfully.')
except FileNotFoundError:
    logging.error(f'Model file {model_file} not found. Please ensure the file is available.')
    raise
except Exception as e:
    logging.error(f'Error loading model: {e}')
    raise

# Initialize Flask app
app = Flask('laptop_price_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        laptop_data = request.get_json()

        # Required fields for input validation
        required_fields = [
            "brand", "color", "condition", "gpu", "processor",
            "processor_speed", "processor_speed_unit", "type",
            "display_width", "display_height", "os", "storage_type",
            "hard_drive_capacity", "hard_drive_capacity_unit",
            "ssd_capacity", "ssd_capacity_unit", "screen_size_inch",
            "ram_size", "ram_size_unit"
        ]

        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in laptop_data]
        if missing_fields:
            return jsonify({"error": f"Missing fields in input: {', '.join(missing_fields)}"}), 400

        # Preprocess input data
        X = preprocessor.transform([laptop_data])

        # Make prediction
        price_pred = model.predict(X)[0]

        # Return prediction result
        result = {
            'predicted_price': float(price_pred)
        }
        return jsonify(result)
    except Exception as e:
        logging.error(f'Error during prediction: {e}')
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
