from flask import Flask, request, jsonify
import numpy as np
from openvino.runtime import Core

# Initialize Flask app
app = Flask(__name__)

# Load the OpenVINO model
MODEL_PATH = "random_forest_model.xml"  # Update with the correct IR model path
core = Core()
compiled_model = core.compile_model(MODEL_PATH, "CPU")
infer_request = compiled_model.create_infer_request()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the POST request
        input_data = request.json.get('input')
        if not input_data or not isinstance(input_data, list):
            return jsonify({"error": "Invalid input. Expected a list of numeric features."}), 400

        # Convert input data to a NumPy array
        input_array = np.array(input_data).reshape(1, -1)

        # Run inference
        predictions = infer_request.infer({0: input_array})
        result = predictions[compiled_model.outputs[0]].tolist()

        # Return the result as JSON
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
