# src/flask_api.py
from flask import Flask, request, jsonify
import numpy as np
from openvino.runtime import Core
import os

app = Flask(__name__)

# Load the OpenVINO IR model
ie_core = Core()
model_ir = ie_core.read_model(model="models/openvino_model/lstm_model.xml")
compiled_model_ir = ie_core.compile_model(model=model_ir, device_name="CPU")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data['input']).reshape(1, -1, 1).astype(np.float32)

    # Create an inference request
    infer_request = compiled_model_ir.create_infer_request()

    # Perform inference
    results = infer_request.infer({"input": input_data})

    # Get the output from the inference request
    predicted_cash_flow = results["output"]

    return jsonify({'prediction': predicted_cash_flow.tolist()})

def create_app():
    return app

if __name__ == '__main__':
    app.run(debug=True)
