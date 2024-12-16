# src/openvino_conversion.py
from openvino.tools.ovc import convert_model
import os

def convert_model_to_openvino():
    # Ensure the models directory exists
    os.makedirs("models/openvino_model", exist_ok=True)

    # Convert the ONNX model to OpenVINO IR format
    onnx_model_path = "models/lstm_model.onnx"
    output_model_path = "models/openvino_model/lstm_model"
    convert_model(onnx_model_path, output_model=output_model_path)