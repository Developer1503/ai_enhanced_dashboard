# src/model_conversion.py
import tensorflow as tf
import tf2onnx
import os

def convert_model_to_onnx():
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load the trained model
    model = tf.keras.models.load_model("models/lstm_model.keras")

    # Convert the model to ONNX format
    input_signature = [tf.TensorSpec(shape=[None, 30, 1], dtype=tf.float32)]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

    # Save the ONNX model
    onnx_model_path = "models/lstm_model.onnx"
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
