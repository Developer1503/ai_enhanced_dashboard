# src/__init__.py

# Import the modules from the src directory
from .model_training import train_model
from .model_conversion import convert_model_to_onnx
from .openvino_conversion import convert_model_to_openvino
from .flask_api import create_app

# You can also define package-level variables or functions if needed
__all__ = ['train_model', 'convert_model_to_onnx', 'convert_model_to_openvino', 'create_app']
