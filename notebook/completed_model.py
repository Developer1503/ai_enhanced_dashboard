# model_training_and_conversion.ipynb

# Import necessary libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import os
import tf2onnx
from openvino.tools.ovc import convert_model

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)
os.makedirs("models/openvino_model", exist_ok=True)

# Generate synthetic cash flow data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
cash_flows = np.random.uniform(1000, 5000, size=365)  # Random values between 1000 and 5000

# Use only the Daily Cash Flow column for forecasting
cash_flow = cash_flows.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
cash_flow_scaled = scaler.fit_transform(cash_flow)

# Create sequences for the model
sequence_length = 30  # Use the past 30 days to predict the next day
X, y = [], []
for i in range(sequence_length, len(cash_flow_scaled)):
    X.append(cash_flow_scaled[i-sequence_length:i, 0])
    y.append(cash_flow_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for LSTM input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define and train the LSTM model using the Functional API
inputs = Input(shape=(sequence_length, 1))
x = LSTM(50, return_sequences=True)(inputs)
x = LSTM(50, return_sequences=False)(x)
x = Dense(25)(x)
outputs = Dense(1, name="output")(x)  # Name the output layer

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Save the model
model.save("models/lstm_model.keras")

# Evaluate the model
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and display mean squared error
mse = mean_squared_error(y_test_actual, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, color='blue', label='Actual Cash Flow')
plt.plot(y_pred, color='red', label='Predicted Cash Flow')
plt.title('Cash Flow Prediction')
plt.xlabel('Time')
plt.ylabel('Cash Flow')
plt.legend()
plt.show()

# Convert the model to ONNX format
input_signature = [tf.TensorSpec(shape=[None, 30, 1], dtype=tf.float32)]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature)

# Save the ONNX model
onnx_model_path = "models/lstm_model.onnx"
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Convert the ONNX model to OpenVINO IR format
output_model_path = "models/openvino_model/lstm_model"
convert_model(onnx_model_path, output_model=output_model_path)

# Verify the generated files
print("Generated files:")
print(f"Keras model: models/lstm_model.keras")
print(f"ONNX model: models/lstm_model.onnx")
print(f"OpenVINO IR model: models/openvino_model/lstm_model.xml")
print(f"OpenVINO IR weights: models/openvino_model/lstm_model.bin")
