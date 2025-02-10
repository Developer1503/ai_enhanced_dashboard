# 🚀 AI-Enhanced Cash Flow Dashboard

## 📌 Overview
This project is an **AI-powered dashboard** for **cash flow analysis and prediction**. It leverages an **LSTM model** for time series forecasting and provides an **interactive dashboard** for data visualization. The application is built using **Flask, Dash, and OpenVINO** for optimized model inference.

## 📁 Directory Structure
```
├── dataset/
│   ├── 📄 preprocessed_cash_flow.csv   # Preprocessed dataset
├── model/
│   ├── 🧠 lstm_model.keras             # Trained LSTM model (Keras format)
│   ├── 🔄 lstm_model.onnx              # Converted ONNX model
│   ├── openvino_model/
│       ├── ⚡ lstm_model.xml           # OpenVINO IR format model
├── notebook/
│   ├── 📜 completed_model.py           # Model training & conversion script
├── server/
│   ├── 🔌 __init__.py                  # Initializes server package
│   ├── 📊 dashboard.py                  # Dash application for visualization
│   ├── 🌍 flask_api.py                  # Flask API for model predictions
│   ├── 🔄 model_conversion.py           # Converts Keras model to ONNX
│   ├── 🎯 model_training.py             # Trains the LSTM model
│   ├── ⚙️ openvino_conversion.py        # Converts ONNX model to OpenVINO
│   ├── ⚙️ .idea/                         # PyCharm project configurations
```

## 🎯 Features
✅ **AI-Powered Forecasting**: LSTM model trained on cash flow data.  
✅ **Optimized Inference**: Model converted to ONNX and OpenVINO for efficient predictions.  
✅ **Interactive Dashboard**: Built with Dash, offering filters and various visualizations.  
✅ **Flask API**: Serves predictions using OpenVINO-optimized model.  
✅ **Data Export**: Users can export filtered data to CSV or Excel.  

## ⚡ Installation & Setup
### 📌 Prerequisites
Ensure you have the following installed:
- 🐍 Python 3.8+
- 📦 pip
- 🔗 Virtual environment (recommended)
- 📜 Required dependencies from `requirements.txt`

### 🛠️ Installation Steps
1. **Clone the repository**:
   ```sh
   git clone https://github.com/your-repo/ai-cashflow-dashboard.git
   cd ai-cashflow-dashboard
   ```
2. **Create and activate a virtual environment**:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the Flask API**:
   ```sh
   cd server
   python flask_api.py
   ```
5. **Run the Dash dashboard**:
   ```sh
   python dashboard.py
   ```
6. **Access the dashboard** at `http://127.0.0.1:8050/`

## 🧠 Model Training & Conversion
To **retrain the LSTM model** and **convert it to ONNX/OpenVINO**, execute:
```sh
cd notebook
python completed_model.py
```

## 🔗 API Endpoints
- **`POST /predict`**: Accepts JSON input and returns model predictions.
- **`GET /health`**: Returns API status.

## 📊 Dash Features
- 🎛️ **Filters**: Categories, regions, transaction types, date ranges  
- 📈 **Visualizations**: Line charts, bar charts, histograms, heatmaps, pie charts  
- 📂 **Data export**: CSV, Excel  

## 🚀 Deployment
For **production deployment**, use Gunicorn and a reverse proxy (Nginx/Apache):
```sh
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Contributions are welcome! Please create a **pull request** for major changes.

## 📬 Contact
For issues or inquiries, reach out to [vedantshinde305@gmail.com](mailto:vedantshinde@gmail.com).
