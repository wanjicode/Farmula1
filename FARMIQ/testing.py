import base64
import logging
import os
import sys
import time
from io import BytesIO
from threading import Thread
import atexit

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

# --- 1. APP INITIALIZATION & CONFIGURATION ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Use non-interactive backend for Matplotlib
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('flask_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- 2. GLOBAL VARIABLES & CONSTANTS ---

# Variables for Crop Yield Prediction (from app.py)
model_data = None
available_areas = ['Albania', 'United States', 'India', 'China', 'Brazil', 'France', 'Germany', 'Italy', 'Spain', 'Kenya']
available_crops = ['Maize', 'Potatoes', 'Rice, paddy', 'Soybeans', 'Sorghum', 'Wheat', 'Barley', 'Oats']

# Constants for Image Upload (from image.py)
IMAGE_API_URL = "https://us-central1-striking-shadow-456004-c3.cloudfunctions.net/predict"

# --- 3. HELPER CLASSES & FUNCTIONS ---

# Server health monitor (from image.py)
class ServerMonitor:
    """Simple monitor to track server health and restarts."""
    def __init__(self):
        self.restart_count = 0
        self.max_restarts = 5
        
    def should_restart(self):
        self.restart_count += 1
        return self.restart_count <= self.max_restarts

monitor = ServerMonitor()

# Functions from app.py (Crop Yield Prediction)
def load_model():
    """Load the trained crop yield prediction model."""
    global model_data, available_areas, available_crops
    try:
        model_data = joblib.load('crop_yield_prediction_model.pkl')
        logging.info("✅ Crop yield model loaded successfully!")
        
        # Try to get areas and crops from the loaded model
        try:
            areas = model_data['label_encoders']['Area'].classes_.tolist()
            crops = model_data['label_encoders']['Item'].classes_.tolist()
            if areas: available_areas = areas
            if crops: available_crops = crops
            logging.info(f"📋 Found {len(available_areas)} areas and {len(available_crops)} crops in model")
        except Exception as e:
            logging.warning(f"⚠️ Could not extract areas/crops from model, using defaults: {e}")
            
    except Exception as e:
        logging.error(f"❌ Error loading model file: {e}")
        logging.info("💡 Running in demo mode with static data for crop prediction.")

def create_prediction_charts(base_prediction, area, item, year, rainfall, pesticides, temperature):
    """Create visualization charts for crop yield predictions."""
    charts = {}
    try:
        # Chart 1: Rainfall Impact
        plt.figure(figsize=(10, 6))
        rainfall_range = np.linspace(500, 2500, 20)
        rainfall_predictions = [predict_yield(area, item, year, rain, pesticides, temperature) / 100 for rain in rainfall_range]
        plt.plot(rainfall_range, rainfall_predictions, linewidth=3, color='#4A6B3A', marker='o')
        plt.axvline(x=rainfall, color='#8B5E3C', linestyle='--', linewidth=2, label=f'Current: {rainfall}mm')
        plt.xlabel('Rainfall (mm/year)')
        plt.ylabel('Predicted Yield (kg/ha)')
        plt.title(f'Yield Sensitivity to Rainfall\n{item} in {area}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        charts['rainfall_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Chart 2: Temperature Impact
        plt.figure(figsize=(10, 6))
        temp_range = np.linspace(10, 30, 20)
        temp_predictions = [predict_yield(area, item, year, rainfall, pesticides, temp) / 100 for temp in temp_range]
        plt.plot(temp_range, temp_predictions, linewidth=3, color='#c62828', marker='o')
        plt.axvline(x=temperature, color='#4A6B3A', linestyle='--', linewidth=2, label=f'Current: {temperature}°C')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Predicted Yield (kg/ha)')
        plt.title(f'Yield Sensitivity to Temperature\n{item} in {area}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        charts['temperature_chart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    except Exception as e:
        logging.error(f"Chart generation error: {e}")
    
    return charts

def predict_yield(area, item, year, rainfall, pesticides, temperature):
    """Predict crop yield using the loaded model or a demo fallback."""
    try:
        if model_data is None:
            return demo_prediction(item, rainfall, pesticides, temperature)
            
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        
        rainfall_to_temp_ratio = rainfall / temperature if temperature > 0 else 0
        decade = (year // 10) * 10
        
        input_data = {
            'Area': area, 'Item': item, 'Year': year,
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides, 'avg_temp': temperature,
            'rainfall_to_temp_ratio': rainfall_to_temp_ratio,
            'pesticide_efficiency': 0, 'decade': decade
        }
        
        input_df = pd.DataFrame([input_data])
        input_df['Area'] = label_encoders['Area'].transform([area])[0]
        input_df['Item'] = label_encoders['Item'].transform([item])[0]
        
        expected_features = getattr(model, 'feature_names_in_', input_df.columns)
        input_df = input_df[expected_features]
        prediction = model.predict(input_df)[0]
        
        return prediction
        
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

def demo_prediction(item, rainfall, pesticides, temperature):
    """Fallback prediction when the model is not available."""
    base_yields = {'Maize': 50000, 'Potatoes': 80000, 'Rice, paddy': 40000, 'Soybeans': 30000, 'Wheat': 45000}
    base_yield = base_yields.get(item, 40000)
    prediction = base_yield * (rainfall / 1500) * (1 - abs(temperature - 17) / 10) * min(pesticides / 100, 2)
    return max(prediction, 10000)

# Functions from image.py (Server Management)
def check_dependencies():
    """Check if all required packages are installed."""
    try:
        import flask, flask_cors, requests, pandas, numpy, joblib, matplotlib
        logging.info("✅ All dependencies are available")
        return True
    except ImportError as e:
        logging.error(f"❌ Missing dependency: {e}")
        return False

def setup_environment():
    """Setup environment variables and paths."""
    os.environ['PYTHONUNBUFFERED'] = '1'

# --- 4. FLASK API ROUTES ---

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for crop yield predictions."""
    try:
        data = request.get_json()
        logging.info(f"📥 Received prediction request: {data}")
        
        prediction_hg = predict_yield(
            data['area'], data['item'], int(data['year']),
            float(data['rainfall']), float(data['pesticides']), float(data['temperature'])
        )
        
        if prediction_hg is not None:
            charts = create_prediction_charts(
                prediction_hg, data['area'], data['item'], int(data['year']),
                float(data['rainfall']), float(data['pesticides']), float(data['temperature'])
            )
            response = {
                'success': True,
                'predictions': {
                    'hg_per_ha': round(prediction_hg, 2),
                    'kg_per_ha': round(prediction_hg / 100, 2),
                },
                'inputs': data,
                'charts': charts
            }
            logging.info(f"✅ Prediction successful: {prediction_hg:.0f} hg/ha")
        else:
            response = {'success': False, 'error': 'Prediction failed'}
            
    except Exception as e:
        logging.error(f"❌ /predict error: {e}")
        response = {'success': False, 'error': str(e)}
    
    return jsonify(response)

@app.route('/get_options')
def get_options():
    """API endpoint to get available areas and crops."""
    return jsonify({
        'success': True,
        'areas': available_areas,
        'crops': available_crops
    })

@app.route("/upload", methods=["POST"])
def upload_api():
    """API endpoint for uploading an image and forwarding it."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        logging.info(f"📥 Received file for upload: {file.filename}")
        files = {"file": (file.filename, file.stream, file.mimetype)}
        response = requests.post(IMAGE_API_URL, files=files, timeout=30)
        
        logging.info(f"External API response status: {response.status_code}")
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            logging.error(f"External API error: {response.text}")
            return jsonify({
                "error": "External API request failed", 
                "details": response.text[:500]
            }), response.status_code

    except requests.exceptions.Timeout:
        logging.error("Request to external API timed out")
        return jsonify({"error": "Request timeout - external service is slow"}), 504
    except Exception as e:
        logging.error(f"❌ /upload error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check_api():
    """API endpoint for health checks."""
    return jsonify({
        "status": "healthy", 
        "message": "Server is running",
        "restart_count": monitor.restart_count
    })

# --- 5. SERVER STARTUP AND MANAGEMENT ---

def run_server():
    """Run the Flask server with pre-flight checks."""
    try:
        if not check_dependencies():
            logging.error("Stopping server due to missing dependencies.")
            return False
            
        setup_environment()
        
        # Load the ML model before starting the server
        load_model()
        
        logging.info("🚀 Starting Combined API Server...")
        logging.info(f"🌱 Available crops: {len(available_crops)}")
        logging.info(f"🌍 Available areas: {len(available_areas)}")
        logging.info("🌐 API server running at: http://127.0.0.1:5000")
        
        app.run(debug=False, host="0.0.0.0", port=5000)
        return True
        
    except Exception as e:
        logging.critical(f"💥 Server crashed: {e}")
        return False

def main():
    """Main function with auto-restart capability."""
    logging.info("Application starting...")
    
    while monitor.should_restart():
        success = run_server()
        if success:
            break  # Server stopped normally
            
        logging.warning(f"Server crashed. Restarting... (Attempt {monitor.restart_count}/{monitor.max_restarts})")
        time.sleep(5)
    
    if monitor.restart_count > monitor.max_restarts:
        logging.error("Maximum restart attempts reached. Exiting.")
        sys.exit(1)

@atexit.register
def cleanup():
    logging.info("Server shutting down...")

if __name__ == "__main__":
    main()
