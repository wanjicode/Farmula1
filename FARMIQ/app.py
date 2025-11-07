from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')  # Required for server environments
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app

# --- [The rest of your Python code remains exactly the same] ---
# Global variables
model_data = None
available_areas = ['Albania', 'United States', 'India', 'China', 'Brazil', 'France', 'Germany', 'Italy', 'Spain', 'Kenya']
available_crops = ['Maize', 'Potatoes', 'Rice, paddy', 'Soybeans', 'Sorghum', 'Wheat', 'Barley', 'Oats']

def load_model():
    """Load the trained model"""
    global model_data
    try:
        model_data = joblib.load('crop_yield_prediction_model.pkl')
        print("✅ Model loaded successfully!")
        
        # Try to get areas and crops from model
        try:
            areas = model_data['label_encoders']['Area'].classes_.tolist()
            crops = model_data['label_encoders']['Item'].classes_.tolist()
            print(f"📋 Found {len(areas)} areas and {len(crops)} crops in model")
            return areas, crops
        except Exception as e:
            print(f"⚠️ Could not extract areas/crops from model: {e}")
            return available_areas, available_crops
            
    except Exception as e:
        print(f"❌ Error loading model file: {e}")
        print("💡 Using demo mode with static data")
        return available_areas, available_crops

# Load model on startup
available_areas, available_crops = load_model()

def create_prediction_charts(base_prediction, area, item, year, rainfall, pesticides, temperature):
    """Create visualization charts for predictions"""
    charts = {}
    
    try:
        # Chart 1: Sensitivity Analysis - Rainfall Impact
        plt.figure(figsize=(10, 6))
        rainfall_range = np.linspace(500, 2500, 20)
        rainfall_predictions = []
        
        for rain in rainfall_range:
            pred = predict_yield(area, item, year, rain, pesticides, temperature)
            rainfall_predictions.append(pred / 100)  # Convert to kg/ha
        
        plt.plot(rainfall_range, rainfall_predictions, linewidth=3, color='#4A6B3A', marker='o')
        plt.axvline(x=rainfall, color='#8B5E3C', linestyle='--', linewidth=2, label=f'Current: {rainfall}mm')
        plt.xlabel('Rainfall (mm/year)')
        plt.ylabel('Predicted Yield (kg/ha)')
        plt.title(f'Yield Sensitivity to Rainfall\n{item} in {area}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        buf1 = BytesIO()
        plt.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
        buf1.seek(0)
        charts['rainfall_chart'] = base64.b64encode(buf1.getvalue()).decode('utf-8')
        plt.close()
        
        # Chart 2: Sensitivity Analysis - Temperature Impact
        plt.figure(figsize=(10, 6))
        temp_range = np.linspace(10, 30, 20)
        temp_predictions = []
        
        for temp in temp_range:
            pred = predict_yield(area, item, year, rainfall, pesticides, temp)
            temp_predictions.append(pred / 100)  # Convert to kg/ha
        
        plt.plot(temp_range, temp_predictions, linewidth=3, color='#c62828', marker='o')
        plt.axvline(x=temperature, color='#4A6B3A', linestyle='--', linewidth=2, label=f'Current: {temperature}°C')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Predicted Yield (kg/ha)')
        plt.title(f'Yield Sensitivity to Temperature\n{item} in {area}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
        buf2.seek(0)
        charts['temperature_chart'] = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close()
        
        # Chart 3: Crop Comparison
        plt.figure(figsize=(10, 6))
        top_crops = available_crops[:6]
        crop_predictions = []
        
        for crop in top_crops:
            pred = predict_yield(area, crop, year, rainfall, pesticides, temperature)
            crop_predictions.append(pred / 100)
        
        colors = ['#4A6B3A', '#8B5E3C', '#A27B5C', '#3E5C32', '#C8A384', '#6B8E23']
        bars = plt.bar(top_crops, crop_predictions, color=colors, alpha=0.8)
        
        for bar, value in zip(bars, crop_predictions):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                     f'{value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Crop Type')
        plt.ylabel('Predicted Yield (kg/ha)')
        plt.title(f'Yield Comparison for Different Crops in {area}')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        buf3 = BytesIO()
        plt.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
        buf3.seek(0)
        charts['crop_comparison_chart'] = base64.b64encode(buf3.getvalue()).decode('utf-8')
        plt.close()

    except Exception as e:
        print(f"Chart generation error: {e}")
    
    return charts


def predict_yield(area, item, year, rainfall, pesticides, temperature):
    """Predict crop yield with all required features"""
    try:
        if model_data is None:
            # Demo mode - return realistic fake predictions
            return demo_prediction(item, rainfall, pesticides, temperature)
            
        # Real model prediction with ALL features
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        
        # Calculate engineered features that were used during training
        rainfall_to_temp_ratio = rainfall / temperature if temperature > 0 else 0
        decade = (year // 10) * 10
        
        # Create input data with ALL features
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
        
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
        else:
            expected_features = input_df.columns
        
        input_df = input_df[expected_features]
        prediction = model.predict(input_df)[0]
        input_df['pesticide_efficiency'] = prediction / (pesticides + 1)
        
        return prediction
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def demo_prediction(item, rainfall, pesticides, temperature):
    """Fallback prediction when model is not available"""
    base_yields = {
        'Maize': 50000, 'Potatoes': 80000, 'Rice, paddy': 40000,
        'Soybeans': 30000, 'Sorghum': 35000, 'Wheat': 45000,
        'Barley': 40000, 'Oats': 35000
    }
    base_yield = base_yields.get(item, 40000)
    
    rain_factor = rainfall / 1500
    temp_factor = 1 - abs(temperature - 17) / 10
    pest_factor = min(pesticides / 100, 2)
    
    prediction = base_yield * rain_factor * temp_factor * pest_factor
    return max(prediction, 10000)

@app.route('/predict', methods=['POST'])
def predict_api():
    """Handle predictions with chart generation"""
    try:
        data = request.get_json()
        print(f"📥 Received: {data}")
        
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
                    'tonnes_per_ha': round(prediction_hg / 10000, 3)
                },
                'inputs': data,
                'charts': charts
            }
            print(f"✅ Prediction successful: {prediction_hg:.0f} hg/ha")
        else:
            response = {'success': False, 'error': 'Prediction failed'}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        response = {'success': False, 'error': str(e)}
    
    return jsonify(response)

@app.route('/get_options')
def get_options():
    """API endpoint for available options"""
    return jsonify({
        'success': True,
        'areas': available_areas,
        'crops': available_crops
    })

if __name__ == '__main__':
    print("🚀 Starting Crop Yield Predictor API Server...")
    print(f"🌱 Available crops: {len(available_crops)}")
    print("🌐 API server running at: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)