from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging
import sys
import os
import time
from threading import Thread
import atexit

app = Flask(__name__)
CORS(app)

# Configure logging to file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('flask_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

API_URL = "https://us-central1-striking-shadow-456004-c3.cloudfunctions.net/predict"

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import flask
        import flask_cors
        import requests
        logging.info("All dependencies are available")
        return True
    except ImportError as e:
        logging.error(f"Missing dependency: {e}")
        return False

def setup_environment():
    """Setup environment variables and paths"""
    # Add any environment setup here
    os.environ['PYTHONUNBUFFERED'] = '1'

class ServerMonitor:
    """Simple monitor to track server health"""
    def __init__(self):
        self.restart_count = 0
        self.max_restarts = 5
        
    def should_restart(self):
        self.restart_count += 1
        return self.restart_count <= self.max_restarts

monitor = ServerMonitor()

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        app.logger.info(f"Received file: {file.filename}")
        files = {"file": (file.filename, file.stream, file.mimetype)}
        response = requests.post(API_URL, files=files, timeout=30)
        
        app.logger.info(f"External API response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                app.logger.info(f"API response: {result}")
                return jsonify(result)
            except ValueError:
                return jsonify({"prediction": response.text})
        else:
            app.logger.error(f"API error: {response.text}")
            return jsonify({
                "error": "External API request failed", 
                "status_code": response.status_code,
                "details": response.text[:500]
            }), 500

    except requests.exceptions.Timeout:
        app.logger.error("Request timeout")
        return jsonify({"error": "Request timeout - external service took too long"}), 504
    except requests.exceptions.ConnectionError:
        app.logger.error("Connection error")
        return jsonify({"error": "Cannot connect to external service"}), 503
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "message": "Server is running",
        "restart_count": monitor.restart_count
    })

def run_server():
    """Run the Flask server with error handling"""
    try:
        if not check_dependencies():
            logging.error("Missing dependencies. Please install required packages.")
            return False
            
        setup_environment()
        
        logging.info("Starting Flask server...")
        # Run in production mode (debug=False)
        app.run(debug=False, host="0.0.0.0", port=5000)
        return True
        
    except Exception as e:
        logging.error(f"Server crashed: {e}")
        return False

def main():
    """Main function with auto-restart capability"""
    logging.info("Application starting...")
    
    while monitor.should_restart():
        success = run_server()
        if success:
            break  # Server stopped normally
            
        logging.warning(f"Server crashed. Restarting... ({monitor.restart_count}/{monitor.max_restarts})")
        time.sleep(5)  # Wait before restarting
    
    if monitor.restart_count > monitor.max_restarts:
        logging.error("Maximum restart attempts reached. Exiting.")
        sys.exit(1)

# Cleanup function
def cleanup():
    logging.info("Server shutting down...")

atexit.register(cleanup)

if __name__ == "__main__":
    main()