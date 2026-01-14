import os
import socket
import time
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
import threading
import json
import datetime
from werkzeug.utils import secure_filename
import cv2
import uuid
import traceback

# Create a function to ensure localhost is available before starting Flask
def ensure_loopback_available():
    """Make sure localhost/loopback interface is available"""
    attempts = 0
    while attempts < 30:  # Try for 30 seconds
        try:
            # Try to bind to localhost to check if it's available
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('127.0.0.1', 0))  # Bind to a random port
            s.close()
            print("Loopback interface is available.")
            return True
        except socket.error:
            print(f"Waiting for loopback interface to be ready... (attempt {attempts+1}/30)")
            attempts += 1
            time.sleep(1)
    print("WARNING: Could not confirm loopback interface availability!")
    return False

# Initialize app with explicit loopback listening
app = Flask(__name__)

# Folders setup
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CAPTURE_FOLDER = os.path.join(app.root_path, 'static', 'captured')
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

PROCESSED_FOLDER = os.path.join(app.root_path, 'static', 'processed')
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Local storage for results
LOCAL_STORAGE_DIR = os.path.join(app.root_path, 'local_storage')
RICE_STORAGE = os.path.join(LOCAL_STORAGE_DIR, 'rice')
DAL_STORAGE = os.path.join(LOCAL_STORAGE_DIR, 'dal')

os.makedirs(RICE_STORAGE, exist_ok=True)
os.makedirs(DAL_STORAGE, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CAPTURE_FOLDER'] = CAPTURE_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Constants
MAX_IMAGES = 50
MAC_ADDRESS = "d8:3a:dd:c0:77:fd"

# Initialize camera only when needed
camera = None

def initialize_camera():
    global camera
    if camera is None:
        try:
            # Import numpy first and force it to be loaded before other modules
            import numpy
            import warnings
            
            # Temporarily suppress the numpy.dtype warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="numpy.dtype size changed")
                
                from camera import Camera
                camera = Camera()
                print("Camera initialized successfully")
        except Exception as e:
            print(f"Warning: Camera initialization failed: {str(e)}")
            camera = None
    return camera

def cleanup_old_images(directory, max_files=10):
    """Deletes the oldest image if the directory contains more than `max_files` images."""
    files = sorted(
        (os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg')),
        key=os.path.getctime  # Sort by creation time (oldest first)
    )
    while len(files) > max_files:
        os.remove(files.pop(0))  # Remove oldest file

def manage_captured_images():
    """Ensures that only the last 10 captured images are stored."""
    images = sorted(os.listdir(CAPTURE_FOLDER), key=lambda x: os.path.getctime(os.path.join(CAPTURE_FOLDER, x)))
    while len(images) > MAX_IMAGES:
        os.remove(os.path.join(CAPTURE_FOLDER, images.pop(0)))

def gen(camera):
    """Generates frames for live video feed."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page with live feed"""
    return render_template('index.html')

@app.route('/wifi')
def wifi_page():
    """WiFi setup page"""
    return render_template('wifi.html')

@app.route('/video_feed')
def video_feed():
    """Live video feed endpoint"""
    camera = initialize_camera()
    if camera:
        return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Camera not available", 500

@app.route('/capture', methods=['POST'])
def capture():
    """Captures an image from the camera and saves it"""
    try:
        camera = initialize_camera()
        if camera:
            timestamp = int(time.time())
            filename = f"captured_{timestamp}.jpg"
            filepath = os.path.join(CAPTURE_FOLDER, filename)
            frame = camera.capture()
            if frame is not None:
                cv2.imwrite(filepath, frame)
                manage_captured_images()
                return jsonify({
                    "status": "success",
                    "image_url": url_for('static', filename=f'captured/{filename}'),
                    "timestamp": timestamp
                })
            else:
                return jsonify({"error": "Failed to capture frame"}), 500
        else:
            return jsonify({"error": "Camera not available"}), 500
    except Exception as e:
        print(f"Error capturing image: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/process_image', methods=['POST'])
def process_image_route():
    """
    Process a rice image: detect grains, classify using ML model, and return results.
    """
    try:
        data = request.get_json()
        image_path = data.get("image_path")

        if not image_path:
            return jsonify({"error": "Invalid request: missing image_path"}), 400

        # Build full path to the image
        image_full_path = os.path.join(app.root_path, image_path.lstrip('/'))
        
        if not os.path.exists(image_full_path):
            return jsonify({"error": "Image not found at path: " + image_full_path}), 404
        
        image = cv2.imread(image_full_path)
        if image is None:
            return jsonify({"error": "Failed to read image file"}), 400

        # Import and run the RPi-optimized processing
        try:
            from process_image_rpi_final import process_image, load_model_once
            
            # Ensure model is loaded
            if not load_model_once():
                return jsonify({"error": "Failed to load ML model"}), 500
            
            # Process the image
            processed_result = process_image(image)
            
            # Unpack results (10 values)
            final_image = processed_result[0]
            perfect_count = processed_result[1]
            chalky_count = processed_result[2]
            black_count = processed_result[3]
            yellow_count = processed_result[4]
            brown_count = processed_result[5]
            broken_percentages = processed_result[6]
            broken_grain_count = processed_result[7]
            stone_count = processed_result[8]
            husk_count = processed_result[9]

            # Calculate total
            total_objects = (perfect_count + chalky_count + black_count + yellow_count + 
                           brown_count + broken_grain_count + stone_count + husk_count)

            # Save processed image
            processed_filename = f"processed_{int(time.time())}.jpg"
            processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
            cv2.imwrite(processed_filepath, final_image)

            # Cleanup old processed images
            cleanup_old_images(PROCESSED_FOLDER, max_files=MAX_IMAGES)

            return jsonify({
                "status": "success",
                "processed_image_url": url_for('static', filename=f'processed/{processed_filename}'),
                "total_objects": total_objects,
                "perfect_count": perfect_count,
                "chalky_count": chalky_count,
                "black_count": black_count,
                "yellow_count": yellow_count,
                "brown_count": brown_count,
                "broken_percentages": broken_percentages,
                "broken_grain_count": broken_grain_count,
                "stone_count": stone_count,
                "husk_count": husk_count
            })
            
        except ImportError as e:
            print(f"Import error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Import error: {str(e)}"}), 500
        except Exception as e:
            print(f"Processing error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Route error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/process_dal', methods=['POST'])
def process_dal_route():
    """
    Processes an image to detect and analyze dal grains.
    Returns detailed analysis results including broken percentages and black dal count.
    """
    return jsonify({"error": "DAL processing not yet implemented"}), 501


@app.route('/gallery')
def gallery():
    """Display gallery of processed images"""
    try:
        processed_images = sorted(
            [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith('.jpg')],
            key=lambda x: os.path.getctime(os.path.join(PROCESSED_FOLDER, x)),
            reverse=True
        )
        return jsonify({"images": processed_images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/clear_processed', methods=['POST'])
def clear_processed():
    """Clear all processed images"""
    try:
        for f in os.listdir(PROCESSED_FOLDER):
            os.remove(os.path.join(PROCESSED_FOLDER, f))
        return jsonify({"status": "success", "message": "Processed folder cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        from process_image_rpi_final import load_model_once
        model_ready = load_model_once()
        
        return jsonify({
            "status": "ok" if model_ready else "model_not_loaded",
            "model_loaded": model_ready,
            "timestamp": datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.now().isoformat()
        }), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    print(f"Internal server error: {str(e)}")
    print(traceback.format_exc())
    return jsonify({"error": "Internal server error"}), 500


# ==================== STARTUP ====================

if __name__ == '__main__':
    ensure_loopback_available()
    
    print("\n" + "="*50)
    print("Rice Grader Flask Application")
    print("="*50)
    print(f"Root path: {app.root_path}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Capture folder: {CAPTURE_FOLDER}")
    print(f"Processed folder: {PROCESSED_FOLDER}")
    print("="*50 + "\n")
    
    # Start Flask on RPi (0.0.0.0 for external access)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
