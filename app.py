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
        if not frame:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_locally(data, grain_type):
    """Save analysis results locally as JSON files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    storage_dir = RICE_STORAGE if grain_type == 'rice' else DAL_STORAGE
    
    # Add timestamp to data
    data['timestamp'] = timestamp
    data['device_id'] = MAC_ADDRESS
    data['synced'] = False
    
    # Save to file
    filename = f"{grain_type}_{timestamp}.json"
    filepath = os.path.join(storage_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved {grain_type} results locally: {filename}")
    return filename

# Routes
@app.route('/video_feed')
def video_feed():
    """Live streaming route."""
    cam = initialize_camera()
    if cam:
        return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("Camera not available", status=503)

@app.route('/capture', methods=['POST'])
def capture():
    """Captures an image from the camera and saves it."""
    cam = initialize_camera()
    if not cam:
        return jsonify({"error": "Camera not available"}), 503
        
    frame = cam.get_frame()
    if frame:
        timestamp = int(time.time())  # Unique filename based on timestamp
        filename = f"captured_{timestamp}.jpg"
        filepath = os.path.join(app.config['CAPTURE_FOLDER'], filename)
        print(f"Saving captured image to: {filepath}")
        try:
            with open(filepath, "wb") as f:
                f.write(frame)
            print(f"Successfully saved image, size: {len(frame)} bytes")
            manage_captured_images()  # Keep only the last 10 images
            return jsonify({"image_url": url_for('static', filename=f'captured/{filename}')})
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return jsonify({"error": f"Capture failed: {str(e)}"}), 500
    print("Camera returned no frame data")
    return jsonify({"error": "Capture failed: No frame data"}), 500

@app.route('/process_image', methods=['POST'])
def process_image_route():
    """
    Processes an image to detect and analyze rice grains, stones, and husks.
    Returns detailed analysis results including different types of grains.
    """
    # data = request.get_json()
    # image_path = data.get("image_path")

    # if not image_path:
    #     return jsonify({"error": "Invalid request"}), 400

    # # Build full path to the image (remove any leading '/' if present)
    # image_full_path = os.path.join(app.root_path, image_path.lstrip('/'))
    
    # image = cv2.imread(image_full_path)
    # if image is None:
    #     return jsonify({"error": "Image not found"}), 404

    # # Import process_image lazily to avoid import errors at startup
    # try:
    #     from process_image import process_image
    #     processed_result = process_image(image)
        
    #     # Unpack results from the tuple (updated to match new return values)
    #     final_image = processed_result[0]
    #     full_grain_count = processed_result[1]
    #     chalky_count = processed_result[2]
    #     black_count = processed_result[3]
    #     yellow_count = processed_result[4]
    #     brown_count = processed_result[5]
    #     broken_percentages = processed_result[6]
    #     broken_grain_count = processed_result[7]
    #     stone_count = processed_result[8]
    #     husk_count = processed_result[9]

    #     # Calculate total count
    #     total_objects = full_grain_count + chalky_count + black_count + yellow_count + brown_count + broken_grain_count + stone_count + husk_count

    #     # Save processed image with a timestamp-based filename
    #     processed_filename = f"processed_{int(time.time())}.jpg"
    #     processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
    #     cv2.imwrite(processed_filepath, final_image)

    #     # Cleanup old processed images (keep only 50)
    #     cleanup_old_images(PROCESSED_FOLDER, max_files=MAX_IMAGES)

    #     return jsonify({
    #         "processed_image_url": url_for('static', filename=f'processed/{processed_filename}'),
    #         "total_objects": total_objects,
    #         "full_grain_count": full_grain_count,
    #         "chalky_count": chalky_count,
    #         "black_count": black_count,
    #         "yellow_count": yellow_count,
    #         "brown_count": brown_count,
    #         "broken_percentages": broken_percentages,
    #         "broken_grain_count": broken_grain_count,
    #         "stone_count": stone_count,
    #         "husk_count": husk_count
    #     })
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Invalid request"}), 400

    # Build full path to the image (remove any leading '/' if present)
    image_full_path = os.path.join(app.root_path, image_path.lstrip('/'))
    
    image = cv2.imread(image_full_path)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    # Import process_image lazily to avoid import errors at startup
    try:
        from process_image import detect_and_count_rice_grains
        processed_result = detect_and_count_rice_grains(image)
        
        # Unpack results from the new function (7 values)
        final_image = processed_result[0]
        full_grain_count = processed_result[1]
        broken_grain_count = processed_result[2]
        chalky_count = processed_result[3]
        black_count = processed_result[4]
        yellow_count = processed_result[5]
        broken_percentages = processed_result[6]

        # Set default values for stone and husk since new version doesn't detect them
        stone_count = 0
        husk_count = 0
        brown_count = 0  # Also not detected in new version

        # Calculate total count
        total_objects = full_grain_count + chalky_count + black_count + yellow_count + brown_count + broken_grain_count + stone_count + husk_count

        # Save processed image with a timestamp-based filename
        processed_filename = f"processed_{int(time.time())}.jpg"
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_filepath, final_image)

        # Cleanup old processed images (keep only 50)
        cleanup_old_images(PROCESSED_FOLDER, max_files=MAX_IMAGES)

        return jsonify({
            "processed_image_url": url_for('static', filename=f'processed/{processed_filename}'),
            "total_objects": total_objects,
            "full_grain_count": full_grain_count,
            "chalky_count": chalky_count,
            "black_count": black_count,
            "yellow_count": yellow_count,
            "brown_count": brown_count,
            "broken_percentages": broken_percentages,
            "broken_grain_count": broken_grain_count,
            "stone_count": stone_count,
            "husk_count": husk_count
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process_dal', methods=['POST'])
def process_dal_route():
    """
    Processes an image to detect and analyze dal grains.
    Returns detailed analysis results including broken percentages and black dal count.
    """
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "Invalid request"}), 400

    # Build full path to the image (remove any leading '/' if present)
    image_full_path = os.path.join(app.root_path, image_path.lstrip('/'))
    
    image = cv2.imread(image_full_path)
    if image is None:
        return jsonify({"error": "Image not found"}), 404

    # Process dal image with enhanced process_dal function
    try:
        from procress_dal import process_dal
        full_grain_count, broken_grain_count, broken_percent, visualization_image, black_dal = process_dal(image)

        # Save processed image with a timestamp-based filename
        processed_filename = f"processed_dal_{int(time.time())}.jpg"
        processed_filepath = os.path.join(PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(processed_filepath, visualization_image)

        # Cleanup old processed images (keep only 50)
        cleanup_old_images(PROCESSED_FOLDER, max_files=MAX_IMAGES)
        
        return jsonify({
            "processed_image_url": url_for('static', filename=f'processed/{processed_filename}'),
            "full_grain_count": full_grain_count,
            "broken_grain_count": broken_grain_count,
            "broken_percent": broken_percent,
            "black_dal": black_dal  # Fixed: changed from black_dal_count to black_dal
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_results', methods=['POST'])
def save_results():
    """
    Receives and saves analysis results locally.
    Tries to start MongoDB sync but won't block if it fails.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Determine if it's dal or rice result and save accordingly
        if data.get('total_objects', 0) > 0 or data.get('full_grain_count', 0) > 0:
            if 'chalky_count' in data:
                print("==== RICE RESULT SAVED ====")
                grain_type = 'rice'
            else:
                print("==== DAL RESULT SAVED ====")
                grain_type = 'dal'
                
            filename = save_locally(data, grain_type)
            
            # Try to start MongoDB sync but don't block if it fails
            try:
                # Start a separate thread that attempts to import and sync
                def try_sync():
                    try:
                        # We're explicitly using a try/except here since imports might fail
                        import importlib
                        mongo_sync = importlib.import_module('mongodb_sync')
                        mongo_sync.attempt_sync_to_mongodb(app.root_path)
                    except Exception as e:
                        print(f"MongoDB sync failed but app continues: {e}")
                
                # Start the sync in a background thread
                sync_thread = threading.Thread(target=try_sync)
                sync_thread.daemon = True
                sync_thread.start()
                
            except Exception as e:
                # Even if thread creation fails, we still return success for saving locally
                print(f"Could not start sync thread: {e}")
                
            return jsonify({
                "success": True,
                "message": f"Results saved locally as {filename}"
            })
        else:
            return jsonify({"error": "No valid results to save"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/wifi', methods=['GET', 'POST'])
def wifi():
    """Handles WiFi configuration."""
    if request.method == 'POST':
        ssid = request.form.get('ssid')
        password = request.form.get('password')
        # This will be handled by AJAX calls to /wifi/connect
    return render_template('wifi.html')

@app.route('/wifi/scan', methods=['GET'])
def wifi_scan():
    """Scan for available WiFi networks."""
    try:
        # Import wifi_manager here to avoid startup errors
        from wifi_manager import scan_networks
        networks = scan_networks()
        return jsonify({
            'success': True,
            'networks': networks
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/wifi/connect', methods=['POST'])
def wifi_connect():
    """Connect to a WiFi network."""
    try:
        data = request.get_json()
        ssid = data.get('ssid')
        password = data.get('password')
        
        if not ssid:
            return jsonify({
                'success': False,
                'error': 'SSID is required'
            })
        
        # Import here to avoid startup issues
        from wifi_manager import connect_to_network
        success, message = connect_to_network(ssid, password)
        
        return jsonify({
            'success': success,
            'error' if not success else 'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/wifi/status', methods=['GET'])
def wifi_status():
    """Get current WiFi connection status."""
    try:
        # Import here to avoid startup issues
        from wifi_manager import get_connection_status
        status = get_connection_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'connected': False,
            'error': str(e)
        })

@app.route('/wifi/disconnect', methods=['POST'])
def wifi_disconnect():
    """Disconnect from the current WiFi network."""
    try:
        from wifi_manager import disconnect_from_network
        success, message = disconnect_from_network()
        
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles file upload and renders the main page."""
    uploaded_image = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        uploaded_image = filepath
    return render_template('index.html', uploaded_image=uploaded_image)

@app.route('/system/shutdown', methods=['POST'])
def system_shutdown():
    try:
        # Respond first, then shutdown
        threading.Thread(target=lambda: (time.sleep(1), os.system('sudo shutdown now'))).start()
        return jsonify({"success": True, "message": "System will shutdown now."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/system/restart', methods=['POST'])
def system_restart():
    try:
        threading.Thread(target=lambda: (time.sleep(1), os.system('sudo reboot'))).start()
        return jsonify({"success": True, "message": "System will restart now."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    def get_mac_address():
        mac = uuid.getnode()
        mac_address = ':'.join(['{:02x}'.format((mac >> ele) & 0xff) for ele in range(40, -1, -8)])
        return mac_address
    
    # Make sure loopback interface is ready
    ensure_loopback_available()
    
    # # Check MAC address if not in offline mode
    # if os.environ.get('FLASK_OFFLINE_MODE') != '1':
    #     mac = get_mac_address()
    #     print(f"Current MAC Address: {mac}")
        
    #     if mac != MAC_ADDRESS:
    #         print("MAC address does not match with original. Exiting...")
    #         exit(1)
    # else:
    #     print("Running in offline mode - skipping MAC check")
        
    # Start Flask app with specific host bindings to ensure localhost works
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)