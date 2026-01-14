# 🍚 Raspberry Pi Deployment Guide - EfficientNet Rice Grader

## Final Working Code for RPi

This guide provides the production-ready code and setup for your Raspberry Pi rice grading system with EfficientNet transfer learning model.

---

## 📦 Files Overview

### 1. **process_image_rpi_final.py** ✅
   - RPi-optimized image processing module
   - Uses EfficientNet for ML classification
   - Efficient memory management
   - Proper error handling

### 2. **app_rpi_final.py** ✅
   - Updated Flask app for RPi
   - Correct imports and unpacking
   - Enhanced error logging
   - Health check endpoint

---

## 🚀 Deployment Steps

### Step 1: Prepare Raspberry Pi

```bash
# SSH into RPi
ssh pi@raspberrypi.local
# or
ssh pi@192.168.x.x

# Update system
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2: Install Dependencies

```bash
# Install Python packages
pip3 install tensorflow
pip3 install opencv-python
pip3 install opencv-contrib-python
pip3 install numpy
pip3 install flask
pip3 install opencv-python-headless  # For RPi without display

# Optional: Install TensorFlow Lite (lighter, faster)
pip3 install tflite-runtime
```

### Step 3: Transfer Files to RPi

```bash
# From your development machine
# Transfer the processing module
scp process_image_rpi_final.py pi@raspberrypi.local:/home/pi/Desktop/compiled/

# Transfer the Flask app
scp app_rpi_final.py pi@raspberrypi.local:/home/pi/Desktop/compiled/

# Transfer the EfficientNet model
scp efficientnet_rice_final_inference.keras pi@raspberrypi.local:/home/pi/Desktop/compiled/

# Verify transfer
ssh pi@raspberrypi.local ls -lh /home/pi/Desktop/compiled/
```

### Step 4: Update File Permissions

```bash
ssh pi@raspberrypi.local chmod +x /home/pi/Desktop/compiled/app_rpi_final.py
ssh pi@raspberrypi.local chmod +x /home/pi/Desktop/compiled/process_image_rpi_final.py
```

### Step 5: Backup Original Files

```bash
ssh pi@raspberrypi.local << 'EOF'
cd /home/pi/Desktop/compiled/
cp app.py app.py.backup
cp process_image.py process_image.py.backup
EOF
```

### Step 6: Replace Files with New Versions

```bash
ssh pi@raspberrypi.local << 'EOF'
cd /home/pi/Desktop/compiled/
# Copy the new optimized versions
cp app_rpi_final.py app.py
cp process_image_rpi_final.py process_image.py
EOF
```

### Step 7: Verify Model Path

```bash
ssh pi@raspberrypi.local << 'EOF'
# Check if model exists
ls -lh /home/pi/Desktop/compiled/efficientnet_rice_final_inference.keras
# Should show the model file
EOF
```

### Step 8: Test Processing Module

```bash
ssh pi@raspberrypi.local << 'EOF'
cd /home/pi/Desktop/compiled/
python3 process_image.py
EOF
```

Expected output:
```
Loading EfficientNet model...
✓ Model loaded successfully!
```

### Step 9: Start Flask Application

```bash
# Option 1: Direct run (for testing)
ssh pi@raspberrypi.local << 'EOF'
cd /home/pi/Desktop/compiled/
python3 app.py
EOF
```

Expected output:
```
==================================================
Rice Grader Flask Application
==================================================
Root path: /home/pi/Desktop/compiled
...
 * Running on http://0.0.0.0:5000
```

### Step 10: Create SystemD Service (Optional but Recommended)

```bash
# Create service file
ssh pi@raspberrypi.local << 'EOF'
sudo nano /etc/systemd/system/rice-grader.service
EOF
```

Paste this content:
```ini
[Unit]
Description=Rice Grader Flask Application
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Desktop/compiled
ExecStart=/usr/bin/python3 /home/pi/Desktop/compiled/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
ssh pi@raspberrypi.local << 'EOF'
sudo systemctl daemon-reload
sudo systemctl enable rice-grader.service
sudo systemctl start rice-grader.service
sudo systemctl status rice-grader.service
EOF
```

---

## 🧪 Testing

### Test 1: Check Model Loading

```bash
curl http://raspberrypi.local:5000/health
```

Expected response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-01-14T10:30:45.123456"
}
```

### Test 2: Capture Image

```bash
curl -X POST http://raspberrypi.local:5000/capture
```

Expected response:
```json
{
  "status": "success",
  "image_url": "/static/captured/captured_1234567890.jpg",
  "timestamp": 1234567890
}
```

### Test 3: Process Image

```bash
curl -X POST http://raspberrypi.local:5000/process_image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/static/captured/captured_1234567890.jpg"}'
```

Expected response:
```json
{
  "status": "success",
  "processed_image_url": "/static/processed/processed_1234567890.jpg",
  "total_objects": 45,
  "perfect_count": 32,
  "chalky_count": 5,
  "black_count": 2,
  "yellow_count": 3,
  "brown_count": 2,
  "broken_grain_count": 1,
  "stone_count": 0,
  "husk_count": 0,
  "broken_percentages": {
    "25%": 0,
    "50%": 1,
    "75%": 0
  }
}
```

### Test 4: Check Logs

```bash
# Direct run logs
ssh pi@raspberrypi.local tail -f /tmp/rice-grader.log

# Or systemd logs
ssh pi@raspberrypi.local sudo journalctl -u rice-grader.service -f
```

---

## 🔧 Configuration

### Adjust Model Path

If your model is at a different location:

```bash
# Edit app.py on RPi
ssh pi@raspberrypi.local << 'EOF'
nano /home/pi/Desktop/compiled/process_image.py
EOF
```

Find and update line 17:
```python
MODEL_PATH = '/your/actual/model/path/efficientnet_rice_final_inference.keras'
```

### Adjust Flask Settings

Edit `app.py`:
- Change `host` from `'0.0.0.0'` to `'127.0.0.1'` if you want local-only access
- Change `port` from `5000` to another port if needed
- Set `debug=True` for development, `debug=False` for production

---

## 📊 Performance Optimization

### 1. Reduce TensorFlow Logging

Already included in `process_image.py`:
```python
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

### 2. Monitor Memory Usage

```bash
# SSH into RPi
watch -n 1 free -h
# or
top -p $(pgrep -f app.py)
```

### 3. Monitor CPU Usage

```bash
# Check CPU frequency and temperature
vcgencmd measure_temp
vcgencmd measure_clock arm
```

### 4. Convert to TensorFlow Lite (Advanced)

If you need faster inference:

```python
# On development machine, convert model to TFLite
import tensorflow as tf

model = tf.keras.models.load_model('efficientnet_rice_final_inference.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

Then update `process_image.py` to use TFLite (see INTEGRATION_GUIDE.md for details).

---

## ⚠️ Troubleshooting

### Issue: "Model not found" Error

```bash
# SSH to RPi and verify model path
ssh pi@raspberrypi.local ls -lh /home/pi/Desktop/compiled/efficientnet_rice_final_inference.keras

# If not found, transfer it again
scp efficientnet_rice_final_inference.keras pi@raspberrypi.local:/home/pi/Desktop/compiled/
```

### Issue: Out of Memory Error

```bash
# Check available memory
free -h

# Kill unnecessary processes
sudo killall chromium
sudo killall firefox

# Restart Flask with memory limits
# Edit systemd service to add memory limit:
# MemoryLimit=512M
```

### Issue: "Import Error" for TensorFlow

```bash
# Reinstall TensorFlow
pip3 uninstall tensorflow -y
pip3 install tensorflow

# Or use TensorFlow Lite (lighter)
pip3 install tflite-runtime
```

### Issue: Slow Processing

```bash
# Check if model is quantized
# Convert to INT8 quantized model for faster inference

# Monitor processing time
# Edit app.py to add timing:
import time
start = time.time()
result = process_image(image)
print(f"Processing took {time.time() - start:.2f}s")
```

### Issue: Flask Connection Refused

```bash
# Check if Flask is running
ps aux | grep app.py

# Check if port 5000 is in use
sudo netstat -tlnp | grep 5000

# Change port in app.py if needed
app.run(host='0.0.0.0', port=5001, ...)
```

### Issue: Camera Not Found

```bash
# Check if camera is properly connected
vcgencmd get_camera

# Enable camera in raspi-config
sudo raspi-config
# Enable Camera in Interface Options
```

---

## 📝 File Checklist

Before deployment, ensure you have:

- ✅ `process_image_rpi_final.py` - Processing module
- ✅ `app_rpi_final.py` - Flask application  
- ✅ `efficientnet_rice_final_inference.keras` - Model file
- ✅ `templates/index.html` - Web interface
- ✅ `static/css/styles.css` - Styling
- ✅ `static/js/script.js` - Frontend logic
- ✅ `config.py` - Configuration
- ✅ `camera.py` - Camera module (if available)

---

## 🔐 Security Notes

For production on RPi:

1. **Add Authentication:**
   ```python
   from flask_httpauth import HTTPBasicAuth
   auth = HTTPBasicAuth()
   
   @auth.verify_password
   def verify_password(username, password):
       if username == 'admin' and password == 'your_secure_password':
           return username
   
   @app.route('/process_image', methods=['POST'])
   @auth.login_required
   def process_image_route():
       ...
   ```

2. **Use HTTPS:**
   ```python
   app.run(ssl_context='adhoc')  # Requires pyopenssl
   ```

3. **Restrict IP Access:**
   ```python
   app.run(host='192.168.x.x')  # Only accessible from specific IP
   ```

4. **Rate Limiting:**
   ```python
   from flask_limiter import Limiter
   limiter = Limiter(app)
   
   @app.route('/process_image', methods=['POST'])
   @limiter.limit("10 per minute")
   def process_image_route():
       ...
   ```

---

## 📞 Support

If you encounter issues:

1. Check `/tmp/rice-grader.log` for error messages
2. Run `python3 process_image.py` to test processing module independently
3. Verify model file exists: `ls -lh efficientnet_rice_final_inference.keras`
4. Check TensorFlow installation: `python3 -c "import tensorflow as tf; print(tf.__version__)"`

---

## 🎉 Ready to Deploy!

Your RPi rice grader is now configured with:
- ✅ EfficientNet transfer learning model
- ✅ Optimized image processing
- ✅ Error handling and logging
- ✅ Health check endpoints
- ✅ Production-ready Flask app

Deploy and test it! 🚀
