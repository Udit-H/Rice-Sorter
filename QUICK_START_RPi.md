# 🚀 QUICK DEPLOYMENT CHECKLIST FOR RPi

## Pre-Deployment Verification

### ✅ Local Testing (on your development machine)
- [ ] Test `process_image_rpi_final.py` standalone
- [ ] Verify model loads: `python3 process_image_rpi_final.py`
- [ ] Check model file exists: `ls -lh efficientnet_rice_final_inference.keras`

---

## RPi Setup Commands (Copy-Paste Ready)

### 1. Install Dependencies
```bash
pip3 install tensorflow opencv-python numpy flask
```

### 2. Transfer Files
```bash
# Replace X.X.X.X with your RPi IP address
scp process_image_rpi_final.py pi@X.X.X.X:/home/pi/Desktop/compiled/
scp app_rpi_final.py pi@X.X.X.X:/home/pi/Desktop/compiled/
scp efficientnet_rice_final_inference.keras pi@X.X.X.X:/home/pi/Desktop/compiled/
```

### 3. Backup Original Files
```bash
ssh pi@X.X.X.X << 'EOF'
cd /home/pi/Desktop/compiled/
cp app.py app.py.backup
cp process_image.py process_image.py.backup
EOF
```

### 4. Replace with New Code
```bash
ssh pi@X.X.X.X << 'EOF'
cd /home/pi/Desktop/compiled/
cp app_rpi_final.py app.py
cp process_image_rpi_final.py process_image.py
EOF
```

### 5. Test Model Loading
```bash
ssh pi@X.X.X.X << 'EOF'
cd /home/pi/Desktop/compiled/
python3 process_image.py
EOF
```

Expected: `✓ Model loaded successfully!`

### 6. Start Flask
```bash
ssh pi@X.X.X.X << 'EOF'
cd /home/pi/Desktop/compiled/
python3 app.py
EOF
```

Expected: `* Running on http://0.0.0.0:5000`

---

## Test Endpoints from Your Machine

### Health Check
```bash
curl http://X.X.X.X:5000/health
```

### Capture Image
```bash
curl -X POST http://X.X.X.X:5000/capture
```

### Process Image
```bash
curl -X POST http://X.X.X.X:5000/process_image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/static/captured/captured_TIMESTAMP.jpg"}'
```

Replace `TIMESTAMP` with actual timestamp from capture response.

---

## Run as Background Service

### Option 1: Screen Session (Simple)
```bash
ssh pi@X.X.X.X << 'EOF'
screen -S rice-grader -d -m python3 /home/pi/Desktop/compiled/app.py
screen -ls  # Check if running
EOF
```

### Option 2: SystemD Service (Production)
```bash
ssh pi@X.X.X.X << 'EOF'
sudo tee /etc/systemd/system/rice-grader.service > /dev/null << 'SERVICE'
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
SERVICE

sudo systemctl daemon-reload
sudo systemctl enable rice-grader.service
sudo systemctl start rice-grader.service
sudo systemctl status rice-grader.service
EOF
```

---

## Troubleshooting

### Check if Model is Loading
```bash
ssh pi@X.X.X.X python3 -c "from process_image import load_model_once; load_model_once()"
```

### Check Model File
```bash
ssh pi@X.X.X.X ls -lh /home/pi/Desktop/compiled/efficientnet_rice_final_inference.keras
```

### View Flask Logs (if using systemd)
```bash
ssh pi@X.X.X.X sudo journalctl -u rice-grader.service -f
```

### Check Memory Usage
```bash
ssh pi@X.X.X.X free -h
```

### Check Port 5000 Status
```bash
ssh pi@X.X.X.X sudo netstat -tlnp | grep 5000
```

---

## Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "Model not found" | Transfer model file: `scp efficientnet_rice_final_inference.keras pi@X.X.X.X:/home/pi/Desktop/compiled/` |
| Connection refused | Check Flask is running: `ssh pi@X.X.X.X ps aux \| grep app.py` |
| Out of memory | Check free memory: `ssh pi@X.X.X.X free -h` and kill unnecessary processes |
| ImportError TensorFlow | Reinstall: `ssh pi@X.X.X.X pip3 install --upgrade tensorflow` |
| Camera not found | Enable camera: `ssh pi@X.X.X.X sudo raspi-config` (Interface Options → Camera) |
| Slow processing | Check RPi temperature: `ssh pi@X.X.X.X vcgencmd measure_temp` (throttling if >80°C) |

---

## Final Verification

Once running on RPi, verify:

```bash
# 1. Model loads correctly
curl http://X.X.X.X:5000/health
# Expected: {"status": "ok", "model_loaded": true}

# 2. Can capture images
curl -X POST http://X.X.X.X:5000/capture
# Expected: {"status": "success", ...}

# 3. Can process images (after capturing one)
curl -X POST http://X.X.X.X:5000/process_image \
  -H "Content-Type: application/json" \
  -d '{"image_path": "/static/captured/captured_1234567890.jpg"}'
# Expected: {"status": "success", "perfect_count": X, ...}
```

---

## You're Ready! 🎉

Your Raspberry Pi rice grader with EfficientNet is now deployed and ready for use.

### Files Used:
- ✅ `process_image_rpi_final.py` → renamed to `process_image.py`
- ✅ `app_rpi_final.py` → renamed to `app.py`
- ✅ `efficientnet_rice_final_inference.keras` (model file)

### What's New:
- EfficientNet ML model integration
- Proper grain classification
- Error logging and debugging
- Health check endpoint
- Production-ready Flask app

---

## Support
See `RPI_DEPLOYMENT_GUIDE.md` for detailed troubleshooting and advanced configurations.
