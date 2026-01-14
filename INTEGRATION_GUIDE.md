# Changes Required for New Keras Transfer Learning Model

## Summary
This document outlines all changes needed to integrate your new Keras transfer learning model into the Raspberry Pi rice grading system.

---

## 1. NEW FILE CREATED: process_image_updated.py

**Location:** `a:\RVCE\SECOND YEAR\EL Sem 3\compiled\process_image_updated.py`

**Status:** ✅ CREATED

**Key Features:**
- Loads Keras model at startup
- Uses watershed segmentation to detect grains
- Extracts grain crops and saves them to 'grains' folder
- Performs batch ML classification using your model
- Returns all classification counts (perfect, chalky, black, yellow, brown, husk)

**CONFIGURATION (Lines 9-16):**
```python
IMAGE_SIZE = (224, 224)  # EfficientNet standard input size
MODEL_PATH = '/home/rvce/Desktop/compiled/efficientnet_rice_final_inference.keras'
LABEL_MAP = {0: "black", 1: "brown", 2: "yellow", 3: "chalky", 4: "perfect", 5: "husk"}
```

**✅ CONFIGURED:**
1. ✅ MODEL_PATH set to `efficientnet_rice_final_inference.keras`
2. ✅ IMAGE_SIZE set to 224×224 (EfficientNet standard)
3. ✅ EfficientNet preprocessing configured
4. ⚠️ **VERIFY:** Check if your LABEL_MAP indices match your model's output

---

## 2. FILE TO UPDATE: app.py

**Location:** `a:\RVCE\SECOND YEAR\EL Sem 3\compiled\app.py`

**Change Required:** Update import statement to use new file

**Current Code (Line 240):**
```python
from process_image import detect_and_count_rice_grains
```

**New Code:**
```python
from process_image_updated import process_image
```

**Also Update Processing Logic (Lines 242-256):**
 
**Current Code:**
```python
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
brown_count = 0
```

**New Code:**
```python
processed_result = process_image(image)

# Unpack results from process_image (10 values)
final_image = processed_result[0]
full_grain_count = processed_result[1]  # perfect count
chalky_count = processed_result[2]
black_count = processed_result[3]
yellow_count = processed_result[4]
brown_count = processed_result[5]
broken_percentages = processed_result[6]
broken_grain_count = processed_result[7]
stone_count = processed_result[8]
husk_count = processed_result[9]
```

---

## 3. FILE TO UPDATE: config.py (Optional but Recommended)

**Location:** `a:\RVCE\SECOND YEAR\EL Sem 3\compiled\config.py`

**Add Model Configuration:**
```python
# Model Configuration for Rice Classification
MODEL_PATH = "/home/rvce/Desktop/compiled/your_new_model.keras"
MODEL_IMAGE_SIZE = (224, 224)  # Adjust based on your model
NUM_CLASSES = 6
```

Then update `process_image_updated.py` to import from config:
```python
from config import MODEL_PATH, MODEL_IMAGE_SIZE, NUM_CLASSES
```

---

## 4. RASPBERRY PI OPTIMIZATION RECOMMENDATIONS

### A. Install TensorFlow Lite (Faster on RPi)

**Convert your model to TFLite:**
```python
# Run this on your development machine
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('rice_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Then modify `process_image_updated.py` to use TFLite:**
```python
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_grain(grain_crop):
    img = preprocess_grain_image(grain_crop)
    img_batch = np.expand_dims(img, axis=0).astype(np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    return LABEL_MAP[predicted_class], confidence
```

### B. Memory Management on RPi

Add to top of `process_image_updated.py`:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Limit TensorFlow memory usage
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

### C. Install Required Packages on RPi

```bash
# On Raspberry Pi, run:
pip3 install tensorflow  # or tensorflow-lite for lighter version
pip3 install opencv-python
pip3 install numpy
pip3 install flask
```

---

## 5. OTHER FILES TO CHECK

### Check if these files import process_image:

Run search in workspace:
```bash
grep -r "from process_image import\|import process_image" .
```

**Files that might need updates:**
- ✅ `app.py` - Already identified
- `process_image_copy.py` - Backup file (no change needed)
- `p_copy.py` - Check if it imports process_image
- `process_brown_rice.py` - Check if it imports process_image
- `procress_dal.py` - Check if it imports process_image (note: typo in filename)
- `mongodb_sync.py` - Might reference image processing
- `mongo_sync_standalone.py` - Might reference image processing

---

## 6. TESTING CHECKLIST

Before deploying to Raspberry Pi:

- [ ] Update MODEL_PATH with actual filename
- [ ] Update IMAGE_SIZE to match your model
- [ ] Configure correct preprocessing (VGG/ResNet/MobileNet/etc.)
- [ ] Update LABEL_MAP if your class indices differ
- [ ] Test process_image_updated.py standalone (run main function)
- [ ] Update app.py import and unpacking logic
- [ ] Test full Flask application locally
- [ ] Transfer model file to RPi at correct path
- [ ] Install all dependencies on RPi
- [ ] Test on RPi with sample images
- [ ] Monitor memory usage on RPi
- [ ] Consider converting to TFLite for better RPi performance

---

## 7. DEPLOYMENT STEPS FOR RASPBERRY PI

1. **Transfer files to RPi:**
   ```bash
   scp process_image_updated.py rvce@raspberrypi:/home/rvce/Desktop/compiled/
   scp your_new_model.keras rvce@raspberrypi:/home/rvce/Desktop/compiled/
   ```

2. **Update app.py on RPi:**
   - Change import statement
   - Update unpacking logic

3. **Test the model loads correctly:**
   ```bash
   cd /home/rvce/Desktop/compiled/
   python3 process_image_updated.py
   ```

4. **Restart Flask application:**
   ```bash
   sudo systemctl restart your-flask-service
   # OR
   python3 app.py
   ```

5. **Monitor logs for errors:**
   ```bash
   tail -f /var/log/your-app.log
   ```

---

## 8. TROUBLESHOOTING

### Common Issues:

**Model not found:**
- Verify MODEL_PATH is correct
- Check file exists: `ls -la /home/rvce/Desktop/compiled/*.keras`

**Out of memory on RPi:**
- Convert to TensorFlow Lite
- Reduce IMAGE_SIZE if possible
- Process fewer grains per batch

**Wrong predictions:**
- Verify preprocessing matches training
- Check LABEL_MAP indices match model output
- Ensure IMAGE_SIZE matches training

**Slow performance:**
- Use TensorFlow Lite instead of full TensorFlow
- Consider using MobileNetV2 base model
- Reduce image size if accuracy permits

---

## QUESTIONS TO ANSWER:

1. **What is your new model filename?**
   - Update MODEL_PATH in process_image_updated.py

2. **What input size does your model expect?**
   - Update IMAGE_SIZE (common: 224x224, 299x299)

3. **Which transfer learning base did you use?**
   - VGG16/VGG19
   - ResNet50
   - MobileNetV2
   - InceptionV3
   - Other?
   
   This determines the preprocessing function to use.

4. **What are your class labels and indices?**
   - Verify LABEL_MAP matches your model's output

5. **What preprocessing was used during training?**
   - Must replicate exactly in preprocess_grain_image()
