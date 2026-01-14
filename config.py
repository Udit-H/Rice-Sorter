# MongoDB Configuration
MONGO_URI = "mongodb+srv://ricegrader115:Rice%40115@cluster0.ajq7rxy.mongodb.net/grain_analyzer?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "grain_analyzer"
RICE_COLLECTION = "rice_analysis"
DAL_COLLECTION = "dal_analysis"

# Local Storage Configuration
LOCAL_STORAGE_DIR = "local_storage"
SYNC_INTERVAL_SECONDS = 120  # Check every 2 minutes

# Model Configuration for Rice Classification
MODEL_PATH = "/home/rvce/Desktop/compiled/efficientnet_rice_final_inference.keras"
MODEL_IMAGE_SIZE = (224, 224)  # EfficientNet standard input
NUM_CLASSES = 6