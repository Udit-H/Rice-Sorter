#!/usr/bin/env python3
"""
Standalone MongoDB synchronization script.
Run this in a separate terminal to handle data syncing with MongoDB
without affecting the main Flask application.
"""
import pymongo
import os
import json
import glob
import time
import socket
import logging
import datetime
import sys
from pymongo.errors import ConnectionFailure
import config
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='mongodb_sync_standalone.log',
    filemode='a'
)
logger = logging.getLogger('mongodb_sync_standalone')

# Add console handler to show logs in terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logger.addHandler(console)

# Global MongoDB client
_mongodb_client = None
_last_connection_time = None
_running = True

def signal_handler(sig, frame):
    """Handle interrupt signals to gracefully exit the script."""
    global _running
    logger.info("Received shutdown signal. Cleaning up...")
    _running = False
    if _mongodb_client is not None:
        try:
            _mongodb_client.close()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_internet_connection():
    """Check if there is an internet connection available."""
    try:
        # Try to connect to Google DNS server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except (socket.timeout, socket.error):
        return False

def validate_mongodb_connection(client):
    """Validate if MongoDB connection is still alive."""
    try:
        # The ismaster command is cheap and does not require auth
        client.admin.command('ismaster')
        return True
    except Exception:
        return False

def connect_to_mongodb():
    """Connect to MongoDB and return client."""
    global _mongodb_client, _last_connection_time
    
    # First check internet connection
    if not check_internet_connection():
        logger.warning("No internet connection detected, skipping MongoDB connection attempt")
        return None
    
    # If client exists, check if connection is still valid
    if _mongodb_client is not None:
        if validate_mongodb_connection(_mongodb_client):
            logger.debug("Reusing existing MongoDB connection")
            return _mongodb_client
        else:
            # Close stale connection
            try:
                logger.info("Closing stale MongoDB connection")
                _mongodb_client.close()
            except:
                pass
            _mongodb_client = None
    
    # Create a new connection with shorter timeout
    try:
        logger.info("Attempting to connect to MongoDB...")
        client = pymongo.MongoClient(
            config.MONGO_URI, 
            serverSelectionTimeoutMS=3000,  # Shorter timeout (3 seconds)
            connectTimeoutMS=3000,
            socketTimeoutMS=3000
        )
        
        # Quick validation check with timeout
        client.admin.command('ismaster', serverSelectionTimeoutMS=3000)
        
        # Store the client and connection time
        _mongodb_client = client
        _last_connection_time = datetime.datetime.now()
        logger.info("New MongoDB connection established successfully")
        return client
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failure: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        return None

def create_rice_document(data):
    """Create a MongoDB document for rice analysis results."""
    return {
        "total_objects": data.get("total_objects", 0),
        "full_grain_count": data.get("full_grain_count", 0),
        "broken_grain_count": data.get("broken_grain_count", 0),
        "chalky_count": data.get("chalky_count", 0),
        "black_count": data.get("black_count", 0),
        "yellow_count": data.get("yellow_count", 0),
        "brown_count": data.get("brown_count", 0),
        "stone_count": data.get("stone_count", 0),
        "husk_count": data.get("husk_count", 0),
        "broken_percentages": {
            "25%": data.get("broken_percentages", {}).get("25%", 0),
            "50%": data.get("broken_percentages", {}).get("50%", 0),
            "75%": data.get("broken_percentages", {}).get("75%", 0),
        },
        "device_id": data.get("device_id", "unknown"),
        "timestamp": data.get("timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        "created_at": datetime.datetime.now(),
    }

def create_dal_document(data):
    """Create a MongoDB document for dal analysis results."""
    return {
        "total_objects": data.get("total_objects", 0),
        "full_grain_count": data.get("full_grain_count", 0),
        "broken_grain_count": data.get("broken_grain_count", 0),
        "black_dal": data.get("black_dal", 0),  # Added black_dal field
        "broken_percentages": {
            "25%": data.get("broken_percentages", {}).get("25%", 0),
            "50%": data.get("broken_percentages", {}).get("50%", 0),
            "75%": data.get("broken_percentages", {}).get("75%", 0),
        },
        "device_id": data.get("device_id", "unknown"),
        "timestamp": data.get("timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        "created_at": datetime.datetime.now(),
    }

def sync_data_to_mongodb(app_root_path):
    """
    Sync local data to MongoDB.
    Delete local data after successful sync.
    """
    # Try to connect to MongoDB
    client = connect_to_mongodb()
    if client is None:
        logger.warning("Could not connect to MongoDB. Skipping sync.")
        return False
    
    try:
        db = client[config.DB_NAME]
        
        # Define local storage directories
        local_storage_dir = os.path.join(app_root_path, 'local_storage')
        rice_storage = os.path.join(local_storage_dir, 'rice')
        dal_storage = os.path.join(local_storage_dir, 'dal')
        
        # Sync rice data
        rice_files = glob.glob(os.path.join(rice_storage, "*.json"))
        rice_collection = db[config.RICE_COLLECTION]
        
        for file_path in rice_files:
            try:
                with open(file_path, 'r') as f:
                    rice_data = json.load(f)
                    
                # Create MongoDB document
                rice_doc = create_rice_document(rice_data)
                
                # Insert into MongoDB
                result = rice_collection.insert_one(rice_doc)
                
                if result.acknowledged:
                    # Delete local file after successful insertion
                    os.remove(file_path)
                    logger.info(f"Synced rice data from {file_path} to MongoDB.")
                else:
                    logger.warning(f"MongoDB insert not acknowledged for {file_path}")
            except Exception as e:
                logger.error(f"Error syncing rice data from {file_path}: {str(e)}")
        
        # Sync dal data
        dal_files = glob.glob(os.path.join(dal_storage, "*.json"))
        dal_collection = db[config.DAL_COLLECTION]
        
        for file_path in dal_files:
            try:
                with open(file_path, 'r') as f:
                    dal_data = json.load(f)
                    
                # Create MongoDB document
                dal_doc = create_dal_document(dal_data)
                
                # Insert into MongoDB
                result = dal_collection.insert_one(dal_doc)
                
                if result.acknowledged:
                    # Delete local file after successful insertion
                    os.remove(file_path)
                    logger.info(f"Synced dal data from {file_path} to MongoDB.")
                else:
                    logger.warning(f"MongoDB insert not acknowledged for {file_path}")
            except Exception as e:
                logger.error(f"Error syncing dal data from {file_path}: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error during MongoDB sync: {str(e)}")
        
        # If there was a connection error, invalidate the client
        global _mongodb_client
        if _mongodb_client is not None:
            try:
                _mongodb_client.close()
            except:
                pass
            _mongodb_client = None
        
        return False

def run_sync_loop(app_root_path):
    """Run the main sync loop."""
    logger.info("=== MongoDB Sync Standalone Process Started ===")
    logger.info(f"Monitoring directory: {os.path.join(app_root_path, 'local_storage')}")
    
    # Try to sync immediately on startup, but don't block if it fails
    try:
        if check_internet_connection():
            logger.info("Internet available on startup - attempting initial sync")
            sync_data_to_mongodb(app_root_path)
        else:
            logger.info("No internet connection on startup - skipping initial sync")
    except Exception as e:
        logger.error(f"Error during initial sync: {str(e)}")
    
    # Main loop
    while _running:
        try:
            # Check for files to sync
            rice_files = glob.glob(os.path.join(app_root_path, 'local_storage', 'rice', "*.json"))
            dal_files = glob.glob(os.path.join(app_root_path, 'local_storage', 'dal', "*.json"))
            
            if rice_files or dal_files:
                total_files = len(rice_files) + len(dal_files)
                logger.info(f"Found {total_files} files to sync ({len(rice_files)} rice, {len(dal_files)} dal)")
                
                # Check internet before trying to sync
                if check_internet_connection():
                    logger.info("Internet connection available, attempting sync")
                    sync_result = sync_data_to_mongodb(app_root_path)
                    if sync_result:
                        logger.info("MongoDB sync completed successfully")
                    else:
                        logger.warning("MongoDB sync attempt failed")
                else:
                    logger.info("No internet connection, skipping MongoDB sync")
            else:
                logger.debug("No files to sync")
                
            # Wait before next check
            time.sleep(config.SYNC_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Error in sync loop: {str(e)}")
            time.sleep(60)  # Shorter retry interval after error

if __name__ == "__main__":
    if len(sys.argv) > 1:
        app_root_path = sys.argv[1]
    else:
        app_root_path = os.path.dirname(os.path.abspath(__file__))
    
    run_sync_loop(app_root_path)