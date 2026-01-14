import pymongo
import os
import json
import glob
import time
import socket
import logging
import datetime
from pymongo.errors import ConnectionFailure
from mongodb_models import create_rice_document, create_dal_document
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='mongodb_sync.log',
    filemode='a'
)
logger = logging.getLogger('mongodb_sync')

# Global MongoDB client
_mongodb_client = None
_last_connection_time = None

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

def attempt_sync_to_mongodb(app_root_path):
    """Attempt to sync data to MongoDB if internet is available."""
    if not check_internet_connection():
        logger.info("No internet connection available. Skipping MongoDB sync.")
        return False
    
    return sync_data_to_mongodb(app_root_path)

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

def schedule_sync_task(app_root_path):
    """Schedule a periodic sync task."""
    logger.info("Starting MongoDB sync scheduler")
    
    # Add initial delay to ensure app starts without being blocked
    logger.info("Delaying first MongoDB connection attempt to ensure app startup")
    time.sleep(10)  # 10 second initial delay
    
    while True:
        try:
            # Check for internet before attempting connection
            if check_internet_connection():
                logger.info("Internet connection available, attempting sync")
                sync_result = sync_data_to_mongodb(app_root_path)
                if sync_result:
                    logger.info("MongoDB sync completed successfully")
                else:
                    logger.warning("MongoDB sync attempt failed")
            else:
                logger.info("No internet connection, skipping MongoDB sync")
                
            # Wait before next sync attempt
            time.sleep(config.SYNC_INTERVAL_SECONDS)
        except Exception as e:
            logger.error(f"Error in sync task: {str(e)}")
            time.sleep(60)  # Shorter retry interval after error