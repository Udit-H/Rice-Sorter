import os
import shutil
import zipfile

def fix_model_file():
    model_path = "efficientnet_rice_final_inference.keras"
    backup_path = "efficientnet_rice_final_inference_backup_dir"
    
    # Check if it's a directory
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print(f"Found model directory at: {model_path}")
        print("This should be a file, not a directory. Attempting to convert...")
        
        # 1. Rename directory to backup
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        os.rename(model_path, backup_path)
        print(f"Renamed directory to: {backup_path}")
        
        # 2. Zip contents
        print("Zipping contents...")
        try:
            with zipfile.ZipFile(model_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(backup_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate arcname (relative path inside zip)
                        arcname = os.path.relpath(file_path, backup_path)
                        zipf.write(file_path, arcname)
                        print(f"  Added: {arcname}")
            
            print(f"\nSuccessfully created model file: {model_path}")
            print("You can verify this by checking if it's now a file.")
            
        except Exception as e:
            print(f"Error creating zip: {e}")
            # Try to restore
            if os.path.exists(model_path):
                os.remove(model_path)
            os.rename(backup_path, model_path)
            print("Restored original directory.")
            
    elif os.path.exists(model_path) and os.path.isfile(model_path):
        print(f"'{model_path}' is already a file. No action needed.")
    else:
        print(f"Path '{model_path}' not found.")

if __name__ == "__main__":
    fix_model_file()
