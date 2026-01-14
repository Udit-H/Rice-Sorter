from datetime import datetime

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
        "timestamp": data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        "created_at": datetime.now(),
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
        "timestamp": data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S_%f")),
        "created_at": datetime.now(),
    }