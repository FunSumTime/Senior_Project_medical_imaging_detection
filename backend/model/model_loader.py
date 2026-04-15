from huggingface_hub import hf_hub_download
import tensorflow as tf

# name from my project
HF_REPO_ID = "FunSumTime/Medical_Senior_Project" 

_LOADED_MODELS = {}

def load_model_by_key(key):
    if key not in _LOADED_MODELS:
        # Determine the filename based on the key
        if key == "efficientnet_stage1":
            filename = "model1_unfrozen.keras"
        elif key == "efficientnet_stage2":
            filename = "model2_unfrozen.keras"
        elif key == "densenet_stage1":
            filename = "densenet_model1_unfrozen"
        elif key == "densenet_stage2":
            filename = "densenet_model2_unfrozen"

        else:
            raise ValueError(f"Unknown model key: {key}")

        print(f"Downloading/Loading {filename} from Hugging Face...")
        
        # This downloads the file to a hidden cache on Render (or uses the cached version if already downloaded)
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
        
        # Load the model into memory
        _LOADED_MODELS[key] = tf.keras.models.load_model(model_path)
        
    return _LOADED_MODELS[key]

from config import MODEL_REGISTRY

# ... (Your existing Hugging Face load_model_by_key code) ...

# ADD THIS MISSING FUNCTION AT THE BOTTOM
def get_cam_layer(key):
    """Fetches the target CAM layer for Grad-CAM from the registry."""
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key].get("cam_layer", "feat_maps")
    
    # Fallback just in case
    return "feat_maps"