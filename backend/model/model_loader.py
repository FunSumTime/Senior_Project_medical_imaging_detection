from tensorflow import keras
from config import MODEL_REGISTRY

_MODEL_CACHE = {}

# take the model from the frontend load  it  and put it in the datastructer

def load_model_by_key(model_key: str):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")

    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    model_path = MODEL_REGISTRY[model_key]["path"]

    model = keras.models.load_model(
        model_path,
        safe_mode=False,
    )

    _MODEL_CACHE[model_key] = model
    return model


def get_cam_layer(model_key: str) -> str:
    return MODEL_REGISTRY[model_key]["cam_layer"]