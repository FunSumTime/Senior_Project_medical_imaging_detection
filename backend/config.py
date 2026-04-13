import os
# My models are  trained on colab so  i will just load them

# -------------------------
# Model registry
# keys are what frontend sends
# values are local model folders/files
# to get what model to use
# -------------------------
MODEL_REGISTRY = {
    "efficientnet_stage1": {
        "path": os.path.join("saved_models", "model1_unfrozen.keras"),
        "cam_layer": "feat_maps",
        "type": "stage1",
    },
    "efficientnet_stage2": {
        "path": os.path.join("saved_models", "model2_unfrozen.keras"),
        "cam_layer": "feat_maps",
        "type": "stage2",
    },
    "densenet_stage1": {
        "path": os.path.join("saved_models", "densenet_model1_unfrozen.keras"),
        "cam_layer": "feat_maps",
        "type": "stage1",
    },
    "densenet_stage2": {
        "path": os.path.join("saved_models", "densenet_model2_unfrozen.keras"),
        "cam_layer": "feat_maps",
        "type": "stage2",
    },
}

DEFAULT_STAGE1_MODEL = "densenet_stage1"
DEFAULT_STAGE2_MODEL = "densenet_stage2"

IMG_SIZE = (224, 224)
GRID_ROWS = 4
GRID_COLS = 5
TOP_K_CELLS = 4
PAD_RATIO = 0.12

DEFAULT_THRESHOLD = 0.50

CLASS_NAMES = ["normal", "pneumonia"]
PNEUMONIA_CLASS_INDEX = 1

# need to just add more vars for differnt models  that can do  differnt classficationns
