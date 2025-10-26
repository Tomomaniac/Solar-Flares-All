# config.py
# Central configuration for model filenames and mapping to filters (131 / 193).
# Update paths/names if your actual filenames differ.

from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# Feature columns files
FEATURE_COLUMNS_V3 = MODELS_DIR / "feature_columns_v3.pkl"
FEATURE_COLUMNS_V1 = MODELS_DIR / "feature_columns_v1.pkl"
FEATURE_COLUMNS_PRED = MODELS_DIR / "feature_columns_prediction.pkl"

# Detection models: order matters -> v3 should be checked before v1
# We'll format strings with the filter code (e.g., "131" or "193")
DETECTION_MODEL_TEMPLATES = {
    "v3": MODELS_DIR / "model_v3_{}.pkl",
    "v1": MODELS_DIR / "model_v1_{}.pkl",
}
# Order to evaluate detection models:
DETECTION_ORDER = ["v3", "v1"]

# Prediction models:
PREDICTION_MODEL_TEMPLATES = {
    "prediction_v1": MODELS_DIR / "model_prediction_{}.pkl",
    "prediction_v2": MODELS_DIR / "pred3_{}.pkl",  # your pred3 naming (v2)
}

# Preselected images folder (app will pick a random file on load)
PRESELECTED_IMAGES_FOLDER = BASE_DIR / "preselected_images"

# Allowed image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".fits"}
