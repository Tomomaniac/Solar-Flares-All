# detection_utils.py
# Solar Flare Detection Utilities

import numpy as np
import matplotlib.cm as cm
import joblib
import pickle

# ====
# PREPROCESSING FUNCTIONS
# ====

def preprocess_image(img_array):
    """Normalize image to [0, 1] range"""
    img_min = np.min(img_array)
    img_max = np.max(img_array)
    
    if img_max - img_min == 0:
        return np.zeros_like(img_array, dtype=float)
    
    img_normalized = (img_array - img_min) / (img_max - img_min)
    return img_normalized


def apply_colormap(img_normalized, colormap='hot'):
    """Apply colormap to normalized image (hot spots show red)"""
    cmap = cm.get_cmap(colormap)
    img_colored = cmap(img_normalized)
    img_colored_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)
    return img_colored_rgb


# ====
# FEATURE EXTRACTION FUNCTIONS
# ====

def extract_features_v1(img_normalized):
    """Extract features for Model v1 (normalized only)"""
    features = {}
    
    features['max_brightness'] = np.max(img_normalized)
    features['mean_brightness'] = np.mean(img_normalized)
    features['std_brightness'] = np.std(img_normalized)
    
    threshold_90 = 0.9 * features['max_brightness']
    bright_pixels_90 = img_normalized >= threshold_90
    bright_coords = np.argwhere(bright_pixels_90)
    
    if len(bright_coords) > 0:
        centroid_y = np.mean(bright_coords[:, 0])
        centroid_x = np.mean(bright_coords[:, 1])
        features['centroid_y'] = centroid_y
        features['centroid_x'] = centroid_x
        
        distances_from_centroid = np.sqrt(
            (bright_coords[:, 0] - centroid_y)**2 + 
            (bright_coords[:, 1] - centroid_x)**2
        )
        features['max_distance_from_centroid'] = np.max(distances_from_centroid)
        features['mean_distance_from_centroid'] = np.mean(distances_from_centroid)
        features['std_distance_from_centroid'] = np.std(distances_from_centroid)
        
        brightest_idx = np.unravel_index(np.argmax(img_normalized), img_normalized.shape)
        distances_from_brightest = np.sqrt(
            (bright_coords[:, 0] - brightest_idx[0])**2 + 
            (bright_coords[:, 1] - brightest_idx[1])**2
        )
        features['max_distance_from_brightest'] = np.max(distances_from_brightest)
        
        features['bright_pixels_std_y'] = np.std(bright_coords[:, 0])
        features['bright_pixels_std_x'] = np.std(bright_coords[:, 1])
        features['spatial_spread'] = np.sqrt(
            features['bright_pixels_std_y']**2 + 
            features['bright_pixels_std_x']**2
        )
        
        area = len(bright_coords)
        perimeter_approx = 2 * np.pi * features['spatial_spread']
        features['compactness'] = (perimeter_approx**2) / (4 * np.pi * area) if area > 0 else 0
        
        y_range = np.ptp(bright_coords[:, 0])
        x_range = np.ptp(bright_coords[:, 1])
        features['aspect_ratio'] = max(y_range, x_range) / (min(y_range, x_range) + 1e-6)
        
        for pct in [90, 80, 70]:
            threshold = (pct / 100) * features['max_brightness']
            bright_pixels = img_normalized >= threshold
            features[f'concentration_{pct}'] = np.sum(bright_pixels) / (img_normalized.shape[0] * img_normalized.shape[1])
        
        features['num_bright_pixels_90'] = len(bright_coords)
    else:
        features['centroid_y'] = img_normalized.shape[0] / 2
        features['centroid_x'] = img_normalized.shape[1] / 2
        features['max_distance_from_centroid'] = 0
        features['mean_distance_from_centroid'] = 0
        features['std_distance_from_centroid'] = 0
        features['max_distance_from_brightest'] = 0
        features['bright_pixels_std_y'] = 0
        features['bright_pixels_std_x'] = 0
        features['spatial_spread'] = 0
        features['compactness'] = 0
        features['aspect_ratio'] = 1
        features['concentration_90'] = 0
        features['concentration_80'] = 0
        features['concentration_70'] = 0
        features['num_bright_pixels_90'] = 0
    
    return features


def extract_features_v3(img_normalized, img_raw):
    """Extract features for Model v3 (normalized + raw)"""
    features = extract_features_v1(img_normalized)
    features['raw_max'] = np.max(img_raw)
    features['raw_mean'] = np.mean(img_raw)
    return features


# ====
# MODEL LOADING
# ====

def load_models():
    """Load all models and feature columns"""
    models = {
        'v1_131': joblib.load("models/model_v1_131.pkl"),
        'v1_193': joblib.load("models/model_v1_193.pkl"),
        'v3_131': joblib.load("models/model_v3_131.pkl"),
        'v3_193': joblib.load("models/model_v3_193.pkl")
    }
    
    with open("models/feature_columns_v1.pkl", 'rb') as f:
        feature_cols_v1 = pickle.load(f)
    
    with open("models/feature_columns_v3.pkl", 'rb') as f:
        feature_cols_v3 = pickle.load(f)
    
    return models, feature_cols_v1, feature_cols_v3


# ====
# PREDICTION FUNCTION
# ====

def predict_flare(img_array, model_version, filter_num, models, feature_cols_v1, feature_cols_v3):
    """
    Predict flare from image array
    
    Args:
        img_array: Raw image array
        model_version: "v1" or "v3"
        filter_num: "131" or "193"
        models: Dictionary of loaded models
        feature_cols_v1: Feature columns for v1
        feature_cols_v3: Feature columns for v3
    
    Returns:
        dict with keys: prediction, probabilities, features
    """
    # Normalize image
    img_normalized = preprocess_image(img_array)
    
    # Extract features based on model version
    if model_version == "v1":
        features = extract_features_v1(img_normalized)
        feature_cols = feature_cols_v1
    else:  # v3
        features = extract_features_v3(img_normalized, img_array)
        feature_cols = feature_cols_v3
    
    # Prepare feature vector
    X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
    
    # Select correct model
    model_key = f"{model_version}_{filter_num}"
    model = models[model_key]
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'features': features
    }