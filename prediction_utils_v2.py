# prediction_utils_v2.py
# Solar Flare Prediction Utilities - Version 2 (3-Class)

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
# FEATURE EXTRACTION FUNCTION
# ====

def extract_features_v2(img_normalized, img_raw):
    """Extract features for v2 3-class prediction models"""
    features = {}
    
    # Normalized brightness features
    features['max_brightness'] = np.max(img_normalized)
    features['mean_brightness'] = np.mean(img_normalized)
    features['std_brightness'] = np.std(img_normalized)
    
    # Spatial features (using normalized image)
    threshold_90 = 0.9 * features['max_brightness']
    bright_pixels_90 = img_normalized >= threshold_90
    bright_coords = np.argwhere(bright_pixels_90)
    
    if len(bright_coords) > 0:
        # Centroid
        centroid_y = np.mean(bright_coords[:, 0])
        centroid_x = np.mean(bright_coords[:, 1])
        features['centroid_y'] = centroid_y
        features['centroid_x'] = centroid_x
        
        # Distances from centroid
        distances_from_centroid = np.sqrt(
            (bright_coords[:, 0] - centroid_y)**2 + 
            (bright_coords[:, 1] - centroid_x)**2
        )
        features['max_distance_from_centroid'] = np.max(distances_from_centroid)
        features['mean_distance_from_centroid'] = np.mean(distances_from_centroid)
        features['std_distance_from_centroid'] = np.std(distances_from_centroid)
        
        # Brightest point distance
        brightest_idx = np.unravel_index(np.argmax(img_normalized), img_normalized.shape)
        distances_from_brightest = np.sqrt(
            (bright_coords[:, 0] - brightest_idx[0])**2 + 
            (bright_coords[:, 1] - brightest_idx[1])**2
        )
        features['max_distance_from_brightest'] = np.max(distances_from_brightest)
        
        # Spatial spread
        features['bright_pixels_std_y'] = np.std(bright_coords[:, 0])
        features['bright_pixels_std_x'] = np.std(bright_coords[:, 1])
        features['spatial_spread'] = np.sqrt(
            features['bright_pixels_std_y']**2 + 
            features['bright_pixels_std_x']**2
        )
        
        # Compactness
        area = len(bright_coords)
        perimeter_approx = 2 * np.pi * features['spatial_spread']
        features['compactness'] = (perimeter_approx**2) / (4 * np.pi * area) if area > 0 else 0
        
        # Aspect ratio
        y_range = np.ptp(bright_coords[:, 0])
        x_range = np.ptp(bright_coords[:, 1])
        features['aspect_ratio'] = max(y_range, x_range) / (min(y_range, x_range) + 1e-6)
        
        # Concentration features
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
    
    # Raw brightness features
    features['raw_max'] = np.max(img_raw)
    features['raw_mean'] = np.mean(img_raw)
    
    return features


# ====
# MODEL LOADING
# ====

def load_prediction_models_v2():
    """Load v2 3-class prediction models and feature columns"""
    # Load model dictionaries
    model_dict_131 = joblib.load("./models_prediction/solar_flare_3class_131A_model.pkl")
    model_dict_193 = joblib.load("./models_prediction/solar_flare_3class_193A_model.pkl")
    
    # Extract the actual models from the dictionaries
    models = {
        '131': model_dict_131['model'],
        '193': model_dict_193['model']
    }
    
    # Get feature columns from metadata
    feature_cols = model_dict_131['metadata']['features']
    
    # Also return metadata for reference
    metadata = {
        '131': model_dict_131['metadata'],
        '193': model_dict_193['metadata']
    }
    
    return models, feature_cols, metadata


# ====
# PREDICTION FUNCTION
# ====

def predict_flare_3class(img_array, filter_num, models, feature_cols):
    """
    Predict 3-class: QUIET (0), PRECURSOR (1), or FLARE (2)
    
    Args:
        img_array: Raw image array
        filter_num: "131" or "193"
        models: Dictionary of loaded models
        feature_cols: Feature columns list
    
    Returns:
        dict with keys: prediction, probabilities, features, class_name
    """
    # Handle NaN and Inf values
    img_array = img_array.astype(np.float32)
    img_array[np.isnan(img_array)] = 0
    img_array[np.isinf(img_array)] = 0
    
    # Normalize image
    img_normalized = preprocess_image(img_array)
    
    # Extract features
    features = extract_features_v2(img_normalized, img_array)
    
    # Prepare feature vector
    X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
    
    # Select correct model
    model = models[filter_num]
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # Map prediction to class name
    class_names = {0: 'QUIET', 1: 'PRECURSOR', 2: 'FLARE'}
    class_name = class_names[prediction]
    
    return {
        'prediction': prediction,
        'probabilities': probabilities,
        'features': features,
        'class_name': class_name
    }