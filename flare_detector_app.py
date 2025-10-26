# flare_detector_simple.py
# Simple Solar Flare Detection App with Streamlit

import streamlit as st
import joblib # type: ignore
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

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
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    img_colored = cmap(img_normalized)
    
    # Convert to RGB (0-255)
    img_colored_rgb = (img_colored[:, :, :3] * 255).astype(np.uint8)
    
    return img_colored_rgb

# ============================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================

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

# ============================================================
# LOAD MODELS (CACHED)
# ============================================================

@st.cache_resource
def load_models():
    """Load all models (cached for performance)"""
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
    
#============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(page_title="Solar Flare Detector", page_icon="☀️", layout="centered")
    
    st.title("☀️ Solar Flare Detection App")
    st.markdown("Upload a solar image to detect flares using trained ML models")
    
    # Load models
    try:
        models, feature_cols_v1, feature_cols_v3 = load_models()
        st.success("✓ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter selector
        filter_type = st.selectbox(
            "🔬 Select Filter",
            options=["131 Å", "193 Å"],
            help="Choose the wavelength filter of your image"
        )
        filter_num = filter_type.split()[0]  # Extract "131" or "193"
    
    with col2:
        # Model selector
        model_version = st.selectbox(
            "🤖 Select Model",
            options=["Model v1", "Model v3"],
            help="v1: QUIET vs DURING | v3: (BEFORE+QUIET) vs DURING with raw brightness"
        )
        model_num = model_version.split()[1]  # Extract "v1" or "v3"
    
    with col3:
        # Colormap selector
        colormap = st.selectbox(
            "🎨 Colormap",
            options=["hot", "inferno", "plasma", "magma", "viridis", "gray"],
            help="Choose color scheme (hot = red for bright spots)"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Solar Image",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        help="Upload a solar image file"
    )
    
    if uploaded_file is not None:
        st.markdown("---")
        
        # Display uploaded image
        st.subheader("📷 Uploaded Image")
        
        # Load image and convert to array
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        
        # Normalize image
        img_normalized = preprocess_image(img_array)
        
        # Apply colormap
        img_display = apply_colormap(img_normalized, colormap)
        
        # Display the colored image
        st.image(img_display, caption=uploaded_file.name, use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # Extract features based on model version
                    if model_num == "v1":
                        features = extract_features_v1(img_normalized)
                        feature_cols = feature_cols_v1
                    else:  # v3
                        features = extract_features_v3(img_normalized, img_array)
                        feature_cols = feature_cols_v3
                    
                    # Prepare feature vector
                    X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
                    
                    # Select correct model
                    model_key = f"{model_num}_{filter_num}"
                    model = models[model_key]
                    
                    # Predict
                    prediction = model.predict(X)[0]
                    probabilities = model.predict_proba(X)[0]
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Results")
                    
                    # Create result columns
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.metric("Filter", f"{filter_num} Å")
                        st.metric("Model", model_version)
                    
                    with res_col2:
                        if model_num == "v3":
                            st.metric("Raw Max Brightness", f"{features['raw_max']:.1f}")
                            st.metric("Raw Mean Brightness", f"{features['raw_mean']:.1f}")
                        else:
                            st.metric("Max Brightness", f"{features['max_brightness']:.4f}")
                            st.metric("Mean Brightness", f"{features['mean_brightness']:.4f}")
                    
                    # Prediction result
                    st.markdown("---")
                    
                    if prediction == 1:
                        st.error("### 🔴 FLARE DETECTED")
                    else:
                        st.success("### 🟢 QUIET (No Flare)")
                    
                    # Confidence metrics
                    confidence = probabilities[prediction] * 100
                    flare_prob = probabilities[1] * 100
                    
                    st.metric("Confidence", f"{confidence:.1f}%")
                    st.metric("Flare Probability", f"{flare_prob:.1f}%")
                    
                    # Progress bar for flare probability
                    st.progress(flare_prob / 100)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Solar Flare Detection App**
        
        This app uses machine learning to detect solar flares in EUV images.
        
        **Models:**
        - **Model v1**: Trained on QUIET vs DURING images
          - Accuracy: ~95% (131 Å), ~91% (193 Å)
        
        - **Model v3**: Trained on (BEFORE+QUIET) vs DURING
          - Includes raw brightness features
          - Accuracy: ~91% (131 Å), ~89% (193 Å)
          - More robust for real-world scenarios
        
        **Filters:**
        - **131 Å**: Fe VIII/XXI (flare plasma)
        - **193 Å**: Fe XII/XXIV (corona/flare loops)
        
        **Colormaps:**
        - **hot**: Black → Red → Yellow → White
        - **inferno**: Black → Purple → Orange → Yellow
        - **plasma**: Purple → Pink → Orange → Yellow
        - **magma**: Black → Purple → Red → Yellow
        - **viridis**: Purple → Green → Yellow
        - **gray**: Grayscale
        """)
        
        st.markdown("---")
        st.markdown("**Instructions:**")
        st.markdown("""
        1. Select the filter (131 Å or 193 Å)
        2. Select the model (v1 or v3)
        3. Choose a colormap (hot = red for bright spots)
        4. Upload a solar image
        5. Click "Analyze Image"
        """)

if __name__ == "__main__":
    main()
