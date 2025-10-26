# app.py
# Solar Flare Detection App - Main Interface

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import random
from detection_utils import (
    preprocess_image, 
    apply_colormap, 
    load_models, 
    predict_flare
)

# ====
# HELPER FUNCTIONS
# ====

def get_random_preselected_image():
    """Get a random image from preselected_images folder"""
    preselected_folder = Path("preselected_images")
    
    if not preselected_folder.exists():
        return None
    
    # Get all image files
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(preselected_folder.glob(f'*{ext}')))
        image_files.extend(list(preselected_folder.glob(f'*{ext.upper()}')))
    
    if not image_files:
        return None
    
    return random.choice(image_files)


def detect_filter_from_filename(filename):
    """Detect filter type (131 or 193) from filename"""
    filename_lower = filename.lower()
    
    if "0131" in filename_lower:
        return "131 Å"
    elif "0193" in filename_lower:
        return "193 Å"
    else:
        return None  # No filter detected


# ====
# STREAMLIT APP
# ====

def main():
    st.set_page_config(page_title="Solar Flare Detector", page_icon="☀️", layout="centered")
    
    st.title("☀️ Solar Flare Detection App")
    st.markdown("Upload a solar image to detect flares using trained ML models")
    
    # Load models (cached)
    try:
        models, feature_cols_v1, feature_cols_v3 = load_models_cached()
        st.success("✓ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()
    
    st.markdown("---")
    
    # Initialize session state for preselected image
    if 'preselected_image_loaded' not in st.session_state:
        st.session_state.preselected_image_loaded = False
        st.session_state.preselected_image_path = None
    
    # Load random preselected image on first run
    if not st.session_state.preselected_image_loaded:
        random_image = get_random_preselected_image()
        if random_image:
            st.session_state.preselected_image_path = random_image
        st.session_state.preselected_image_loaded = True
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Solar Image",
        type=['tif', 'tiff', 'png', 'jpg', 'jpeg'],
        help="Upload a solar image file"
    )
    
    # Determine which image to use
    if uploaded_file is not None:
        # User uploaded a file
        img = Image.open(uploaded_file)
        img_array = np.array(img)
        image_name = uploaded_file.name
        is_preselected = False
    elif st.session_state.preselected_image_path is not None:
        # Use preselected image
        img = Image.open(st.session_state.preselected_image_path)
        img_array = np.array(img)
        image_name = st.session_state.preselected_image_path.name
        is_preselected = True
        st.info(f"🎲 Using preselected image: **{image_name}**")
    else:
        img = None
        img_array = None
        image_name = None
        is_preselected = False
    
    # Auto-detect filter from filename
    detected_filter = None
    if image_name:
        detected_filter = detect_filter_from_filename(image_name)
    
    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Set default filter based on detection
        if detected_filter:
            default_index = 0 if detected_filter == "131 Å" else 1
            filter_type = st.selectbox(
                "🔬 Select Filter",
                options=["131 Å", "193 Å"],
                index=default_index,
                help="Choose the wavelength filter of your image (auto-detected from filename)"
            )
            st.caption(f"✓ Auto-detected: {detected_filter}")
        else:
            filter_type = st.selectbox(
                "🔬 Select Filter",
                options=["131 Å", "193 Å"],
                help="Choose the wavelength filter of your image"
            )
        filter_num = filter_type.split()[0]
    
    with col2:
        model_version = st.selectbox(
            "🤖 Select Model",
            options=["Model v1", "Model v3"],
            help="v1: QUIET vs DURING | v3: (BEFORE+QUIET) vs DURING with raw brightness"
        )
        model_num = model_version.split()[1]
    
    with col3:
        colormap = st.selectbox(
            "🎨 Colormap",
            options=["hot", "inferno", "plasma", "magma", "viridis", "gray"],
            help="Choose color scheme (hot = red for bright spots)"
        )
    
    # Display and analyze image if available
    if img is not None:
        st.markdown("---")
        st.subheader("📷 Current Image")
        
        # Display image
        img_normalized = preprocess_image(img_array)
        img_display = apply_colormap(img_normalized, colormap)
        
        st.image(img_display, caption=image_name, use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    # Predict
                    result = predict_flare(
                        img_array, 
                        model_num, 
                        filter_num, 
                        models, 
                        feature_cols_v1, 
                        feature_cols_v3
                    )
                    
                    # Display results
                    display_results(result, model_num, filter_num, model_version)
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Sidebar
    create_sidebar()


@st.cache_resource
def load_models_cached():
    """Cached model loading"""
    return load_models()


def display_results(result, model_num, filter_num, model_version):
    """Display prediction results"""
    prediction = result['prediction']
    probabilities = result['probabilities']
    features = result['features']
    
    st.markdown("---")
    st.subheader("📊 Results")
    
    # Metrics
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
    
    st.markdown("---")
    
    # Prediction
    if prediction == 1:
        st.error("### 🔴 FLARE DETECTED")
    else:
        st.success("### 🟢 QUIET (No Flare)")
    
    # Confidence
    confidence = probabilities[prediction] * 100
    flare_prob = probabilities[1] * 100
    
    st.metric("Confidence", f"{confidence:.1f}%")
    st.metric("Flare Probability", f"{flare_prob:.1f}%")
    st.progress(flare_prob / 100)


def create_sidebar():
    """Create sidebar with information"""
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
        1. A random preselected image loads automatically
        2. Upload your own image to replace it (optional)
        3. Filter is auto-detected from filename (0131/0193)
        4. Select the model (v1 or v3)
        5. Choose a colormap
        6. Click "Analyze Image"
        """)


if __name__ == "__main__":
    main()