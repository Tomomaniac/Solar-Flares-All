# prediction_app.py
# Solar Flare Prediction App - v1 (Binary) and v2 (3-Class)

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import random
from prediction_utils import (
    preprocess_image, 
    apply_colormap, 
    load_prediction_models, 
    predict_flare_precursor
)
from prediction_utils_v2 import (
    load_prediction_models_v2,
    predict_flare_3class
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
        return None


# ====
# STREAMLIT APP
# ====

def main():
    st.set_page_config(page_title="Solar Flare Predictor", page_icon="🔮", layout="centered")
    
    st.title("🔮 Solar Flare Prediction App")
    st.markdown("Upload a solar image to predict flare activity")
    
    # Load models (cached)
    try:
        models_v1, feature_cols_v1 = load_models_v1_cached()
        models_v2, feature_cols_v2, metadata_v2 = load_models_v2_cached()
        st.success("✓ All prediction models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
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
            "🤖 Select Model Version",
            options=["Model v1 (Binary)", "Model v2 (3-Class)"],
            help="v1: PRECURSOR vs QUIET | v2: QUIET vs PRECURSOR vs FLARE"
        )
        model_num = "v1" if "v1" in model_version else "v2"
    
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
        
        # Predict button
        if st.button("🔮 Predict Flare Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing image..."):
                try:
                    if model_num == "v1":
                        # Binary prediction
                        result = predict_flare_precursor(
                            img_array, 
                            filter_num, 
                            models_v1, 
                            feature_cols_v1
                        )
                        display_results_v1(result, filter_num)
                    else:
                        # 3-class prediction
                        result = predict_flare_3class(
                            img_array,
                            filter_num,
                            models_v2,
                            feature_cols_v2
                        )
                        display_results_v2(result, filter_num, metadata_v2)
                    
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # Sidebar
    create_sidebar(metadata_v2)


@st.cache_resource
def load_models_v1_cached():
    """Cached v1 model loading"""
    return load_prediction_models()


@st.cache_resource
def load_models_v2_cached():
    """Cached v2 model loading"""
    return load_prediction_models_v2()


def display_results_v1(result, filter_num):
    """Display v1 binary prediction results"""
    prediction = result['prediction']
    probabilities = result['probabilities']
    features = result['features']
    
    st.markdown("---")
    st.subheader("📊 Prediction Results (Model v1)")
    
    # Metrics
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Filter", f"{filter_num} Å")
        st.metric("Raw Max Brightness", f"{features['raw_max']:.1f}")
    
    with res_col2:
        st.metric("Max Brightness (Norm)", f"{features['max_brightness']:.4f}")
        st.metric("Raw Mean Brightness", f"{features['raw_mean']:.1f}")
    
    st.markdown("---")
    
    # Prediction result
    if prediction == 1:
        st.warning("### ⚠️ FLARE PRECURSOR DETECTED")
        st.markdown("**This region shows signs of potential flare activity**")
    else:
        st.success("### ✅ QUIET REGION")
        st.markdown("**No significant flare precursor signatures detected**")
    
    # Confidence metrics
    confidence = probabilities[prediction] * 100
    precursor_prob = probabilities[1] * 100
    quiet_prob = probabilities[0] * 100
    
    st.markdown("---")
    st.subheader("📈 Confidence Metrics")
    
    conf_col1, conf_col2 = st.columns(2)
    
    with conf_col1:
        st.metric("Prediction Confidence", f"{confidence:.1f}%")
        st.metric("Precursor Probability", f"{precursor_prob:.1f}%")
    
    with conf_col2:
        st.metric("Quiet Probability", f"{quiet_prob:.1f}%")
    
    # Progress bar for precursor probability
    st.markdown("**Flare Precursor Risk Level:**")
    st.progress(precursor_prob / 100)
    
    # Risk interpretation
    if precursor_prob >= 80:
        st.error("🔴 **HIGH RISK** - Strong flare precursor signatures")
    elif precursor_prob >= 50:
        st.warning("🟡 **MODERATE RISK** - Some flare precursor indicators")
    else:
        st.success("🟢 **LOW RISK** - Minimal flare precursor activity")


def display_results_v2(result, filter_num, metadata):
    """Display v2 3-class prediction results"""
    prediction = result['prediction']
    probabilities = result['probabilities']
    features = result['features']
    class_name = result['class_name']
    
    st.markdown("---")
    st.subheader("📊 Prediction Results (Model v2 - 3-Class)")
    
    # Show model accuracy
    model_metadata = metadata[filter_num]
    st.caption(f"Model Accuracy: {model_metadata['performance']['accuracy']*100:.1f}%")
    
    # Metrics
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric("Filter", f"{filter_num} Å")
        st.metric("Raw Max Brightness", f"{features['raw_max']:.1f}")
    
    with res_col2:
        st.metric("Max Brightness (Norm)", f"{features['max_brightness']:.4f}")
        st.metric("Raw Mean Brightness", f"{features['raw_mean']:.1f}")
    
    st.markdown("---")
    
    # Prediction result
    if prediction == 2:  # FLARE
        st.error("### 🔴 ACTIVE FLARE DETECTED")
        st.markdown("**This region is currently experiencing a solar flare**")
    elif prediction == 1:  # PRECURSOR
        st.warning("### ⚠️ FLARE PRECURSOR DETECTED")
        st.markdown("**This region shows signs of potential flare activity**")
    else:  # QUIET
        st.success("### ✅ QUIET REGION")
        st.markdown("**No significant flare activity detected**")
    
    # Confidence metrics
    confidence = probabilities[prediction] * 100
    quiet_prob = probabilities[0] * 100
    precursor_prob = probabilities[1] * 100
    flare_prob = probabilities[2] * 100
    
    st.markdown("---")
    st.subheader("📈 Class Probabilities")
    
    # Three columns for three classes
    prob_col1, prob_col2, prob_col3 = st.columns(3)
    
    with prob_col1:
        st.metric("Quiet", f"{quiet_prob:.1f}%")
        st.progress(quiet_prob / 100)
    
    with prob_col2:
        st.metric("Precursor", f"{precursor_prob:.1f}%")
        st.progress(precursor_prob / 100)
    
    with prob_col3:
        st.metric("Flare", f"{flare_prob:.1f}%")
        st.progress(flare_prob / 100)
    
    st.markdown("---")
    st.metric("Prediction Confidence", f"{confidence:.1f}%")
    
    # Risk interpretation
    if prediction == 2:
        st.error("🔴 **ACTIVE FLARE** - Immediate solar activity")
    elif prediction == 1:
        if precursor_prob >= 80:
            st.error("🟠 **HIGH PRECURSOR RISK** - Strong flare indicators")
        elif precursor_prob >= 50:
            st.warning("🟡 **MODERATE PRECURSOR RISK** - Some flare indicators")
        else:
            st.info("🟡 **LOW PRECURSOR RISK** - Weak flare indicators")
    else:
        st.success("🟢 **QUIET** - Minimal flare activity")


def create_sidebar(metadata):
    """Create sidebar with information"""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Solar Flare Prediction App**
        
        This app predicts solar flare activity using two model versions.
        
        **Model v1 (Binary):**
        - **PRECURSOR (1)**: Images taken BEFORE a flare
        - **QUIET (0)**: Images from quiet regions
        
        **Model v2 (3-Class):**
        - **QUIET (0)**: Stable regions
        - **PRECURSOR (1)**: Pre-flare indicators
        - **FLARE (2)**: Active flare regions
        """)
        
        # Show v2 model performance
        st.markdown("**Model v2 Performance:**")
        st.markdown(f"- 131 Å: {metadata['131']['performance']['accuracy']*100:.1f}% accuracy")
        st.markdown(f"- 193 Å: {metadata['193']['performance']['accuracy']*100:.1f}% accuracy")
        
        st.markdown("""
        **Features:**
        Both models use 20 features including:
        - Normalized brightness statistics
        - Spatial distribution metrics
        - Raw brightness values
        - Concentration patterns
        
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
        4. Select model version (v1 or v2)
        5. Choose a colormap
        6. Click "Predict Flare Risk"
        7. View the prediction and confidence metrics
        """)
        
        st.markdown("---")
        st.markdown("**Model Comparison:**")
        st.markdown("""
        - **v1**: Simpler binary classification
        - **v2**: More detailed 3-class classification
        - v2 can distinguish active flares from precursors
        """)


if __name__ == "__main__":
    main()