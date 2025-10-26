# pages/3_Prediction.py
"""
Prediction-only page:
- Loads a random preselected image on load (from preselected_images/)
- Allows uploading an image (preserve original filename)
- Detects filter (131/193) from filename with an override
- Runs prediction models (prediction_v1 and prediction_v2/pred3)
- Allows selecting which prediction models to run and shows probabilities/predicted class
"""

import streamlit as st
from pathlib import Path
import tempfile
from typing import Optional
import traceback

import utils
from config import PRESELECTED_IMAGES_FOLDER

def save_uploaded_to_temp(uploaded) -> Path:
    tmpdir = tempfile.mkdtemp()
    dest = Path(tmpdir) / Path(uploaded.name).name
    with open(dest, "wb") as f:
        f.write(uploaded.read())
    return dest

st.set_page_config(page_title="Prediction — Solar Flare App", layout="wide")
st.title("Prediction — Run prediction models")

with st.sidebar:
    st.header("Controls")
    use_preselected = st.checkbox("Use random preselected image on load", value=True)
    st.caption(f"Preselected folder: {PRESELECTED_IMAGES_FOLDER}")
    st.markdown("Select prediction models to run:")
    pred_choices = st.multiselect("Prediction models", options=["prediction_v1", "prediction_v2 (pred3)"], default=["prediction_v1", "prediction_v2 (pred3)"])

# session_state for current image
if "prediction_current_image" not in st.session_state:
    try:
        st.session_state.prediction_current_image = utils.get_random_preselected_image()
    except Exception:
        st.session_state.prediction_current_image = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input image")
    uploaded = st.file_uploader("Upload image or FITS", type=None)
    refresh = st.button("Refresh random preselected image")

    if refresh:
        try:
            st.session_state.prediction_current_image = utils.get_random_preselected_image()
        except Exception as e:
            st.error(f"Could not load a preselected image: {e}")

    chosen_path: Optional[Path] = None
    if uploaded is not None:
        try:
            chosen_path = save_uploaded_to_temp(uploaded)
            st.session_state.prediction_current_image = chosen_path
        except Exception as e:
            st.error(f"Could not save uploaded file: {e}")

    if chosen_path is None:
        if st.session_state.prediction_current_image:
            chosen_path = Path(st.session_state.prediction_current_image)
        else:
            st.info("No image available. Upload or add files to preselected_images/")

    if chosen_path:
        st.write(f"Selected: `{chosen_path.name}`")
        try:
            preview = utils.prepare_display_image(chosen_path)
            st.image(preview, use_column_width=True)
        except Exception as e:
            st.write("Preview not available:", e)

    detected_filter = None
    if chosen_path:
        try:
            detected_filter = utils.parse_filter_from_filename(str(chosen_path.name))
            st.success(f"Auto-detected filter: {detected_filter}")
        except Exception:
            st.warning("Could not auto-detect filter (131/193) from filename. Please select.")

    chosen_override = st.selectbox("Filter (auto / override)", options=["auto", "131", "193"])
    filter_code = detected_filter if chosen_override == "auto" else chosen_override

    st.markdown("---")
    run_btn = st.button("Run prediction")

with col2:
    st.subheader("Results")
    if not chosen_path:
        st.info("Select or upload an image to run prediction.")
    else:
        st.markdown(f"**File:** `{chosen_path.name}`")
        st.markdown(f"**Filter (in use):** `{filter_code}`" if filter_code else "**Filter (in use):** _unknown_")
    out_container = st.container()

def show_prediction_results(results, selected_labels):
    # Filter results by label mapping
    label_map = {
        "prediction_v1": "model_prediction",
        "prediction_v2 (pred3)": "pred3",
    }
    filtered = []
    for r in results:
        # check which template this result corresponds to by model filename
        name = Path(r.get("model_path", "")).name if r.get("model_path") else r.get("model_label", "")
        if "model_prediction_" in name and "prediction_v1" in selected_labels:
            filtered.append(r)
        elif "pred3_" in name and "prediction_v2 (pred3)" in selected_labels:
            filtered.append(r)
        elif r.get("model_label") in ("prediction_v1", "pred3") and r.get("model_label") in selected_labels:
            filtered.append(r)
    if not filtered:
        out_container.info("No prediction models matched the selected filters (or models missing).")
        return

    # Display results
    for r in filtered:
        if "error" in r:
            out_container.error(f"{r.get('model_label', Path(r.get('model_path', '')).name)}: {r['error']}")
            continue
        label = r.get("model_label") or Path(r.get("model_path", "")).name
        pred = r.get("prediction")
        probs = r.get("probabilities")
        line = f"**{label}**"
        if pred is not None:
            line += f" — pred={pred}"
        if probs is not None:
            line += f" — probs={probs}"
        out_container.write(line)

if run_btn:
    if not chosen_path:
        st.error("No image selected.")
    elif not filter_code:
        st.error("Filter unknown. Please provide an image with 131/193 or select override.")
    else:
        with out_container:
            st.info("Running prediction models...")
            try:
                results = utils.run_prediction_models_for_image(chosen_path, filter_code=filter_code)
            except FileNotFoundError as e:
                st.error(e)
                results = []
            except Exception as e:
                st.exception(e)
                results = []
            # Show filtered results according to user selection
            show_prediction_results(results, pred_choices)
