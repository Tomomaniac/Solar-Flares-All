# pages/2_Detection.py
"""
Detection-only page:
- Loads a random preselected image on load (from preselected_images/)
- Allows uploading an image (preserve original filename so filter detection works)
- Detects filter (131/193) from filename with an override
- Runs detection models (v3 -> v1) and shows results
- Option to run only specific detection versions
"""

import streamlit as st
from pathlib import Path
import tempfile
from typing import Optional
import traceback

import utils
from config import PRESELECTED_IMAGES_FOLDER

def save_uploaded_to_temp(uploaded) -> Path:
    """
    Save st.uploaded_file to a temporary file preserving the original filename
    so parse_filter_from_filename can detect filter tokens.
    """
    tmpdir = tempfile.mkdtemp()
    dest = Path(tmpdir) / Path(uploaded.name).name
    with open(dest, "wb") as f:
        f.write(uploaded.read())
    return dest


st.set_page_config(page_title="Detection — Solar Flare App", layout="wide")
st.title("Detection — Run detection models (v3 → v1)")

with st.sidebar:
    st.header("Controls")
    use_preselected = st.checkbox("Use random preselected image on load", value=True)
    st.caption(f"Preselected folder: {PRESELECTED_IMAGES_FOLDER}")
    st.markdown("Select which detection versions to run:")
    detection_versions = st.multiselect("Detection versions", options=["v3", "v1"], default=["v3", "v1"])

# session_state for current image
if "detection_current_image" not in st.session_state:
    try:
        st.session_state.detection_current_image = utils.get_random_preselected_image()
    except Exception:
        st.session_state.detection_current_image = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input image")
    uploaded = st.file_uploader("Upload image or FITS", type=None)
    refresh = st.button("Refresh random preselected image")

    if refresh:
        try:
            st.session_state.detection_current_image = utils.get_random_preselected_image()
        except Exception as e:
            st.error(f"Could not load a preselected image: {e}")

    chosen_path: Optional[Path] = None
    if uploaded is not None:
        try:
            chosen_path = save_uploaded_to_temp(uploaded)
            st.session_state.detection_current_image = chosen_path
        except Exception as e:
            st.error(f"Could not save uploaded file: {e}")

    if chosen_path is None:
        if st.session_state.detection_current_image:
            chosen_path = Path(st.session_state.detection_current_image)
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
    run_btn = st.button("Run detection")
    compare_versions_btn = st.button("Compare v3 vs v1 (side-by-side)")

with col2:
    st.subheader("Results")
    if not chosen_path:
        st.info("Select or upload an image to run detection.")
    else:
        st.markdown(f"**File:** `{chosen_path.name}`")
        st.markdown(f"**Filter (in use):** `{filter_code}`" if filter_code else "**Filter (in use):** _unknown_")
    out_container = st.container()

# Inference functions
def show_detection_results(results):
    for r in results:
        if "error" in r:
            out_container.error(f"{Path(r.get('model_path', '')).name}: {r['error']}")
        else:
            model_name = Path(r.get("model_path", "")).name
            pred = r.get("prediction")
            prob = r.get("probability")
            raw = r.get("raw")
            line = f"**{model_name}**"
            if pred is not None:
                line += f" — pred={pred}"
            if prob is not None:
                line += f" — prob={prob:.3f}"
            if raw is not None:
                line += f" — raw={raw}"
            out_container.write(line)

if run_btn:
    if not chosen_path:
        st.error("No image selected.")
    elif not filter_code:
        st.error("Filter unknown. Please provide an image with 131/193 or select override.")
    else:
        with out_container:
            st.info("Running detection models...")
            try:
                results = utils.run_detection_pipeline_for_image(chosen_path, filter_code=filter_code)
            except FileNotFoundError as e:
                st.error(e)
                results = []
            except Exception as e:
                st.exception(e)
                results = []
            show_detection_results(results)
            # summary
            positive = any((r.get("prediction") == 1) or (r.get("probability") and r.get("probability") >= 0.5) for r in results if "error" not in r)
            if positive:
                st.success("Overall detection: POSITIVE")
            else:
                st.warning("Overall detection: NEGATIVE")

if compare_versions_btn:
    if not chosen_path:
        st.error("No image selected.")
    elif not filter_code:
        st.error("Filter unknown. Please provide an image with 131/193 or select override.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### v3 results")
            try:
                res_v3 = utils.run_detection_pipeline_for_image(chosen_path, filter_code=filter_code)
                # filter only v3 entries
                for r in res_v3:
                    if "v3" in Path(r.get("model_path", "")).name:
                        st.write(r)
            except Exception as e:
                st.error(f"v3 error: {e}")
        with col_b:
            st.markdown("#### v1 results")
            try:
                res_v1 = utils.run_detection_pipeline_for_image(chosen_path, filter_code=filter_code)
                for r in res_v1:
                    if "v1" in Path(r.get("model_path", "")).name:
                        st.write(r)
            except Exception as e:
                st.error(f"v1 error: {e}")
