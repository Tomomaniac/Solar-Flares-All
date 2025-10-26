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


st.set_page_config(page_title="Detection — Run detection models (v3 → v1)", page_icon="🔎", layout="centered")
st.title("Detection — Run detection models (v3 → v1)")
st.markdown("Upload a solar image or use a preselected one to run detection models (v3 → v1)")

with st.sidebar:
    st.header("Controls")
    use_preselected = st.checkbox("Use random preselected image on load", value=True, key="side_use_preselected")
    st.caption(f"Preselected folder: {PRESELECTED_IMAGES_FOLDER}")
    st.markdown("Select which detection versions to run:")
    detection_versions = st.multiselect("Detection versions", options=["v3", "v1"], default=["v3", "v1"], key="side_detection_versions")

# session_state for current image
if "detection_current_image" not in st.session_state:
    try:
        st.session_state.detection_current_image = utils.get_random_preselected_image()
    except Exception:
        st.session_state.detection_current_image = None

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Input image")
    uploaded = st.file_uploader("📁 Upload image or FITS", type=None, key="uploader")
    refresh = st.button("🔄 Refresh random preselected image", key="refresh_preselected")

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

    detected_filter = None
    preview = None
    if chosen_path:
        st.write(f"Selected: `{chosen_path.name}`")
        try:
            detected_filter = utils.parse_filter_from_filename(str(chosen_path.name))
        except Exception:
            detected_filter = None

        # Three selectors: Filter, Model, Colormap (match app_with_all)
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            if detected_filter:
                default_index = 0 if detected_filter == "131 Å" else 1
                filter_type = st.selectbox(
                    "🔬 Select Filter",
                    options=["131 Å", "193 Å"],
                    index=default_index,
                    key="det_filter_select_main_inner",
                )
                st.caption(f"✓ Auto-detected: {detected_filter}")
            else:
                filter_type = st.selectbox(
                    "🔬 Select Filter",
                    options=["131 Å", "193 Å"],
                    key="det_filter_select_main_inner",
                )
            filter_num = filter_type.split()[0]

        with s_col2:
            model_version = st.selectbox(
                "🤖 Select Model",
                options=["Model v3", "Model v1"],
                key="det_model_select_main_inner",
            )
            model_num = "v3" if "v3" in model_version else "v1"

        with s_col3:
            colormap = st.selectbox(
                "🎨 Colormap",
                options=["hot", "inferno", "plasma", "magma", "viridis", "gray"],
                key="det_colormap_select_main_inner",
            )

        # Prepare preview (will display in right column for larger view)
        try:
            preview = utils.prepare_display_image(chosen_path, colormap=colormap)
        except Exception as e:
            preview = None

    chosen_override = st.selectbox("Filter (auto / override)", options=["auto", "131", "193"], key="filter_override_main")
    filter_code = detected_filter if chosen_override == "auto" else chosen_override

    st.markdown("---")
    run_btn = st.button("🔍 Run detection", key="run_detection")
    compare_versions_btn = st.button("Compare v3 vs v1 (side-by-side)", key="compare_versions")

with col2:
    st.subheader("Results")
    # Show larger preview at top of results column when available
    try:
        if preview is not None:
            st.image(preview, caption=f"Preview: {chosen_path.name}", use_column_width=True)
    except Exception:
        pass
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
                results = utils.run_detection_pipeline_for_image(chosen_path, filter_code=filter_code, detection_versions=detection_versions)
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
