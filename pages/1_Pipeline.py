"""
pages/1_Pipeline.py

Streamlit "Pipeline" page:
- Loads a random preselected image on load (from preselected_images/)
- Allows uploading an image (or use preselected)
- Detects filter (131/193) from filename (fallback: user-selectable)
- Runs detection models in order (v3 then v1). If detection is negative, runs prediction models.
- Provides a "Compare side-by-side" mode that runs detection and prediction simultaneously.
- Refresh button to load a new random preselected image.
"""

import streamlit as st
from pathlib import Path
import tempfile
import traceback
from typing import Optional

import utils
from config import PRESELECTED_IMAGES_FOLDER

# pages/1_Pipeline.py — replace save_uploaded_to_temp with this
def save_uploaded_to_temp(uploaded) -> Path:
    """
    Save st.uploaded_file to a temporary file preserving the original filename.
    Returns a Path to the saved file.
    """
    # create a temporary directory and save using the original filename so parse_filter_from_filename can see it
    tmpdir = tempfile.mkdtemp()
    dest = Path(tmpdir) / Path(uploaded.name).name
    with open(dest, "wb") as f:
        f.write(uploaded.read())
    return dest


def detect_positive_from_results(results) -> bool:
    """
    Determine whether detection pipeline overall is positive.
    Rules:
      - If any model returns prediction == 1 -> positive
      - Else if any model returns probability >= 0.5 -> positive
      - Else negative
    """
    for r in results:
        if r.get("prediction") is not None:
            try:
                if int(r["prediction"]) == 1:
                    return True
            except Exception:
                pass
        prob = r.get("probability")
        if prob is not None:
            try:
                if float(prob) >= 0.5:
                    return True
            except Exception:
                pass
    return False


def format_detection_result(r):
    """Return a string summary for a detection model result dict."""
    if "error" in r:
        return f"ERROR: {r['error']}"
    model_name = r.get("model_path") and Path(r["model_path"]).name or "model"
    pred = r.get("prediction")
    prob = r.get("probability")
    raw = r.get("raw")
    parts = [f"{model_name}"]
    if pred is not None:
        parts.append(f"pred={pred}")
    if prob is not None:
        parts.append(f"prob={prob:.3f}")
    if raw is not None:
        parts.append(f"raw={raw}")
    return " | ".join(parts)


def format_prediction_result(r):
    """Return a string summary for a prediction model result dict."""
    if "error" in r:
        return f"ERROR: {r['error']}"
    name = r.get("model_label") or (r.get("model_path") and Path(r["model_path"]).name) or "model"
    pred = r.get("prediction")
    probs = r.get("probabilities")
    parts = [f"{name}"]
    if pred is not None:
        parts.append(f"pred={pred}")
    if probs is not None:
        parts.append(f"probs={probs}")
    return " | ".join(parts)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Pipeline — Solar Flare App", layout="wide")
st.title("Pipeline — Detection then Prediction (fallback)")

# Sidebar: controls
with st.sidebar:
    st.header("Controls")
    st.markdown(
        "Use the controls below to select an image, refresh the random image, and run the pipeline."
    )
    use_preselected = st.checkbox("Use random preselected image on load", value=True)
    st.caption(f"Preselected folder: {PRESELECTED_IMAGES_FOLDER}")

# Initialize current image in session_state as Path or None
if "current_image_path" not in st.session_state:
    try:
        # utils.get_random_preselected_image returns a Path
        st.session_state.current_image_path = utils.get_random_preselected_image()
    except Exception:
        st.session_state.current_image_path = None

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Input image")

    uploaded = st.file_uploader("Upload an image or FITS file (optional)", type=None)

    refresh = st.button("Refresh random preselected image")

    if refresh:
        try:
            st.session_state.current_image_path = utils.get_random_preselected_image()
        except Exception as e:
            st.error(f"Could not load a preselected image: {e}")

    # If user uploaded an image, save it to temp and use it
    chosen_path: Optional[Path] = None
    if uploaded is not None:
        try:
            chosen_path = save_uploaded_to_temp(uploaded)
            st.session_state.current_image_path = chosen_path
        except Exception as e:
            st.error(f"Could not save uploaded file: {e}")
            chosen_path = None

    # Fallback to session_state preselected image
    if chosen_path is None:
        if st.session_state.current_image_path:
            chosen_path = Path(st.session_state.current_image_path)
        else:
            st.info("No image available yet. Upload an image or add files to preselected_images/")

    if chosen_path:
        st.write("Selected file:")
        st.write(f"- Path / name: {chosen_path}")
        # show image preview using utils.prepare_display_image (handles TIFF mode 'F' and FITS if astropy present)
        try:
            preview_img = utils.prepare_display_image(chosen_path)
            st.image(preview_img, use_column_width=True)
        except RuntimeError as re:
            st.write("Preview not available:", re)
        except Exception as e:
            st.write("Preview not available:", e)

    # allow override of detected filter in case filename doesn't contain it
    forced_filter = None
    detected_filter = None
    filter_code: Optional[str] = None
    if chosen_path:
        try:
            detected_filter = utils.parse_filter_from_filename(str(chosen_path.name))
            st.success(f"Detected filter from filename: {detected_filter}")
        except Exception:
            detected_filter = None
            st.warning("Could not auto-detect filter (131/193) from filename. Please select:")

        forced_filter = st.selectbox("Filter (131/193) - override if needed", options=["auto", "131", "193"])
        if forced_filter == "auto":
            filter_code = detected_filter
        else:
            filter_code = forced_filter

    # Action buttons
    st.markdown("---")
    run_pipeline_btn = st.button("Run pipeline (Detection → Prediction fallback)")
    compare_btn = st.button("Compare detection vs prediction side-by-side")


with col2:
    st.subheader("Results")
    if not chosen_path:
        st.info("No image selected. Upload or use a preselected image to run models.")
    else:
        st.markdown(f"**File:** `{chosen_path.name}`")
        st.markdown(f"**Filter (in use):** `{filter_code}`" if filter_code else "**Filter (in use):** _unknown_")

    results_container = st.container()

# ---------- Pipeline logic ----------
def run_detection_then_prediction(image_path: Path, filter_code: Optional[str]):
    with results_container:
        st.info("Running detection models (v3 → v1)...")
        # If filter_code is provided and differs from filename-detected, warn the user,
        # but utils.run_detection_pipeline_for_image determines filter from filename internally.
        try:
            detection_results = utils.run_detection_pipeline_for_image(image_path)
        except FileNotFoundError as e:
            st.error(f"Model / file missing: {e}")
            return
        except Exception as e:
            st.exception(e)
            return

        # Display detection results
        st.markdown("### Detection results")
        for r in detection_results:
            st.write(format_detection_result(r))

        # Determine positive/negative
        positive = detect_positive_from_results(detection_results)
        if positive:
            st.success("Detection: POSITIVE (at least one detection model indicates a flare)")
            # show which model flagged it
            flagged = [format_detection_result(r) for r in detection_results if (r.get("prediction") == 1) or (r.get("probability") and r.get("probability") >= 0.5)]
            if flagged:
                st.write("Flagged by:")
                for f in flagged:
                    st.write(f"- {f}")
        else:
            st.warning("Detection: NEGATIVE (no detection model indicates a flare). Running prediction models as fallback...")
            # Run predictions
            try:
                prediction_results = utils.run_prediction_models_for_image(image_path)
            except FileNotFoundError as e:
                st.error(f"Model / feature columns missing: {e}")
                return
            except Exception as e:
                st.exception(e)
                return

            st.markdown("### Prediction results (fallback)")
            for r in prediction_results:
                st.write(format_prediction_result(r))

            # If predictions have probabilities, display them in a simple table
            probs_table = []
            for r in prediction_results:
                if r.get("probabilities") is not None:
                    probs_table.append({
                        "model": r.get("model_label"),
                        "prediction": r.get("prediction"),
                        "probabilities": str(r.get("probabilities")),
                    })
            if probs_table:
                st.table(probs_table)


def run_compare_detection_vs_prediction(image_path: Path, filter_code: Optional[str]):
    """
    Runs detection and prediction and displays side-by-side columns for visual comparison.
    """
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Detection (v3 → v1)")
        try:
            detection_results = utils.run_detection_pipeline_for_image(image_path)
            for r in detection_results:
                st.write(format_detection_result(r))
            pos = detect_positive_from_results(detection_results)
            if pos:
                st.success("Detection: POSITIVE")
            else:
                st.warning("Detection: NEGATIVE")
        except Exception as e:
            st.error(f"Detection error: {e}")
            st.write(traceback.format_exc())

    with col_b:
        st.markdown("#### Prediction (all prediction models)")
        try:
            prediction_results = utils.run_prediction_models_for_image(image_path)
            for r in prediction_results:
                st.write(format_prediction_result(r))
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.write(traceback.format_exc())


# Button actions
if run_pipeline_btn:
    if not chosen_path:
        st.error("No image selected.")
    else:
        if not filter_code:
            st.error("Cannot run: filter (131/193) unknown. Provide an image filename with filter or select it manually.")
        else:
            run_detection_then_prediction(chosen_path, filter_code)

if compare_btn:
    if not chosen_path:
        st.error("No image selected.")
    else:
        if not filter_code:
            st.error("Cannot run: filter (131/193) unknown. Provide an image filename with filter or select it manually.")
        else:
            run_compare_detection_vs_prediction(chosen_path, filter_code)

# Footer / tips
st.markdown("---")
st.write(
    "Tips: model files and feature column pickles must be present in `models/` as described in the README. "
    "The results shown rely on the (placeholder) feature extractor in `utils.py` unless you replace it with your training extractor."
)
