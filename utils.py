"""
utils.py

Small compatibility helper used by the Streamlit pages.

Provides:
- get_random_preselected_image(): pick a random file from preselected_images/
- prepare_display_image(path): open image, normalize and apply colormap for preview
- parse_filter_from_filename(filename): detect 131/193 from filename
- run_detection_pipeline_for_image(path, filter_code, detection_versions=None):
  run detection models (v3, v1) using detection_utils and return summarized results

This file is intentionally small and re-uses functions from detection_utils.
"""
from pathlib import Path
import random
from typing import Optional, List

import numpy as np
from PIL import Image

from detection_utils import preprocess_image, apply_colormap, load_models, predict_flare


PRESELECTED_DIR = Path("preselected_images")


def get_random_preselected_image() -> Optional[Path]:
    """Return a random image Path from preselected_images or None if none available."""
    if not PRESELECTED_DIR.exists():
        return None

    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
    image_files = []
    # Search recursively through subdirectories
    for ext in image_extensions:
        image_files.extend(list(PRESELECTED_DIR.rglob(f'*{ext}')))
        image_files.extend(list(PRESELECTED_DIR.rglob(f'*{ext.upper()}')))

    if not image_files:
        return None

    return random.choice(image_files)


def prepare_display_image(path: Path, colormap: str = 'hot'):
    """Open an image file and return an RGB numpy array suitable for st.image preview.

    The function normalizes the image and applies a colormap so previews are consistent.
    """
    img = Image.open(path)
    arr = np.array(img)
    # In case of grayscale images, ensure 2D array
    if arr.ndim == 3 and arr.shape[2] == 4:
        # drop alpha
        arr = arr[:, :, :3]
    if arr.ndim == 3:
        # convert to grayscale by averaging channels for normalization/colormap
        arr_gray = np.mean(arr, axis=2)
    else:
        arr_gray = arr

    arr_norm = preprocess_image(arr_gray)
    arr_rgb = apply_colormap(arr_norm, colormap)
    return arr_rgb


def parse_filter_from_filename(filename: str) -> Optional[str]:
    """Detect '131' or '193' filter code from filename. Returns '131'|'193' or None."""
    fn = filename.lower()
    if '0131' in fn or '131' in fn:
        return '131'
    if '0193' in fn or '193' in fn:
        return '193'
    return None


# Simple module-level cache to avoid reloading models repeatedly
_LOADED_MODELS = None  # tuple(models, feature_cols_v1, feature_cols_v3)


def _ensure_models_loaded():
    global _LOADED_MODELS
    if _LOADED_MODELS is None:
        _LOADED_MODELS = load_models()
    return _LOADED_MODELS


def run_detection_pipeline_for_image(path: Path, filter_code: str, detection_versions: Optional[List[str]] = None):
    """Run detection models on the provided image path.

    Args:
        path: Path to image file
        filter_code: '131' or '193'
        detection_versions: list like ['v3','v1'] or None to use ['v3','v1']

    Returns:
        List of results dicts with keys: model_path, prediction, probability, raw, features
    """
    if detection_versions is None:
        detection_versions = ['v3', 'v1']

    try:
        models, feature_cols_v1, feature_cols_v3 = _ensure_models_loaded()
    except Exception as e:
        raise FileNotFoundError(f"Could not load detection models: {e}")

    # Read image
    img = Image.open(path)
    img_arr = np.array(img).astype(np.float32)
    # If color, convert to gray by averaging channels
    if img_arr.ndim == 3:
        img_gray = np.mean(img_arr, axis=2)
    else:
        img_gray = img_arr

    results = []
    for version in detection_versions:
        try:
            res = predict_flare(
                img_gray,
                version,
                filter_code,
                models,
                feature_cols_v1,
                feature_cols_v3,
            )

            # probability of positive class (1)
            prob = None
            try:
                prob = float(res['probabilities'][1])
            except Exception:
                # fallback if binary label mapping differs
                prob = float(max(res.get('probabilities', [0.0])))

            raw = None
            if 'raw_max' in res.get('features', {}):
                raw = res['features']['raw_max']

            results.append({
                'model_path': f"{version}_{filter_code}",
                'prediction': int(res.get('prediction')) if res.get('prediction') is not None else None,
                'probability': prob,
                'raw': raw,
                'features': res.get('features', {})
            })
        except Exception as e:
            results.append({'model_path': f"{version}_{filter_code}", 'error': str(e)})

    return results
