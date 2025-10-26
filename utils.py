# utils.py
import re
import os
import random
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from PIL import Image

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

# Optional FITS support
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

from config import (
    DETECTION_MODEL_TEMPLATES,
    DETECTION_ORDER,
    PREDICTION_MODEL_TEMPLATES,
    FEATURE_COLUMNS_V3,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_PRED,
    PRESELECTED_IMAGES_FOLDER,
    IMAGE_EXTENSIONS,
)

# Add/replace these imports at top of utils.py if not present
from typing import Optional, List, Dict

# Normalization helper (robust)
def _normalize_array_to_uint8(arr: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Normalize a float array to 0-255 uint8 using percentile clipping.
    Robust to NaNs/Infs and empty arrays.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    # compute percentiles safely
    try:
        lo, hi = np.percentile(arr, clip_percentiles)
    except Exception:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi == lo:
        # constant image -> mid-gray
        norm = np.full_like(arr, fill_value=128, dtype=np.uint8)
    else:
        clipped = np.clip(arr, lo, hi)
        norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return norm


def run_detection_pipeline_for_image(image_path: Path, filter_code: Optional[str] = None) -> List[Dict]:
    """
    Run detection models in order (v3 then v1) for the specified filter.
    If filter_code is None, attempt to parse from filename.
    Returns list of result dicts with keys:
      - version, model_path, prediction (int or None), probability (float or None), raw (optional), error (optional)
    """
    # determine filter
    if filter_code is None:
        filter_code = parse_filter_from_filename(str(image_path.name))
    # build ordered detection model paths
    model_paths = build_detection_model_paths(filter_code)
    results = []

    # extract features (uses your basic extractor or a real one)
    try:
        feats = basic_extract_features_from_image(image_path)
    except Exception as e:
        # return single error entry
        return [{"version": None, "model_path": None, "error": f"Feature extraction failed: {e}"}]

    for model_path in model_paths:
        version = "v3" if "v3" in model_path.name else "v1"
        try:
            model = load_model(model_path)
        except Exception as e:
            results.append({"version": version, "model_path": model_path, "error": f"Model load error: {e}"})
            continue

        # Try predict_proba if available
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feats.to_frame().T)
                # if binary, assume positive class is index 1
                pos_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
                pred = 1 if pos_prob >= 0.5 else 0
                results.append({
                    "version": version,
                    "model_path": model_path,
                    "prediction": pred,
                    "probability": pos_prob,
                    "raw": probs.tolist()
                })
                continue
        except Exception:
            # ignore and fallback to predict
            pass

        # Fallback to predict()
        try:
            p = model.predict(feats.to_frame().T)
            p0 = int(p[0]) if hasattr(p, "__len__") else int(p)
            results.append({
                "version": version,
                "model_path": model_path,
                "prediction": p0,
                "probability": None,
                "raw": None
            })
        except Exception as e:
            results.append({"version": version, "model_path": model_path, "error": f"Predict error: {e}"})

    return results


def run_prediction_models_for_image(image_path: Path, filter_code: Optional[str] = None) -> List[Dict]:
    """
    Run prediction models (prediction_v1 and pred3/prediction_v2) for the given filter.
    If filter_code is None, attempt to parse from filename.
    Returns list of dicts:
      - model_label, model_path, prediction, probabilities (or None), error (optional)
    """
    if filter_code is None:
        filter_code = parse_filter_from_filename(str(image_path.name))

    model_paths = build_prediction_model_paths(filter_code)  # returns dict label -> Path
    # load feature columns expected by prediction models
    try:
        feature_cols = load_feature_columns_for_prediction()
    except Exception as e:
        return [{"model_label": None, "model_path": None, "error": f"Feature columns load error: {e}"}]

    # extract and align features
    try:
        feats = basic_extract_features_from_image(image_path)
        df = align_features_to_columns(feats, feature_cols)
    except Exception as e:
        return [{"model_label": None, "model_path": None, "error": f"Feature extraction/alignment error: {e}"}]

    results = []
    for label, mp in model_paths.items():
        try:
            model = load_model(mp)
        except Exception as e:
            results.append({"model_label": label, "model_path": mp, "error": f"Model load error: {e}"})
            continue

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                pred = int(model.predict(df)[0])
                results.append({"model_label": label, "model_path": mp, "prediction": pred, "probabilities": probs.tolist()})
                continue
        except Exception:
            pass

        try:
            pred = model.predict(df)
            p0 = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            results.append({"model_label": label, "model_path": mp, "prediction": p0, "probabilities": None})
        except Exception as e:
            results.append({"model_label": label, "model_path": mp, "error": f"Predict error: {e}"})

    return results

# utils.py -- add these imports near the top if not already present
from PIL import Image
import numpy as np

# utils.py -- add this helper (paste under other image helpers)
def _normalize_array_to_uint8(arr: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Normalize a float array to 0-255 uint8 using percentile clipping to reduce effect of outliers.
    """
    # replace NaNs/Infs
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # clip using percentiles (robust)
    lo, hi = np.percentile(arr, clip_percentiles)
    if hi <= lo:
        # fallback to min/max
        lo, hi = arr.min(), arr.max()
    if hi == lo:
        # constant image
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        clipped = np.clip(arr, lo, hi)
        norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return norm


def prepare_display_image(path: Path, max_size=(1024, 1024)) -> Image.Image:
    """
    Load an image file (TIFF, PNG, JPG, FITS) and return a PIL.Image in 8-bit mode
    suitable for Streamlit display. This normalizes float images to uint8.
    - path: Path or str to image
    - max_size: maximum size to downscale for preview (keeps aspect)
    """
    path = Path(path)
    suffix = path.suffix.lower()
    # FITS support
    if suffix == ".fits":
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("FITS preview requested but astropy is not available in the environment.")
        from astropy.io import fits
        with fits.open(str(path)) as hdul:
            data = hdul[0].data.astype(np.float32)
        arr8 = _normalize_array_to_uint8(data)
        img = Image.fromarray(arr8)
        # If single channel, convert to L then RGB for nicer preview
        if img.mode == "L":
            img = img.convert("RGB")
    else:
        # For normal rasters, use PIL then handle float mode
        try:
            pil = Image.open(str(path))
        except Exception as e:
            raise RuntimeError(f"Could not open image for preview: {e}")
        # If mode is "F" or other non-8bit, convert using numpy normalization
        if pil.mode == "F" or pil.mode.startswith("I") or pil.mode == "I;16":
            arr = np.asarray(pil).astype(np.float32)
            arr8 = _normalize_array_to_uint8(arr)
            img = Image.fromarray(arr8).convert("RGB")
        else:
            # Convert palette/mode to RGB for consistent display
            try:
                img = pil.convert("RGB")
            except Exception:
                # last resort: convert to L then RGB
                img = pil.convert("L").convert("RGB")

    # Resize down for preview if necessary while keeping aspect ratio
    img.thumbnail(max_size, Image.BILINEAR)
    return img

# -----------------------
# Filename / filter utils
# -----------------------
def parse_filter_from_filename(filename: str) -> str:
    """
    Parse the filter (131 or 193) from the filename.
    Examples of expected filename patterns:
      - AIA20140225_0020_0131...
      - ..._0131.ext or ..._131.ext or contain 131/193 near underscores.
    Returns '131' or '193'. Raises ValueError if not found.
    """
    bn = os.path.basename(filename)
    bn_noext = os.path.splitext(bn)[0]
    # look for patterns like _0131 or _0193 first
    m = re.search(r"_(0?131|0?193)\b", bn_noext)
    if m:
        return m.group(1).lstrip("0")
    # fallback: any occurrence of 131 or 193
    m2 = re.search(r"(131|193)", bn_noext)
    if m2:
        return m2.group(1)
    raise ValueError(f"Could not detect filter (131/193) from filename: {filename}")


# -----------------------
# Preselected image utils
# -----------------------
def get_preselected_candidates(folder: Path = PRESELECTED_IMAGES_FOLDER) -> List[Path]:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]


def get_random_preselected_image(folder: Path = PRESELECTED_IMAGES_FOLDER) -> Path:
    candidates = get_preselected_candidates(folder)
    if not candidates:
        raise FileNotFoundError(f"No preselected images found in {folder}")
    return random.choice(candidates)


# -----------------------
# Model loading helpers
# -----------------------
def load_model(path: Path):
    """
    Load a model file using joblib (preferred) or pickle fallback.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if joblib_load:
        try:
            return joblib_load(str(path))
        except Exception:
            pass
    # fallback pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def build_detection_model_paths(filter_code: str) -> List[Path]:
    """
    Build ordered list of detection model paths for the filter.
    Respect DETECTION_ORDER (v3 before v1).
    """
    out = []
    for v in DETECTION_ORDER:
        tpl = DETECTION_MODEL_TEMPLATES.get(v)
        if tpl is None:
            continue
        out.append(Path(str(tpl)).with_name(tpl.name.format(filter_code)))
    return out


def build_prediction_model_paths(filter_code: str) -> Dict[str, Path]:
    """
    Return a dict with prediction model label -> path
    """
    out = {}
    for name, tpl in PREDICTION_MODEL_TEMPLATES.items():
        out[name] = Path(str(tpl)).with_name(tpl.name.format(filter_code))
    return out


# -----------------------
# Feature columns loaders
# -----------------------
def load_feature_columns_for_detection(version: str) -> List[str]:
    if version == "v3":
        p = FEATURE_COLUMNS_V3
    elif version == "v1":
        p = FEATURE_COLUMNS_V1
    else:
        raise ValueError("Unsupported detection version for feature columns")
    if not Path(p).exists():
        raise FileNotFoundError(f"Feature columns file not found: {p}")
    with open(p, "rb") as f:
        cols = pickle.load(f)
    return list(cols)


def load_feature_columns_for_prediction() -> List[str]:
    p = FEATURE_COLUMNS_PRED
    if not Path(p).exists():
        raise FileNotFoundError(f"Feature columns file not found: {p}")
    with open(p, "rb") as f:
        cols = pickle.load(f)
    return list(cols)


# -----------------------
# Basic feature extractor (placeholder)
# -----------------------
def extract_image_array_from_file(path: Path) -> np.ndarray:
    """
    Return a 2D numpy array of image intensity values.
    Tries FITS (if astropy available), otherwise loads with PIL and converts to grayscale.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".fits":
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("FITS file provided but astropy is not installed.")
        with fits.open(str(path)) as hdul:
            data = hdul[0].data.astype(np.float32)
            # Flip/reshape if necessary
            return data
    else:
        img = Image.open(str(path)).convert("L")
        return np.asarray(img).astype(np.float32)


def basic_extract_features_from_image(path: Path, resize_to=(64, 64)) -> pd.Series:
    """
    Simple extractor:
    - read grayscale array
    - resize (via PIL) to reduce dimensionality
    - compute global stats + flattened pixel features (px_0..px_N)
    NOTE: Replace with your real extraction pipeline if you used different features.
    """
    arr = extract_image_array_from_file(path)
    # Normalize and resize using PIL for simplicity
    img = Image.fromarray(np.nan_to_num(arr))
    img = img.convert("L").resize(resize_to, Image.BILINEAR)
    vals = np.asarray(img).astype(np.float32).ravel() / 255.0
    stats = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }
    flat = {f"px_{i}": float(v) for i, v in enumerate(vals)}
    features = {**stats, **flat}
    return pd.Series(features)


def align_features_to_columns(series: pd.Series, columns: List[str]) -> pd.DataFrame:
    df = series.to_frame().T
    df = df.reindex(columns=columns, fill_value=0.0)
    return df


# -----------------------
# Inference wrappers
# -----------------------
def run_detection_pipeline_for_image(image_path: Path) -> List[Dict]:
    """
    Runs detection models in order (v3 then v1) for the right filter detected from filename.
    Returns list of result dicts:
      { "version": "v3", "model_path": Path, "prediction": ..., "probability": ..., "raw": ... }
    """
    filter_code = parse_filter_from_filename(str(image_path))
    model_paths = build_detection_model_paths(filter_code)
    results = []
    # Use features aligned to the specific detection version columns if you wish;
    # here we use basic extractor and pass results to models directly.
    feats = basic_extract_features_from_image(image_path)
    for model_path in model_paths:
        version = "v3" if "v3" in model_path.name else "v1"
        try:
            model = load_model(model_path)
        except Exception as e:
            results.append({"version": version, "model_path": model_path, "error": str(e)})
            continue
        # try predict_proba then predict
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feats.to_frame().T)
                pos_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
                pred = int(round(pos_prob))
                results.append({
                    "version": version,
                    "model_path": model_path,
                    "prediction": pred,
                    "probability": pos_prob,
                    "raw": probs.tolist(),
                })
                continue
        except Exception:
            pass
        try:
            p = model.predict(feats.to_frame().T)
            p0 = int(p[0]) if hasattr(p, "__len__") else int(p)
            results.append({
                "version": version,
                "model_path": model_path,
                "prediction": p0,
                "probability": None,
                "raw": None,
            })
        except Exception as e:
            results.append({"version": version, "model_path": model_path, "error": str(e)})
    return results


def run_prediction_models_for_image(image_path: Path) -> List[Dict]:
    """
    Run available prediction models (both v1 and v2/pred3) for the filter detected from filename.
    Aligns features to feature_columns_prediction.pkl
    Returns list of dicts:
      { "model_label": "prediction_v1", "model_path": Path, "prediction": ..., "probabilities": ... }
    """
    filter_code = parse_filter_from_filename(str(image_path))
    model_paths = build_prediction_model_paths(filter_code)
    # load pred feature cols and extract features
    try:
        feature_cols = load_feature_columns_for_prediction()
    except Exception as e:
        raise
    feats = basic_extract_features_from_image(image_path)
    df = align_features_to_columns(feats, feature_cols)
    out = []
    for label, mp in model_paths.items():
        try:
            model = load_model(mp)
        except Exception as e:
            out.append({"model_label": label, "model_path": mp, "error": str(e)})
            continue
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)
                pred = int(model.predict(df)[0])
                out.append({"model_label": label, "model_path": mp, "prediction": pred, "probabilities": probs.tolist()})
                continue
        except Exception:
            pass
        try:
            pred = model.predict(df)
            p0 = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            out.append({"model_label": label, "model_path": mp, "prediction": p0, "probabilities": None})
        except Exception as e:
            out.append({"model_label": label, "model_path": mp, "error": str(e)})
    return out



