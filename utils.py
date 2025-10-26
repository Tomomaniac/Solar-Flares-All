"""
utils.py
Shared helpers for the Streamlit Solar Flare app.

Responsibilities:
- parse AIA filter (131/193) from filenames
- list / pick random preselected images
- load models (joblib / pickle)
- load feature-columns pickles for detection/prediction
- lightweight feature extraction (placeholder) and alignment to expected columns
- prepare images for display (convert float TIFF/FITS to 8-bit RGB)
- run detection and prediction model pipelines (accept optional filter_code override)
"""

from pathlib import Path
import os
import re
import random
import pickle
from typing import List, Dict, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image

# try joblib first
try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

# optional astropy / FITS support
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except Exception:
    ASTROPY_AVAILABLE = False

# import configuration values (config.py should be in repo root)
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
from typing import List, Dict, Optional
from pathlib import Path

def run_detection_pipeline_for_image(image_path: Path, filter_code: Optional[str] = None) -> List[Dict]:
    """
    Run ordered detection models for the image.
    If filter_code is None, it is parsed from the filename.
    Returns a list of result dicts:
      { "version": "v3"/"v1", "model_path": Path, "prediction": int|None, "probability": float|None, "raw": list|None, "error": str|None }
    """
    image_path = Path(image_path)
    if filter_code is None:
        filter_code = parse_filter_from_filename(image_path.name)
    model_paths = build_detection_model_paths(filter_code)
    results: List[Dict] = []

    # extract features as Series -> DataFrame
    try:
        feats = basic_extract_features_from_image(image_path)
        X_df = feats.to_frame().T
    except Exception as e:
        return [{"version": None, "model_path": None, "error": f"Feature extraction failed: {e}"}]

    for mp in model_paths:
        version = "v3" if "v3" in mp.name else "v1" if "v1" in mp.name else None
        if not mp.exists():
            results.append({"version": version, "model_path": mp, "error": f"Model file not found: {mp}"})
            continue
        try:
            model = load_model(mp)
        except Exception as e:
            results.append({"version": version, "model_path": mp, "error": f"Model load error: {e}"})
            continue

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_df)
                pos_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
                pred = 1 if pos_prob >= 0.5 else 0
                results.append({
                    "version": version,
                    "model_path": mp,
                    "prediction": pred,
                    "probability": pos_prob,
                    "raw": probs.tolist()
                })
                continue
        except Exception:
            # fall back to predict
            pass

        try:
            p = model.predict(X_df)
            p0 = int(p[0]) if hasattr(p, "__len__") else int(p)
            results.append({
                "version": version,
                "model_path": mp,
                "prediction": p0,
                "probability": None,
                "raw": None
            })
        except Exception as e:
            results.append({"version": version, "model_path": mp, "error": f"Predict error: {e}"})

    return results

# -----------------------
# Filename / filter utils
# -----------------------
def parse_filter_from_filename(filename: str) -> str:
    """
    Parse the AIA filter code ('131' or '193') from a filename.

    Examples matched:
      - AIA20140225_0020_0131.tiff
      - some_name_131.png
      - ...0193...
    Returns '131' or '193'. Raises ValueError if not found.
    """
    bn = Path(filename).name
    # check common patterns like _0131 or _0193 (with word boundary)
    m = re.search(r"_(0?131|0?193)\b", bn, flags=re.IGNORECASE)
    if m:
        return m.group(1).lstrip("0")
    # fallback: any occurrence of 131 or 193
    m2 = re.search(r"(131|193)", bn)
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
    Load a model using joblib (preferred) or pickle as fallback.
    Raises FileNotFoundError if the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if joblib_load:
        try:
            return joblib_load(str(path))
        except Exception:
            # fall back to pickle
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def _format_template_path(tpl: Any, filter_code: str) -> Path:
    """
    Given a template (Path or str) that contains a `{}` or similar placeholder,
    return a concrete Path for the given filter_code.
    """
    # If tpl is a Path, use its name for formatting and preserve parent
    if isinstance(tpl, Path):
        parent = tpl.parent
        name = tpl.name
    else:
        tpl = str(tpl)
        parent = Path(tpl).parent
        name = Path(tpl).name
    # attempt to format name with filter_code
    try:
        formatted_name = name.format(filter_code)
    except Exception:
        # if no placeholder, try simple replacement of '{}' or '{filter}'
        if "{}" in name:
            formatted_name = name.replace("{}", filter_code)
        else:
            # try to inject filter_code at end before extension
            stem, suf = os.path.splitext(name)
            formatted_name = f"{stem}_{filter_code}{suf}"
    return parent / formatted_name


def build_detection_model_paths(filter_code: str) -> List[Path]:
    """
    Build ordered list of detection model Paths for the given filter_code.
    Respects DETECTION_ORDER (e.g., ["v3", "v1"]).
    """
    out: List[Path] = []
    for version_key in DETECTION_ORDER:
        tpl = DETECTION_MODEL_TEMPLATES.get(version_key)
        if tpl is None:
            continue
        out.append(_format_template_path(tpl, filter_code))
    return out


def build_prediction_model_paths(filter_code: str) -> Dict[str, Path]:
    """
    Build a dict mapping prediction model label -> Path for the given filter_code.
    E.g., {"prediction_v1": Path(...), "prediction_v2": Path(...)}
    """
    out: Dict[str, Path] = {}
    for label, tpl in PREDICTION_MODEL_TEMPLATES.items():
        out[label] = _format_template_path(tpl, filter_code)
    return out


# -----------------------
# Feature columns loaders
# -----------------------
def load_feature_columns_for_detection(version: str) -> List[str]:
    """
    Load feature columns list for detection version "v3" or "v1".
    """
    if version == "v3":
        p = FEATURE_COLUMNS_V3
    elif version == "v1":
        p = FEATURE_COLUMNS_V1
    else:
        raise ValueError("Unsupported detection version for feature columns")
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"Feature columns file not found: {p}")
    with open(p, "rb") as f:
        cols = pickle.load(f)
    return list(cols)


def load_feature_columns_for_prediction() -> List[str]:
    p = Path(FEATURE_COLUMNS_PRED)
    if not p.exists():
        raise FileNotFoundError(f"Feature columns file not found: {p}")
    with open(p, "rb") as f:
        cols = pickle.load(f)
    return list(cols)


# -----------------------
# Image reading & display helpers
# -----------------------
def extract_image_array_from_file(path: Path) -> np.ndarray:
    """
    Return a 2D numpy array of image intensity values.
    Supports TIFF/PNG/JPG via PIL and FITS (if astropy available).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".fits":
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("FITS file provided but astropy is not installed.")
        with fits.open(str(path)) as hdul:
            data = hdul[0].data.astype(np.float32)
            # ensure 2D
            if data is None:
                raise RuntimeError("FITS file contains no data")
            return np.nan_to_num(data).astype(np.float32)
    else:
        img = Image.open(str(path)).convert("L")
        return np.asarray(img).astype(np.float32)


def _normalize_array_to_uint8(arr: np.ndarray, clip_percentiles=(1, 99)) -> np.ndarray:
    """
    Normalize a float array to 0-255 uint8 using percentile clipping.
    Robust to NaN/Inf/empty arrays.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if arr.size == 0:
        return arr.astype(np.uint8)
    # compute percentiles with fallback
    try:
        lo, hi = np.percentile(arr, clip_percentiles)
    except Exception:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi == lo:
        return np.full_like(arr, fill_value=128, dtype=np.uint8)
    clipped = np.clip(arr, lo, hi)
    norm = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return norm


def prepare_display_image(path: Path, max_size=(1024, 1024)) -> Image.Image:
    """
    Load an image (TIFF/JPG/PNG/FITS) and return a PIL.Image in 8-bit RGB suitable for st.image.
    For FITS, astropy is required.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".fits":
        if not ASTROPY_AVAILABLE:
            raise RuntimeError("FITS preview requested but astropy is not installed.")
        with fits.open(str(path)) as hdul:
            arr = hdul[0].data.astype(np.float32)
            if arr is None:
                raise RuntimeError("FITS contains no data to display")
        arr8 = _normalize_array_to_uint8(arr)
        img = Image.fromarray(arr8)
        if img.mode == "L":
            img = img.convert("RGB")
    else:
        pil = Image.open(str(path))
        if pil.mode == "F" or pil.mode.startswith("I") or pil.mode == "I;16":
            arr = np.asarray(pil).astype(np.float32)
            arr8 = _normalize_array_to_uint8(arr)
            img = Image.fromarray(arr8).convert("RGB")
        else:
            try:
                img = pil.convert("RGB")
            except Exception:
                img = pil.convert("L").convert("RGB")
    img.thumbnail(max_size, Image.BILINEAR)
    return img


# -----------------------
# Feature extraction & alignment
# -----------------------
def basic_extract_features_from_image(path: Path, resize_to=(64, 64)) -> pd.Series:
    """
    Lightweight placeholder feature extractor:
      - loads image/grayscale array
      - resizes to `resize_to`
      - computes basic stats (mean, std, median, min, max) and flattened pixel features px_0..px_N

    Returns a pandas Series (index=feature name).
    NOTE: Replace with the exact feature extractor used during model training for accurate predictions.
    """
    arr = extract_image_array_from_file(Path(path))
    # use PIL resize on array for consistent behavior
    img = Image.fromarray(np.nan_to_num(arr)).convert("L").resize(resize_to, Image.BILINEAR)
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


def align_features_to_columns(features_series: pd.Series, feature_columns: List[str]) -> pd.DataFrame:
    """
    Align a pandas Series of features to expected feature columns:
      - missing columns -> fill with 0.0
      - extra columns -> dropped
    Returns a single-row DataFrame with columns ordered as feature_columns.
    """
    df = features_series.to_frame().T
    df = df.reindex(columns=feature_columns, fill_value=0.0)
    return df


# -----------------------
# Inference wrappers
# -----------------------
def run_detection_pipeline_for_image(image_path: Path, filter_code: Optional[str] = None) -> List[Dict]:
    """
    Run ordered detection models for the image.
    If filter_code is None, it is parsed from the filename.
    Returns a list of result dicts:
      { "version": "v3"/"v1", "model_path": Path, "prediction": int|None, "probability": float|None, "raw": list|None, "error": str|None }
    """
    image_path = Path(image_path)
    if filter_code is None:
        filter_code = parse_filter_from_filename(image_path.name)
    model_paths = build_detection_model_paths(filter_code)
    results: List[Dict] = []

    # extract features as Series -> DataFrame
    try:
        feats = basic_extract_features_from_image(image_path)
        X_df = feats.to_frame().T
    except Exception as e:
        return [{"version": None, "model_path": None, "error": f"Feature extraction failed: {e}"}]

    for mp in model_paths:
        version = "v3" if "v3" in mp.name else "v1" if "v1" in mp.name else None
        if not mp.exists():
            results.append({"version": version, "model_path": mp, "error": f"Model file not found: {mp}"})
            continue
        try:
            model = load_model(mp)
        except Exception as e:
            results.append({"version": version, "model_path": mp, "error": f"Model load error: {e}"})
            continue

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_df)
                pos_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
                pred = 1 if pos_prob >= 0.5 else 0
                results.append({
                    "version": version,
                    "model_path": mp,
                    "prediction": pred,
                    "probability": pos_prob,
                    "raw": probs.tolist()
                })
                continue
        except Exception:
            # fall back to predict
            pass

        try:
            p = model.predict(X_df)
            p0 = int(p[0]) if hasattr(p, "__len__") else int(p)
            results.append({
                "version": version,
                "model_path": mp,
                "prediction": p0,
                "probability": None,
                "raw": None
            })
        except Exception as e:
            results.append({"version": version, "model_path": mp, "error": f"Predict error: {e}"})

    return results


def run_prediction_models_for_image(image_path: Path, filter_code: Optional[str] = None) -> List[Dict]:
    """
    Run the prediction-model family (prediction_v1 and prediction_v2/pred3) for the image.
    If filter_code is None, it is parsed from the filename.
    Returns a list of dicts:
      { "model_label": label, "model_path": Path, "prediction": int|None, "probabilities": list|None, "error": str|None }
    """
    image_path = Path(image_path)
    if filter_code is None:
        filter_code = parse_filter_from_filename(image_path.name)
    model_paths = build_prediction_model_paths(filter_code)

    # load expected feature columns for prediction
    try:
        feature_cols = load_feature_columns_for_prediction()
    except Exception as e:
        return [{"model_label": None, "model_path": None, "error": f"Feature columns load error: {e}"}]

    # extract features and align
    try:
        feats = basic_extract_features_from_image(image_path)
        X_df = align_features_to_columns(feats, feature_cols)
    except Exception as e:
        return [{"model_label": None, "model_path": None, "error": f"Feature extraction/alignment error: {e}"}]

    results: List[Dict] = []
    for label, mp in model_paths.items():
        if not mp.exists():
            results.append({"model_label": label, "model_path": mp, "error": f"Model file not found: {mp}"})
            continue
        try:
            model = load_model(mp)
        except Exception as e:
            results.append({"model_label": label, "model_path": mp, "error": f"Model load error: {e}"})
            continue

        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_df)
                pred = int(model.predict(X_df)[0])
                results.append({"model_label": label, "model_path": mp, "prediction": pred, "probabilities": probs.tolist()})
                continue
        except Exception:
            pass

        try:
            pred = model.predict(X_df)
            p0 = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            results.append({"model_label": label, "model_path": mp, "prediction": p0, "probabilities": None})
        except Exception as e:
            results.append({"model_label": label, "model_path": mp, "error": f"Predict error: {e}"})

    return results
