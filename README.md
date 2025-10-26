# Solar-Flare-Identification-app

# ☀️ Solar Flare Detection App

Machine learning-powered web application for detecting solar flares in EUV images from SDO/AIA.

## 🚀 Live Demo

[Try it here!](https://your-app-name.streamlit.app)

## 📖 About

This application uses Random Forest classifiers trained on NASA's Solar Dynamics Observatory (SDO) Atmospheric Imaging Assembly (AIA) data to detect solar flares in extreme ultraviolet (EUV) images.

### Models

- **Model v1**: Trained on QUIET vs DURING flare images
  - 131 Å: ~95% accuracy
  - 193 Å: ~91% accuracy

- **Model v3**: Trained on (BEFORE+QUIET) vs DURING with raw brightness features
  - 131 Å: ~91% accuracy
  - 193 Å: ~89% accuracy
  - More robust for real-world scenarios

### Filters

- **131 Å**: Fe VIII/XXI - Captures flare plasma
- **193 Å**: Fe XII/XXIV - Captures corona and flare loops

## 🛠️ Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/solar-flare-detector.git
cd solar-flare-detector

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run flare_detector_simple.py
