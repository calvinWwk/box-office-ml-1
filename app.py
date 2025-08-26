import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Movie Revenue Predictor", page_icon="🎬", layout="centered")

ARTIFACT_DIR = Path("artifacts")
meta_path = ARTIFACT_DIR / "metadata.json"
prep_path = ARTIFACT_DIR / "preprocessor.pkl"

if not meta_path.exists() or not prep_path.exists():
    st.error("Artifacts not found. Please run the training notebook to generate artifacts in the 'artifacts/' folder.")
    st.stop()

with open(meta_path) as f:
    meta = json.load(f)

preprocessor = joblib.load(prep_path)

# Load models
available_models = {}
for name in meta.get("models", []):
    p = ARTIFACT_DIR / f"model_{name}.pkl"
    if p.exists():
        available_models[name] = joblib.load(p)

if not available_models:
    st.error("No models available. Please ensure 'model_linear.pkl' and/or 'model_rf.pkl' are present in 'artifacts/'.")
    st.stop()

st.title("🎬 Movie Revenue Predictor")
st.caption("Pick a model, enter numeric features, and predict the revenue.")

model_choice = st.selectbox("Model", sorted(list(available_models.keys())))
model = available_models[model_choice]

# Build input UI dynamically from metadata
st.subheader("Input features")
cols = meta["feature_names_in"]
feat_summary = meta.get("feature_summary", {})

inputs = {}
for c in cols:
    stats = feat_summary.get(c, {})
    min_v = stats.get("min", 0.0)
    max_v = stats.get("max", 1.0)
    mean_v = stats.get("mean", 0.0)
    # Be robust to bad stats
    if not np.isfinite(min_v): min_v = 0.0
    if not np.isfinite(max_v) or max_v <= min_v: max_v = min_v + 1.0
    if not np.isfinite(mean_v): mean_v = (min_v + max_v) / 2.0

    inputs[c] = st.number_input(f"{c}", value=float(mean_v), min_value=float(min_v), max_value=float(max_v))

if st.button("Predict"):
    row = pd.DataFrame([inputs])
    X_t = preprocessor.transform(row)
    pred = float(model.predict(X_t)[0])
    st.success(f"Estimated {meta['target']} = {pred:,.2f}")
    with st.expander("Debug info"):
        st.json({"model": model_choice, "inputs": inputs})
else:
    st.info("Set the inputs and click **Predict**.")