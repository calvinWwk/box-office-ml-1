
# ML + Streamlit Starter (created 2025-08-26)

## Files
- `train_two_models.ipynb` — trains Linear Regression and Random Forest regressors, evaluates them, and exports artifacts to `artifacts/`.
- `app.py` — Streamlit app that loads the artifacts and serves predictions.
- `requirements.txt` — python dependencies.

## Quickstart

1) (Optional) Create/activate a virtual environment, then install dependencies:
```bash
pip install -r requirements.txt
```

2) Prepare your data CSV (default expected name: `movies.csv`) with a numeric target column **`revenue`** (preferred) or **`target`** and numeric feature columns.

3) Open and run the notebook:
- Set `DATA_PATH` at the top if your file isn't named `movies.csv`.
- Run all cells to train models and export:
  - `artifacts/preprocessor.pkl`
  - `artifacts/model_linear.pkl`
  - `artifacts/model_rf.pkl`
  - `artifacts/metadata.json`

4) Launch the Streamlit app from the same folder:
```bash
streamlit run app.py
```

5) In the UI:
- Choose a model in the dropdown.
- Enter the numeric features (the app builds inputs from `metadata.json`).
- Click **Predict** to see the estimated revenue.

### Notes
- Extend the preprocessing in the notebook if you want to include categorical/text features.
- For explainability, enable the optional SHAP cell in the notebook.
- This starter is framework-agnostic enough to adapt to any tabular regression problem by renaming the target to `revenue` or `target`.
