# Customer Lifetime Value (CLV) Prediction & Deployment Workflow

End-to-end Data Science & Machine Learning regression pipeline and Streamlit web application predicting **Yearly Amount Spent (CLV)** from customer engagement metrics.

## Project Structure
- `ecommerce_customers.csv` — 500-customer dataset with session length, mobile app time, desktop website time, membership duration, and yearly spending.
- `train_pipeline.py` — Complete Python workflow executing Steps 1–8 (Dataset Inspection, EDA plots, Median Imputation & IQR Outlier Removal, Feature Engineering, StandardScaler & 80/20 Train-Test Split, OLS Linear Regression, Evaluation Metrics, and `joblib` Artifact Export).
- `app.py` — Interactive Streamlit web application (Step 9) for real-time CLV simulation.
- `model_artifacts.json` — Pre-computed scaler parameters, model weights, and evaluation metrics.
- `eda_target_distribution.png` & `eda_correlation_heatmap.png` — Generated EDA charts.
- `requirements.txt` — Python package dependencies for local execution and Streamlit Community Cloud deployment.

## Model Results Summary (Steps 6 & 7)
- **Cleaned Dataset**: 489 records (3 missing `Time on App` values imputed with median `12.06` min; 11 extreme target outliers outside `[$309.26, $689.91]` filtered via 1.5×IQR).
- **Intercept (Mean Baseline Spend)**: `$498.81`
- **Standardized Coefficients (`StandardScaler`)**:
  - `Length of Membership`: **+$55.13** per standard deviation (+$61.20 per year)
  - `Time on App`: **+$37.34** per standard deviation (+$38.33 per minute)
  - `Avg. Session Length`: **+$25.30** per standard deviation (+$25.56 per minute)
  - `Time on Website`: **-$0.44** per standard deviation (-$0.43 per minute)
- **Holdout Test Performance (20% Split)**:
  - **MAE**: `$7.22`
  - **RMSE**: `$9.05`
  - **R² Score**: `0.9869` (98.69% of variance explained)

## Quickstart (Step 10: Run & Deploy)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training pipeline to generate linear_clv_model.pkl and clv_scaler.pkl
python train_pipeline.py

# 3. Launch the Streamlit interactive web app
streamlit run app.py
```
