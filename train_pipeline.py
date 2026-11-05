"""
Customer Lifetime Value (CLV) Prediction & Deployment Workflow
Steps 1 to 8: Dataset Inspection, EDA, Cleaning, Feature Engineering,
Preprocessing, Linear Regression Training, Evaluation, and Artifact Export.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    # ==========================================
    # Step 1: Dataset Selection & Inspection
    # ==========================================
    print("--- Step 1: Loading & Inspecting Dataset ---")
    df = pd.read_csv('ecommerce_customers.csv')
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())

    # ==========================================
    # Step 2: Exploratory Data Analysis (EDA)
    # ==========================================
    print("\n--- Step 2: Generating EDA Visualizations ---")
    sns.set_theme(style="whitegrid")
    
    # Target distribution plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Yearly Amount Spent'], kde=True, color='#2563eb')
    plt.title('Distribution of Yearly Amount Spent')
    plt.tight_layout()
    plt.savefig('eda_target_distribution.png', dpi=150)
    plt.close()

    # Pairwise correlation plot
    pairplot_fig = sns.pairplot(df.select_dtypes(include=[np.number]))
    pairplot_fig.savefig('eda_pairplot.png', dpi=120)
    plt.close()

    # Multicollinearity correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='Blues')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png', dpi=150)
    plt.close()

    # ==========================================
    # Step 3: Data Cleaning
    # ==========================================
    print("\n--- Step 3: Cleaning Missing Values & IQR Outliers ---")
    null_count = df['Time on App'].isna().sum()
    median_app = df['Time on App'].median()
    df['Time on App'] = df['Time on App'].fillna(median_app)
    print(f"Imputed {null_count} missing 'Time on App' values with median ({median_app:.2f} min).")

    # IQR Outlier Removal for target variable
    Q1 = df['Yearly Amount Spent'].quantile(0.25)
    Q3 = df['Yearly Amount Spent'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before_rows = len(df)
    df = df[(df['Yearly Amount Spent'] >= lower_bound) & (df['Yearly Amount Spent'] <= upper_bound)].copy()
    print(f"IQR Bounds: [${lower_bound:.2f}, ${upper_bound:.2f}] | Removed {before_rows - len(df)} outliers ({len(df)} clean records remaining).")

    # ==========================================
    # Step 4: Feature Engineering
    # ==========================================
    print("\n--- Step 4: Engineering Domain Features ---")
    df['App_to_Web_Ratio'] = df['Time on App'] / df['Time on Website']
    df['Engagement_Score'] = df['Avg. Session Length'] * df['Length of Membership']
    print(df[['App_to_Web_Ratio', 'Engagement_Score', 'Yearly Amount Spent']].head())

    # ==========================================
    # Step 5: Data Preprocessing & Train-Test Split
    # ==========================================
    print("\n--- Step 5: Train-Test Split & StandardScaling ---")
    feature_cols = ['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']
    X = df[feature_cols]
    y = df['Yearly Amount Spent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # Step 6: Model Training & Coefficient Analysis
    # ==========================================
    print("\n--- Step 6: Training Linear Regression Model ---")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print(f"Intercept: {model.intercept_:.2f}")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"Feature: {feature:<22} | Scaled Coefficient: {coef:.2f}")

    # ==========================================
    # Step 7: Evaluation & Predictions
    # ==========================================
    print("\n--- Step 7: Test Set Evaluation ---")
    predictions = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"MAE:      ${mae:.2f}")
    print(f"RMSE:     ${rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # ==========================================
    # Step 8: Saving Components for Production
    # ==========================================
    print("\n--- Step 8: Saving Production Artifacts ---")
    joblib.dump(model, 'linear_clv_model.pkl')
    joblib.dump(scaler, 'clv_scaler.pkl')
    print("Saved 'linear_clv_model.pkl' and 'clv_scaler.pkl' successfully.")

if __name__ == '__main__':
    main()
