"""
California Housing Price Prediction using Random Forest
Complete with Exploratory Data Analysis (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CALIFORNIA HOUSING PRICE PREDICTION - RANDOM FOREST")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading California Housing Dataset...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

print(f"Dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("[2] EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# 2.1 Basic Statistics
print("\n[2.1] Dataset Information:")
print(df.info())
print("\n[2.2] Descriptive Statistics:")
print(df.describe())

# 2.2 Missing Values
print("\n[2.3] Missing Values:")
print(df.isnull().sum())
print(f"Total missing values: {df.isnull().sum().sum()}")

# 2.3 Target Variable Distribution
print("\n[2.4] Target Variable (MedHouseVal) Statistics:")
print(df['MedHouseVal'].describe())

# 2.4 Correlation Analysis
print("\n[2.5] Correlation Matrix:")
correlation_matrix = df.corr()
print(correlation_matrix['MedHouseVal'].sort_values(ascending=False))

# ============================================================================
# 3. DATA VISUALIZATION
# ============================================================================
print("\n[3] Creating Visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# 3.1 Target Variable Distribution
ax1 = plt.subplot(3, 3, 1)
df['MedHouseVal'].hist(bins=50, edgecolor='black', ax=ax1)
ax1.set_title('Distribution of Median House Values', fontsize=12, fontweight='bold')
ax1.set_xlabel('Median House Value (in $100,000s)')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# 3.2 Box Plot of Target Variable
ax2 = plt.subplot(3, 3, 2)
df.boxplot(column='MedHouseVal', ax=ax2)
ax2.set_title('Box Plot of Median House Values', fontsize=12, fontweight='bold')
ax2.set_ylabel('Median House Value (in $100,000s)')
ax2.grid(True, alpha=0.3)

# 3.3 Correlation Heatmap
ax3 = plt.subplot(3, 3, 3)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax3)
ax3.set_title('Correlation Heatmap', fontsize=12, fontweight='bold')

# 3.4 Feature Distributions
feature_cols = df.columns[:-1]  # All except target
for idx, feature in enumerate(feature_cols[:6], start=4):
    ax = plt.subplot(3, 3, idx)
    df[feature].hist(bins=30, edgecolor='black', ax=ax)
    ax.set_title(f'Distribution of {feature}', fontsize=10, fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('california_housing_eda.png', dpi=300, bbox_inches='tight')
print("Saved: california_housing_eda.png")
plt.close()

# 3.5 Scatter Plots: Features vs Target
fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for idx, feature in enumerate(feature_cols):
    axes[idx].scatter(df[feature], df['MedHouseVal'], alpha=0.3, s=10)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Median House Value', fontsize=10)
    axes[idx].set_title(f'{feature} vs MedHouseVal', fontsize=10, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('california_housing_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: california_housing_scatter.png")
plt.close()

# 3.6 Feature Importance Preview (using correlation)
fig3, ax = plt.subplots(figsize=(10, 6))
corr_with_target = correlation_matrix['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=True)
corr_with_target.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Feature Correlation with Target Variable', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('california_housing_correlation.png', dpi=300, bbox_inches='tight')
print("Saved: california_housing_correlation.png")
plt.close()

# ============================================================================
# 4. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("[4] DATA PREPROCESSING")
print("=" * 80)

# Separate features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for outliers (using IQR method)
print("\n[4.1] Outlier Detection (IQR Method):")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
print("Outliers per column:")
print(outliers)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[4.2] Train-Test Split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Optional: Feature Scaling (Random Forest doesn't require it, but we'll show it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[4.3] Feature Scaling Applied (for reference)")

# ============================================================================
# 5. RANDOM FOREST MODEL
# ============================================================================
print("\n" + "=" * 80)
print("[5] RANDOM FOREST MODEL TRAINING")
print("=" * 80)

# Initialize Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("\n[5.1] Model Parameters:")
print(f"  - n_estimators: {rf_model.n_estimators}")
print(f"  - max_depth: {rf_model.max_depth}")
print(f"  - min_samples_split: {rf_model.min_samples_split}")
print(f"  - min_samples_leaf: {rf_model.min_samples_leaf}")
print(f"  - random_state: {rf_model.random_state}")

# Train the model
print("\n[5.2] Training Random Forest Model...")
rf_model.fit(X_train, y_train)
print("Training completed!")

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n" + "=" * 80)
print("[6] MODEL EVALUATION")
print("=" * 80)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n[6.1] Training Set Metrics:")
print(f"  - Mean Squared Error (MSE): {train_mse:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"  - Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"  - R² Score: {train_r2:.4f}")

print("\n[6.2] Test Set Metrics:")
print(f"  - Mean Squared Error (MSE): {test_mse:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"  - Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"  - R² Score: {test_r2:.4f}")

# Cross-validation
print("\n[6.3] Cross-Validation (5-fold):")
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, 
                            scoring='neg_mean_squared_error')
cv_rmse_scores = np.sqrt(-cv_scores)
print(f"  - CV RMSE Scores: {cv_rmse_scores}")
print(f"  - Mean CV RMSE: {cv_rmse_scores.mean():.4f}")
print(f"  - Std CV RMSE: {cv_rmse_scores.std():.4f}")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("[7] FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance (sorted):")
print(feature_importance.to_string(index=False))

# Visualize feature importance
fig4, ax = plt.subplots(figsize=(10, 6))
feature_importance.plot(x='feature', y='importance', kind='barh', ax=ax, 
                        color='steelblue', legend=False)
ax.set_title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_ylabel('Features')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('california_housing_feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: california_housing_feature_importance.png")
plt.close()

# ============================================================================
# 8. PREDICTION VISUALIZATION
# ============================================================================
print("\n[8] Creating Prediction Visualizations...")

# Actual vs Predicted
fig5, axes = plt.subplots(1, 2, figsize=(15, 5))

# Training set
axes[0].scatter(y_train, y_train_pred, alpha=0.5, s=20)
axes[0].plot([y_train.min(), y_train.max()], 
             [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Values', fontsize=11)
axes[0].set_ylabel('Predicted Values', fontsize=11)
axes[0].set_title(f'Training Set: Actual vs Predicted (R² = {train_r2:.4f})', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Test set
axes[1].scatter(y_test, y_test_pred, alpha=0.5, s=20, color='green')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Values', fontsize=11)
axes[1].set_ylabel('Predicted Values', fontsize=11)
axes[1].set_title(f'Test Set: Actual vs Predicted (R² = {test_r2:.4f})', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('california_housing_predictions.png', dpi=300, bbox_inches='tight')
print("Saved: california_housing_predictions.png")
plt.close()

# Residual Plot
fig6, axes = plt.subplots(1, 2, figsize=(15, 5))

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

axes[0].scatter(y_train_pred, train_residuals, alpha=0.5, s=20)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Values', fontsize=11)
axes[0].set_ylabel('Residuals', fontsize=11)
axes[0].set_title('Training Set: Residual Plot', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_pred, test_residuals, alpha=0.5, s=20, color='green')
axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Values', fontsize=11)
axes[1].set_ylabel('Residuals', fontsize=11)
axes[1].set_title('Test Set: Residual Plot', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('california_housing_residuals.png', dpi=300, bbox_inches='tight')
print("Saved: california_housing_residuals.png")
plt.close()

# ============================================================================
# 9. SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("[9] SAMPLE PREDICTIONS")
print("=" * 80)

# Select random samples from test set
sample_indices = np.random.choice(len(X_test), size=10, replace=False)
sample_X = X_test.iloc[sample_indices]
sample_y_actual = y_test.iloc[sample_indices]
sample_y_pred = rf_model.predict(sample_X)

results_df = pd.DataFrame({
    'Actual': sample_y_actual.values,
    'Predicted': sample_y_pred,
    'Error': np.abs(sample_y_actual.values - sample_y_pred)
})
results_df.index = sample_indices

print("\nSample Predictions (10 random test samples):")
print(results_df.to_string())
print(f"\nAverage Absolute Error: {results_df['Error'].mean():.4f}")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[10] SUMMARY")
print("=" * 80)
print(f"""
Model Performance Summary:
  - Test RMSE: {test_rmse:.4f} (in $100,000s)
  - Test MAE: {test_mae:.4f} (in $100,000s)
  - Test R²: {test_r2:.4f}
  - Cross-Validation RMSE: {cv_rmse_scores.mean():.4f} ± {cv_rmse_scores.std():.4f}

Most Important Features:
  1. {feature_importance.iloc[0]['feature']}: {feature_importance.iloc[0]['importance']:.4f}
  2. {feature_importance.iloc[1]['feature']}: {feature_importance.iloc[1]['importance']:.4f}
  3. {feature_importance.iloc[2]['feature']}: {feature_importance.iloc[2]['importance']:.4f}

Visualizations saved:
  - california_housing_eda.png
  - california_housing_scatter.png
  - california_housing_correlation.png
  - california_housing_feature_importance.png
  - california_housing_predictions.png
  - california_housing_residuals.png
""")

print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
