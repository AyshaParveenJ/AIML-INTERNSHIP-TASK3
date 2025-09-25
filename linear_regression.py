import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# STEP 1: Import, Preprocess, and Select Data
# =========================================================
print("--- Starting Linear Regression Task ---")

# 1. Load the dataset (Ensure 'Housing.csv' is in the same folder)
try:
    # Use 'Housing.csv' as you specified
    df = pd.read_csv('Housing.csv')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: File not found. Check the filename and path.")
    exit()

# 2. Inspect and Select Columns 
# Assuming standard column names for the Housing dataset: 'area' and 'price'
FEATURE_COLUMNS = ['area']  # Feature(s) - Must be 2D (DataFrame)
TARGET_COLUMN = 'price'     # Target - Must be 1D (Series)

# Check if selected columns exist and filter the DataFrame
if TARGET_COLUMN not in df.columns or not all(col in df.columns for col in FEATURE_COLUMNS):
    print("\n❌ ERROR: Column names are incorrect.")
    print("Available Columns:", df.columns.tolist())
    print(f"Please change FEATURE_COLUMNS and TARGET_COLUMN to match your data.")
    exit()

# Basic Data Cleaning: Drop rows where the target or feature is missing
df_clean = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)

X = df_clean[FEATURE_COLUMNS]  
y = df_clean[TARGET_COLUMN]    

print(f"Data cleaned and selected. Target: {TARGET_COLUMN}, Feature: {FEATURE_COLUMNS[0]}")
print(f"Total samples used: {len(X)}")

# =========================================================
# STEP 2: Split Data into Train-Test Sets
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)
print(f"\nTraining set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

# =========================================================
# STEP 3: Fit a Linear Regression Model
# =========================================================
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model trained and predictions made.")

# =========================================================
# STEP 4: Evaluate Model (MAE, MSE, R²)
# =========================================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Root Mean Squared Error (RMSE): ${np.sqrt(mse):,.2f}")
print(f"R-squared (R²): {r2:.4f} (Closer to 1.0 is better)")

# =========================================================
# STEP 5: Plot Regression Line and Interpret Coefficients
# =========================================================
print("\n--- Coefficients Interpretation ---")
print(f"Intercept (c, Base Price): ${model.intercept_:.2f}")
print(f"Coefficient ({FEATURE_COLUMNS[0]}): ${model.coef_[0]:.2f}")
print(f"Interpretation: For every 1 unit increase in {FEATURE_COLUMNS[0]}, the predicted price increases by ${model.coef_[0]:.2f}")


# Plotting the result for Simple Linear Regression
if len(FEATURE_COLUMNS) == 1:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Prices (Test Data)')
    plt.plot(X_test, y_pred, color='red', linewidth=3, label='Regression Line (Predicted Prices)')
    plt.title(f'Linear Regression: Price vs. Area')
    plt.xlabel(FEATURE_COLUMNS[0].capitalize())
    plt.ylabel(TARGET_COLUMN.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

# =========================================================
print("\n--- Task Complete ---")