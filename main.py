import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load the CSV
df = pd.read_csv("car77_timecards.csv")

# Clean column names (remove extra spaces, unify formatting)
df.columns = df.columns.str.strip()

# Manually convert LAP_TIME to seconds
def lap_time_to_seconds(t):
    try:
        mins, secs = t.split(":")
        return int(mins) * 60 + float(secs)
    except Exception:
        return None

df["LAP_TIME_SEC"] = df["LAP_TIME"].apply(lap_time_to_seconds)

# Filter out rows with missing or invalid lap times
df = df[df["LAP_TIME_SEC"].notna()]

# Define features to use
features = ["LAP_NUMBER", "TOP_SPEED", "S1", "S2", "S3", "PIT_TIME"]

# Debug: Show available columns
print("Available columns:", list(df.columns))

# Ensure numeric conversion and fill missing values
for col in features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        print(f"Column '{col}' not found!")

# Define feature matrix and target
X = df[features]
y = df["LAP_TIME_SEC"]

# Debug outputs
print("Feature matrix shape:", X.shape)
print("First few rows of X:\n", X.head())
print("First few target values:\n", y.head())

# Safety check
if X.shape[0] == 0:
    raise ValueError("Feature matrix is empty. Check column values or data filtering.")

# Train/test split. If you train and test on the same data, the model might just memorize values instead of learning patterns. test set gives us a reality check on how well the model is performing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel MSE: {mse:.4f}")

# --- Feature importance output ---
importances = model.feature_importances_
feature_ranking = sorted(zip(importances, features), reverse=True)

print("\n📊 Feature importance chart:")
for i, (importance, name) in enumerate(feature_ranking, 1):
    print(f"{i}. {name} ({importance:.4f})")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted")
plt.title("Actual vs. Predicted Lap Times")
plt.xlabel("Sample")
plt.ylabel("Lap Time (seconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
