import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset.csv")   # Change filename if needed

print(df.head())
print(df.info())

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Remove duplicates
df.drop_duplicates(inplace=True)

# ==========================================
# CONVERT TIMESTAMP
# ==========================================

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Feature Engineering
df["Year"] = df["timestamp"].dt.year
df["Month"] = df["timestamp"].dt.month
df["Day"] = df["timestamp"].dt.day
df["Hour"] = df["timestamp"].dt.hour

# ==========================================
# ENCODE CATEGORICAL (if any)
# ==========================================

encoder = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = encoder.fit_transform(df[col])

# ==========================================
# TARGET COLUMN
# ==========================================

target = "output_amount"

X = df.drop(columns=[target, "timestamp"])
y = df[target]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

pred = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

print("\nModel Performance")
print("MAE :", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2 Score:", r2_score(y_test, pred))

# ==========================================
# ACTUAL VS PREDICTED
# ==========================================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": pred
})

print(comparison.head(10))

# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Samples")
plt.ylabel(target)
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# FUTURE FORECAST
# ==========================================

future = X.tail(20)

future_pred = model.predict(future)

plt.figure(figsize=(10,5))
plt.plot(future_pred, marker="o")
plt.title("Future Trend Forecast")
plt.xlabel("Future Samples")
plt.ylabel(target)
plt.grid(True)
plt.show()

print("\nFuture Predictions")
print(future_pred)
plt.show()