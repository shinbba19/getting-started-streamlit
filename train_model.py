# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# -----------------------
# 1. Load Data
# -----------------------
df = pd.read_csv("Price per sqm_cleaned_data_selection2.csv")

# -----------------------
# 2. Features & Target
# -----------------------
# X = everything except target
X = df.drop(columns=["price_sqm"])
y = df["price_sqm"]

# -----------------------
# 3. Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------
# 4. Train Model
# -----------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -----------------------
# 5. Evaluate Model
# -----------------------
y_pred = model.predict(X_test)

print("✅ Model Performance:")
print(f"R²:   {r2_score(y_test, y_pred):.2f}")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):,.0f} THB/sqm")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f} THB/sqm")

# -----------------------
# 6. Save Model
# -----------------------
joblib.dump(model, "condo_price_model.pkl")
print("💾 Model saved as condo_price_model.pkl")
