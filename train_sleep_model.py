import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle

# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\markl\ARDUINO-TEAM_SINCO\p1classification_model\regress\sleep\Sleep_health_and_lifestyle_dataset.csv")

# ── 2. Preprocess ────────────────────────────────────────────────────────────
df[["BP_Systolic", "BP_Diastolic"]] = (
    df["Blood Pressure"].str.split("/", expand=True).astype(int)
)
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df = df.drop(columns=["Person ID", "Blood Pressure"])
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Occupation", "BMI Category", "Sleep Disorder"],
    drop_first=True,
)

# ── 3. Split features / target ───────────────────────────────────────────────
X = df_encoded.drop(columns=["Sleep Duration"])
y = df_encoded["Sleep Duration"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Scale features ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Train ─────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)

# ── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)

print("=== Model Performance ===")
print(f"R²   : {r2_score(y_test, y_pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f} hours")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} hours")

# ── 7. Feature coefficients ──────────────────────────────────────────────────
coef_df = (
    pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    .assign(Abs=lambda d: d["Coefficient"].abs())
    .sort_values("Abs", ascending=False)
    .drop(columns="Abs")
)

print("\n=== Top 10 Feature Coefficients (standardized) ===")
print(coef_df.head(10).to_string(index=False))

# ── 8. Save model & scaler ───────────────────────────────────────────────────
with open("sleep_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("sleep_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("sleep_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("\n✅ Saved: sleep_model.pkl, sleep_scaler.pkl, sleep_columns.pkl")