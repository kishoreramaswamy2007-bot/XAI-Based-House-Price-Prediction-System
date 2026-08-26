import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "california_housing.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
target = "median_house_value"
X = df.drop(columns=[target])
y = df[target]

numeric = [c for c in X.columns if c != "ocean_proximity"]
categorical = ["ocean_proximity"]

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train_t = preprocessor.fit_transform(X_train)
X_test_t = preprocessor.transform(X_test)

feature_names = numeric + list(
    preprocessor.named_transformers_["cat"]
    .named_steps["onehot"].get_feature_names_out(categorical)
)

model = RandomForestRegressor(
    n_estimators=35, max_depth=12, random_state=42, n_jobs=-1
)
model.fit(X_train_t, y_train)

predictions = model.predict(X_test_t)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.4f}\n")

# Global feature importance
importance = pd.Series(
    model.feature_importances_, index=feature_names
).sort_values()
plt.figure(figsize=(9, 6))
importance.tail(12).plot(kind="barh")
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# SHAP global explanation
explainer = shap.TreeExplainer(model)
X_shap = X_test_t[:100]
shap_values = explainer.shap_values(X_shap)
shap.summary_plot(
    shap_values, X_shap, feature_names=feature_names,
    max_display=10, show=False
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# SHAP local explanation
explanation = explainer(X_shap[:1])[0]
shap.plots.waterfall(explanation, max_display=10, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# PDP for median income
income_index = numeric.index("median_income")
fig, ax = plt.subplots(figsize=(8, 5))
PartialDependenceDisplay.from_estimator(
    model, X_test_t, [income_index],
    feature_names=feature_names, ax=ax
)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pdp_median_income.png"), dpi=150)
plt.close()

pd.DataFrame({
    "actual": y_test.iloc[:20].values,
    "predicted": predictions[:20]
}).to_csv(os.path.join(OUTPUT_DIR, "sample_predictions.csv"), index=False)

print("Project completed.")
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
