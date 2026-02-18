import pandas as pd
import numpy as np
import joblib
import os
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── LOAD DATA ──────────────────────────────────────────
df = pd.read_csv("data/synthetic/applicants.csv")

# ── FEATURES ───────────────────────────────────────────
FEATURES = [
    'monthly_income', 'monthly_expenses', 'family_size',
    'dependents', 'total_assets', 'total_liabilities',
    'net_worth', 'credit_score', 'years_employed',
    'employment_status', 'housing_type'
]

# Encode categorical columns
le_employment = LabelEncoder()
le_housing    = LabelEncoder()

df['employment_status'] = le_employment.fit_transform(df['employment_status'])
df['housing_type']      = le_housing.fit_transform(df['housing_type'])

X = df[FEATURES]
y = (df['label'] == 'APPROVE').astype(int)  # 1 = APPROVE, 0 = DECLINE

# ── TRAIN / TEST SPLIT ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── TRAIN MODEL ────────────────────────────────────────
print("Training RandomForest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'   # handles imbalanced labels
)
model.fit(X_train, y_train)

# ── EVALUATE ───────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['DECLINE', 'APPROVE']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.2%}")

# ── SHAP EXPLAINABILITY ────────────────────────────────
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature importance summary
feature_importance = pd.DataFrame({
    'feature':   FEATURES,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Feature Importances:")
print(feature_importance.to_string(index=False))

# ── SAVE EVERYTHING ────────────────────────────────────
os.makedirs("ml/artifacts", exist_ok=True)

joblib.dump(model,         "ml/artifacts/model.joblib")
joblib.dump(explainer,     "ml/artifacts/shap_explainer.joblib")
joblib.dump(le_employment, "ml/artifacts/le_employment.joblib")
joblib.dump(le_housing,    "ml/artifacts/le_housing.joblib")
joblib.dump(FEATURES,      "ml/artifacts/features.joblib")

print("\n✅ Saved ml/artifacts/model.joblib")
print("✅ Saved ml/artifacts/shap_explainer.joblib")
print("✅ Model training complete!")