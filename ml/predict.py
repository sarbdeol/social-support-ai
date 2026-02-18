import joblib
import numpy as np
import pandas as pd

model         = joblib.load("ml/artifacts/model.joblib")
explainer     = joblib.load("ml/artifacts/shap_explainer.joblib")
le_employment = joblib.load("ml/artifacts/le_employment.joblib")
le_housing    = joblib.load("ml/artifacts/le_housing.joblib")
FEATURES      = joblib.load("ml/artifacts/features.joblib")


def predict(applicant: dict) -> dict:
    emp = le_employment.transform([applicant['employment_status']])[0]
    hou = le_housing.transform([applicant['housing_type']])[0]

    row = pd.DataFrame([{
        'monthly_income':     applicant['monthly_income'],
        'monthly_expenses':   applicant['monthly_expenses'],
        'family_size':        applicant['family_size'],
        'dependents':         applicant['dependents'],
        'total_assets':       applicant['total_assets'],
        'total_liabilities':  applicant['total_liabilities'],
        'net_worth':          applicant['total_assets'] - applicant['total_liabilities'],
        'credit_score':       applicant['credit_score'],
        'years_employed':     applicant['years_employed'],
        'employment_status':  emp,
        'housing_type':       hou
    }])

    prediction = model.predict(row)[0]
    proba      = model.predict_proba(row)[0]
    confidence = round(float(max(proba)) * 100, 1)

    # SHAP — handle both old and new shap output formats
    shap_vals = explainer.shap_values(row)

    if isinstance(shap_vals, list):
        # old format: list of arrays per class
        raw = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
    else:
        # new format: single 3D array (n_samples, n_features, n_classes)
        if shap_vals.ndim == 3:
            raw = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
        else:
            raw = shap_vals[0]

    explanation = {feat: round(float(val), 4) for feat, val in zip(FEATURES, raw)}

    return {
        "decision":    "APPROVE" if prediction == 1 else "DECLINE",
        "confidence":  confidence,
        "explanation": explanation
    }