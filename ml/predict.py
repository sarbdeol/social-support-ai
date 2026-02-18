import joblib
import numpy as np
import pandas as pd

# Load all artifacts once at startup
model         = joblib.load("ml/artifacts/model.joblib")
explainer     = joblib.load("ml/artifacts/shap_explainer.joblib")
le_employment = joblib.load("ml/artifacts/le_employment.joblib")
le_housing    = joblib.load("ml/artifacts/le_housing.joblib")
FEATURES      = joblib.load("ml/artifacts/features.joblib")


def predict(applicant: dict) -> dict:
    """
    Takes applicant dict, returns prediction + confidence + SHAP explanation
    """
    # Encode categoricals
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

    prediction  = model.predict(row)[0]
    proba       = model.predict_proba(row)[0]
    confidence  = round(float(max(proba)) * 100, 1)

    # SHAP values for this prediction
    shap_vals   = explainer.shap_values(row)
    shap_row    = shap_vals[1][0] if prediction == 1 else shap_vals[0][0]
    explanation = dict(zip(FEATURES, [round(float(v), 4) for v in shap_row]))

    return {
        "decision":    "APPROVE" if prediction == 1 else "DECLINE",
        "confidence":  confidence,
        "explanation": explanation  # SHAP values per feature
    }
