import httpx
import json

FASTAPI_URL = "http://localhost:8000/api/v1/predict"

def check_eligibility(state: dict) -> dict:
    """
    Calls the FastAPI ML endpoint to get eligibility prediction + SHAP values.
    """
    data = state.get("extracted_data", {})

    try:
        response = httpx.post(FASTAPI_URL, json=data, timeout=30)
        result   = response.json()
    except Exception as e:
        result = {
            "decision":    "DECLINE",
            "confidence":  0.0,
            "explanation": {},
            "error":       str(e)
        }

    state["ml_result"] = result
    state["messages"].append(
        f"✅ Eligibility check complete — "
        f"Decision: {result['decision']} | "
        f"Confidence: {result['confidence']}%"
    )
    return state