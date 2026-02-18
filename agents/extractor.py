from langchain_community.llms import Ollama
import json

llm = Ollama(model="llama3")

def extract_data(state: dict) -> dict:
    """
    Extracts and structures applicant data from raw form inputs + documents.
    In production this would OCR images and parse PDFs.
    For now it structures the chat form data using LLM.
    """
    applicant_raw = state.get("applicant_raw", {})

    prompt = f"""
You are a data extraction agent for a government social support system.
Extract and structure the following applicant information into valid JSON.
Only return the JSON object, nothing else.

Raw applicant data:
{json.dumps(applicant_raw, indent=2)}

Return this exact JSON structure:
{{
    "monthly_income": <float>,
    "monthly_expenses": <float>,
    "family_size": <int>,
    "dependents": <int>,
    "total_assets": <float>,
    "total_liabilities": <float>,
    "credit_score": <int>,
    "years_employed": <int>,
    "employment_status": <"employed"|"unemployed"|"part_time"|"self_employed">,
    "housing_type": <"owned"|"rented"|"family"|"government">,
    "name": <string>,
    "nationality": <string>
}}
"""
    response = llm.invoke(prompt)

    # Parse LLM response to JSON
    try:
        # Find JSON block in response
        start = response.find('{')
        end   = response.rfind('}') + 1
        extracted = json.loads(response[start:end])
    except Exception:
        # Fallback: use raw data directly
        extracted = applicant_raw

    state["extracted_data"] = extracted
    state["messages"].append(f"✅ Data extraction complete — {len(extracted)} fields extracted")
    return state