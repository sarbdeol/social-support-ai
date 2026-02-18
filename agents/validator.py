from langchain_community.llms import Ollama
import json

llm = Ollama(model="llama3")

def validate_data(state: dict) -> dict:
    """
    Cross-checks extracted data for inconsistencies and flags issues.
    Uses Reflexion pattern — checks, finds issues, re-evaluates.
    """
    data = state.get("extracted_data", {})

    prompt = f"""
You are a data validation agent for a government social support system.
Check the following applicant data for inconsistencies or red flags.
Only return a JSON object, nothing else.

Applicant data:
{json.dumps(data, indent=2)}

Check for:
1. monthly_expenses > monthly_income * 1.5 (severe overspending)
2. credit_score < 300 or > 850 (invalid range)
3. dependents > family_size (impossible)
4. total_liabilities > total_assets * 3 (extreme debt)

Return this exact JSON:
{{
    "is_valid": <true|false>,
    "flags": [<list of issue strings, empty if none>],
    "risk_level": <"low"|"medium"|"high">
}}
"""
    response = llm.invoke(prompt)

    try:
        start      = response.find('{')
        end        = response.rfind('}') + 1
        validation = json.loads(response[start:end])
    except Exception:
        # Fallback validation
        flags = []
        if data.get('monthly_expenses', 0) > data.get('monthly_income', 0) * 1.5:
            flags.append("Expenses significantly exceed income")
        if data.get('dependents', 0) > data.get('family_size', 0):
            flags.append("Dependents exceed family size")
        validation = {
            "is_valid":   len(flags) == 0,
            "flags":      flags,
            "risk_level": "high" if len(flags) > 1 else "medium" if flags else "low"
        }

    state["validation"] = validation
    flag_count = len(validation.get("flags", []))
    state["messages"].append(
        f"✅ Validation complete — Risk: {validation.get('risk_level','unknown').upper()} | "
        f"Flags: {flag_count}"
    )
    return state