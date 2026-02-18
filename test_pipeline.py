from agents.orchestrator import run_pipeline

test_applicant = {
    "name":               "Ahmed Al Mansouri",
    "nationality":        "Emirati",
    "monthly_income":     1200,
    "monthly_expenses":   1400,
    "family_size":        5,
    "dependents":         3,
    "total_assets":       5000,
    "total_liabilities":  8000,
    "credit_score":       420,
    "years_employed":     0,
    "employment_status":  "unemployed",
    "housing_type":       "rented"
}

print("🚀 Running pipeline...\n")
result = run_pipeline(test_applicant)

print("── Agent Messages ──")
for msg in result["messages"]:
    print(msg)

print("\n── Final Output ──")
import json
print(json.dumps(result["final_output"], indent=2))
