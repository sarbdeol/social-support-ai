from pydantic import BaseModel
from typing import Optional

class ApplicantInput(BaseModel):
    monthly_income:     float
    monthly_expenses:   float
    family_size:        int
    dependents:         int
    total_assets:       float
    total_liabilities:  float
    credit_score:       int
    years_employed:     int
    employment_status:  str   # employed / unemployed / part_time / self_employed
    housing_type:       str   # owned / rented / family / government

class PredictionOutput(BaseModel):
    decision:    str
    confidence:  float
    explanation: dict