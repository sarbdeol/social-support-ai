from fastapi import APIRouter
from api.schemas import ApplicantInput, PredictionOutput
from ml.predict import predict

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok", "service": "Social Support AI API"}

@router.post("/predict", response_model=PredictionOutput)
def predict_eligibility(applicant: ApplicantInput):
    result = predict(applicant.model_dump())
    return result