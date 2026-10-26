
import os
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "campusguard_phishing_model.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Trained model not found at {MODEL_PATH}."
    )

model = joblib.load(MODEL_PATH)
app = FastAPI(title="CampusGuard ML API", version="1.0.0")


class EmailRequest(BaseModel):
    subject: str = ""
    body: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(email: EmailRequest):
    text = f"{email.subject or ''} {email.body or ''}".strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Email subject or body is required",
        )

    prediction = int(model.predict([text])[0])

    # Label 1 = phishing; label 0 = legitimate.
    # Linear SVM decision scores are not probabilities.
    decision = float(model.decision_function([text])[0])

    # A transparent heuristic score, not a calibrated probability.
    risk_score = max(0, min(100, round(50 + decision * 10)))

    if prediction == 1:
        status = "PHISHING" if risk_score >= 70 else "SUSPICIOUS"
    else:
        status = "SAFE"

    return {
        "prediction": "phishing" if prediction == 1 else "safe",
        "status": status,
        "riskScore": risk_score,
    }
